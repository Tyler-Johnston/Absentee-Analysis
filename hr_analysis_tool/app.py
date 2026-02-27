from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
HERE = app.root_path

max_expense = 1.0
max_distance = 1.0
max_time = 1.0

rf_classifier = None
clustering_columns = []
normalization_params = None
scaler = None 

def here(path):
    return f"{HERE}/{path}"

try:
    rf_classifier = joblib.load(here('rf_classifier.pkl'))
    clustering_columns = joblib.load(here('clustering_columns.pkl'))
    normalization_params = joblib.load(here('normalization_params.pkl'))
    scaler = joblib.load(here('scaler.pkl'))

    max_expense = normalization_params['max_expense']
    max_distance = normalization_params['max_distance']
    max_time = normalization_params['max_time']
    print("Model, scaler, columns, and normalization parameters loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")

cluster_profiles = {
    0: {
        'name': 'Long-Distance Commuters',
        'risk': 'Low Risk',
        'avg_absence': 5.40,
    },
    1: {
        'name': 'Experienced Urban Workers',
        'risk': 'High Risk',
        'avg_absence': 8.46,
    },
    2: {
        'name': 'Young Family-Oriented',
        'risk': 'Moderate Risk',
        'avg_absence': 7.04,
    },
}

def predict_cluster(features):
    # calculate engineered features
    features['Commute_Burden_Index'] = (
        features['Transportation expense'] / max_expense * 0.3 +
        features['Distance from Residence to Work'] / max_distance * 0.35 +
        features['Estimated commute time'] / max_time * 0.35
    )
    
    features['Home_Responsibility_Index'] = (
        features['Number of children'] + features['Number of pets']
    )

    # Check for missing columns
    missing_cols = [col for col in clustering_columns if col not in features]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    try:
        features_to_scale = [
            'Transportation expense', 'Distance from Residence to Work',
            'Service time', 'Age', 'Home_Responsibility_Index',
            'Commute_Burden_Index', 'Body mass index'
        ]
        
        # extract values that need scaling
        values_to_scale = [[features[col] for col in features_to_scale]]
        
        # scale using the loaded scaler (returns 2D array-like)
        scaled_values = scaler.transform(values_to_scale)[0]  # Get first row
        
        # create dictionary of scaled values
        scaled_dict = dict(zip(features_to_scale, scaled_values))
        
        # build final row: use scaled values where applicable, raw otherwise
        row = []
        for col in clustering_columns:
            if col in scaled_dict:
                row.append(scaled_dict[col])
            else:
                row.append(features[col])
        
        # Predict with scaled data (wrap in list for sklearn)
        cluster_id = rf_classifier.predict([row])[0]
        
        print(f"✓ Prediction successful: Cluster {cluster_id}")
        print(f"  Raw: Age={features.get('Age')}, Distance={features.get('Distance from Residence to Work')}")
        print(f"  Scaled sample: {row[:3]}")
        
        return cluster_id

    except Exception as e:
        print(f"Error in predict_cluster: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Convert form inputs to float
            employee_features = {
                key: float(request.form[key])
                for key in request.form
            }

            # Predict cluster
            cluster_id = predict_cluster(employee_features)
            profile = cluster_profiles[cluster_id]
            return render_template('result.html', cluster=cluster_id, profile=profile)

        except Exception as e:
            print(f"Error in POST /: {e}")
            return render_template('form.html', error=str(e)), 500

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
