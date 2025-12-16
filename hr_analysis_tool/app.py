from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
HERE = app.root_path

MAX_EXPENSE = 1.0
MAX_DISTANCE = 1.0
MAX_TIME = 1.0

rf_classifier = None
clustering_columns = []
normalization_params = None

def here(path):
    return f"{HERE}/{path}"

try:
    rf_classifier = joblib.load(here('rf_classifier.pkl'))
    clustering_columns = joblib.load(here('clustering_columns.pkl'))
    normalization_params = joblib.load(here('normalization_params.pkl'))

    MAX_EXPENSE = normalization_params['max_expense']
    MAX_DISTANCE = normalization_params['max_distance']
    MAX_TIME = normalization_params['max_time']

    print("Model, columns, and normalization parameters loaded successfully.")
except Exception as e:
    print(f"Error loading model, columns, or normalization parameters: {e}")

# --- Cluster profiles ---
cluster_profiles = {
    0: {
        'name': 'Long-Distance Commuters',
        'risk': 'Low Risk',
        'avg_absence': 5.25
    },
    1: {
        'name': 'Experienced Urban Workers', 
        'risk': 'High Risk',
        'avg_absence': 8.36
    },
    2: {
        'name': 'Young Family-Oriented',
        'risk': 'Moderate Risk',
        'avg_absence': 6.88
    }
}

def predict_cluster(features):
    # Calculate Commute_Burden_Index
    features['Commute_Burden_Index'] = (
        features['Transportation expense'] / MAX_EXPENSE * 0.3 +
        features['Distance from Residence to Work'] / MAX_DISTANCE * 0.35 +
        features['Estimated commute time'] / MAX_TIME * 0.35
    )

    # Calculate Home_Responsibility_Index
    features['Home_Responsibility_Index'] = (
        features['Number of children'] + features['Number of pets']
    )

    # Convert to ordered list (same order as clustering_columns)
    row = [features[col] for col in clustering_columns]
    X = [row]

    # Predict cluster
    cluster_id = rf_classifier.predict(X)[0]
    return cluster_id

# --- Flask route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Convert form inputs to float
            employee_features = {
                key: float(request.form[key])
                for key in request.form
                if key in [
                    'Transportation expense',
                    'Distance from Residence to Work',
                    'Estimated commute time',
                    'Age', 'Years until retirement', 'Service time',
                    'Disciplinary failure', 'Education',
                    'Number of children', 'Social drinker', 'Social smoker',
                    'Number of pets'
                ]
            }

            cluster_id = predict_cluster(employee_features)
            profile = cluster_profiles[cluster_id]
            return render_template('result.html', cluster=cluster_id, profile=profile)

        except Exception as e:
            print(f"Error in POST /: {e}")
            return render_template('form.html', error=str(e)), 500

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
