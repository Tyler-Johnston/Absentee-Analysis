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

def here(path):
    return f"{HERE}/{path}"

try:
    rf_classifier = joblib.load(here('rf_classifier.pkl'))
    clustering_columns = joblib.load(here('clustering_columns.pkl'))
    normalization_params = joblib.load(here('normalization_params.pkl'))

    max_expense = normalization_params['max_expense']
    max_distance = normalization_params['max_distance']
    max_time = normalization_params['max_time']

    print("Model, columns, and normalization parameters loaded successfully.")
except Exception as e:
    print(f"Error loading model, columns, or normalization parameters: {e}")

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
    # calculate Commute_Burden_Index
    features['Commute_Burden_Index'] = (
        features['Transportation expense'] / max_expense * 0.3 +
        features['Distance from Residence to Work'] / max_distance * 0.35 +
        features['Estimated commute time'] / max_time * 0.35
    )

    # calculate Home_Responsibility_Index
    features['Home_Responsibility_Index'] = (
        features['Number of children'] + features['Number of pets']
    )

    # make sure all clustering columns are present
    missing_cols = [col for col in clustering_columns if col not in features]
    if missing_cols:
        print(f"Missing columns in features: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # convert to ordered list (same order as clustering_columns)
    try:
        row = [features[col] for col in clustering_columns]
        X = [row] # (a list of one row)

        cluster_id = rf_classifier.predict(X)[0]
        return cluster_id

    except Exception as e:
        print(f"Error in predict_cluster: {e}")
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # convert form inputs to float
            employee_features = {
                key: float(request.form[key])
                for key in request.form
            }

            # predict cluster
            cluster_id = predict_cluster(employee_features)
            profile = cluster_profiles[cluster_id]
            return render_template('result.html', cluster=cluster_id, profile=profile)

        except Exception as e:
            print(f"Error in POST /: {e}")
            return render_template('form.html', error=str(e)), 500

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
