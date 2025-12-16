from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- Absolute path helper: point to files in this folder ---
HERE = os.path.dirname(os.path.abspath(__file__))

def here(path):
    return os.path.join(HERE, path)

# --- Model loading with detailed logging ---
print(f"🌟 Running app.py from: {HERE}")
print(f"🌟 Looking for files in: {os.listdir(HERE) if os.path.exists(HERE) else 'Directory not found'}")

# --- Variables ---
MAX_EXPENSE = 1.0
MAX_DISTANCE = 1.0
MAX_TIME = 1.0

rf_classifier = None
clustering_columns = None
normalization_params = None

# --- Load model files ---
try:
    # Load rf_classifier
    rf_path = here('rf_classifier.pkl')
    print(f"🔍 Checking rf_classifier.pkl at: {rf_path}")
    if not os.path.exists(rf_path):
        print(f"❌ rf_classifier.pkl NOT FOUND at: {rf_path}")
    else:
        rf_classifier = joblib.load(rf_path)
        print(f"✅ rf_classifier loaded from: {rf_path}")

    # Load clustering_columns
    cols_path = here('clustering_columns.pkl')
    print(f"🔍 Checking clustering_columns.pkl at: {cols_path}")
    if not os.path.exists(cols_path):
        print(f"❌ clustering_columns.pkl NOT FOUND at: {cols_path}")
    else:
        clustering_columns = joblib.load(cols_path)
        print(f"✅ clustering_columns loaded from: {cols_path}")

    # Load normalization_params
    norm_path = here('normalization_params.pkl')
    print(f"🔍 Checking normalization_params.pkl at: {norm_path}")
    if not os.path.exists(norm_path):
        print(f"❌ normalization_params.pkl NOT FOUND at: {norm_path}")
    else:
        normalization_params = joblib.load(norm_path)
        print(f"✅ normalization_params loaded from: {norm_path}")

    # Update MAX_ values only if normalization_params exists
    if normalization_params is not None:
        MAX_EXPENSE = normalization_params.get('max_expense', 1.0)
        MAX_DISTANCE = normalization_params.get('max_distance', 1.0)
        MAX_TIME = normalization_params.get('max_time', 1.0)

    print("🎉 All models and parameters loaded successfully.")

except Exception as e:
    print(f"❌ Error loading models: {type(e).__name__}: {e}")

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

# --- Prediction function ---
def predict_cluster(features):
    # Guard: check if model is loaded
    if rf_classifier is None:
        print("🛑 predict_cluster: rf_classifier is None")
        raise RuntimeError("Model file 'rf_classifier.pkl' not loaded; check logs")
    if clustering_columns is None:
        print("🛑 predict_cluster: clustering_columns is None")
        raise RuntimeError("Model file 'clustering_columns.pkl' not loaded; check logs")

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

    # Convert to DataFrame and ensure correct column order
    employee_df = pd.DataFrame([features])
    employee_df = employee_df[clustering_columns]
    cluster_id = rf_classifier.predict(employee_df)[0]
    return cluster_id

# --- Flask route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Convert form inputs to float
            employee_features = {key: float(request.form[key]) for key in request.form}
            cluster_id = predict_cluster(employee_features)
            profile = cluster_profiles[cluster_id]
            return render_template('result.html', cluster=cluster_id, profile=profile)

        except Exception as e:
            print(f"Error in POST /: {e}")
            return render_template('form.html', error=str(e)), 500

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
