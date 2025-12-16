from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

try:
    rf_classifier = joblib.load('rf_classifier.pkl')
    clustering_columns = joblib.load('clustering_columns.pkl')
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model or columns: {e}")

# Cluster profiles
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
    # Convert to DataFrame and ensure correct column order

    #TODO: do the math for the commute burden and home responsibility index.
    
    employee_df = pd.DataFrame([features])
    employee_df = employee_df[clustering_columns]
    cluster_id = rf_classifier.predict(employee_df)[0]
    return cluster_id

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        employee_features = {key: float(request.form[key]) for key in request.form}
        cluster_id = predict_cluster(employee_features)
        profile = cluster_profiles[cluster_id]
        return render_template('result.html', cluster=cluster_id, profile=profile)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
