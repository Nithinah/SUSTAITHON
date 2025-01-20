from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import os
if not os.path.exists("E:\\VIT_Hackathon\\stress_anxiety_depression_model_10_layers.h5"):
    print("Model file not found!")
else:
    model = tf.keras.models.load_model("E:\\VIT_Hackathon\\stress_anxiety_depression_model_10_layers.h5")


# Initialize the Flask app
app = Flask(__name__)

# Predefined thresholds and labels
stress_thresholds = [13, 26]
stress_labels = ['Low Stress', 'Moderate Stress', 'High Stress']
anxiety_thresholds = [4, 9, 14]
anxiety_labels = ['Minimal Anxiety', 'Mild Anxiety', 'Moderate Anxiety', 'Severe Anxiety']
depression_thresholds = [0, 4, 9, 14, 19]
depression_labels = ['No Depression', 'Minimal Depression', 'Mild Depression', 'Moderate Depression', 'Moderately Severe Depression', 'Severe Depression']

# Helper function to categorize predictions
def categorize(value, thresholds, labels):
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return labels[i]
    return labels[-1]

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        input_data = {
            'Age': int(request.form.get('age', 0)),
            'Gender': int(request.form.get('gender', 0)),
            'University': int(request.form.get('university', 0)),
            'Department': int(request.form.get('department', 0)),
            'Academic_Year': int(request.form.get('academic_year', 0)),
            'Current_CGPA': int(request.form.get('cgpa', 0)),
            'waiver_or_scholarship': int(request.form.get('scholarship', 0)),
            **{f'PSS{i}': int(request.form.get(f'PSS{i}', 0)) for i in range(1, 11)},
            **{f'GAD{i}': int(request.form.get(f'GAD{i}', 0)) for i in range(1, 8)},
            **{f'PHQ{i}': int(request.form.get(f'PHQ{i}', 0)) for i in range(1, 10)},
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale input data
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)

        # Predict using the model
        predictions = model.predict(input_scaled)
        stress_value = predictions[0][0][0]
        anxiety_value = predictions[1][0][0]
        depression_value = predictions[2][0][0]

        # Categorize predictions
        results = {
            'Stress': {
                'Value': stress_value,
                'Category': categorize(stress_value, stress_thresholds, stress_labels)
            },
            'Anxiety': {
                'Value': anxiety_value,
                'Category': categorize(anxiety_value, anxiety_thresholds, anxiety_labels)
            },
            'Depression': {
                'Value': depression_value,
                'Category': categorize(depression_value, depression_thresholds, depression_labels)
            }
        }

        # Render results
        return render_template('results.html', results=results)

    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return "An error occurred. Please check the server logs for more details.", 500

if __name__ == '__main__':
    app.run(debug=True)
