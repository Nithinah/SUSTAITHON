# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# Load dataset
file_path = 'E:\\VIT_Hackathon\\Processed.csv'  # Update this path if needed
data = pd.read_csv(file_path)

# Data preprocessing
# Encode categorical features
categorical_columns = ['Age', 'Gender', 'University', 'Department', 'Academic_Year', 'Current_CGPA', 'waiver_or_scholarship']
encoded_data = data.copy()

for col in categorical_columns:
    encoder = LabelEncoder()
    encoded_data[col] = encoder.fit_transform(data[col])

# Separate features and targets
X = encoded_data.drop(['Stress Value', 'Stress Label', 'Anxiety Value', 'Anxiety Label', 'Depression Value', 'Depression Label'], axis=1)
y_stress = encoded_data['Stress Value']
y_anxiety = encoded_data['Anxiety Value']
y_depression = encoded_data['Depression Value']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train_stress, y_test_stress, y_train_anxiety, y_test_anxiety, y_train_depression, y_test_depression = train_test_split(
    X_scaled, y_stress, y_anxiety, y_depression, test_size=0.2, random_state=42
)

# Model architecture
input_layer = Input(shape=(X_train.shape[1],))

# 10 Hidden layers
x = Dense(256, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.3)(x)

# Output layers
stress_output = Dense(1, activation='linear', name='stress_output')(x)
anxiety_output = Dense(1, activation='linear', name='anxiety_output')(x)
depression_output = Dense(1, activation='linear', name='depression_output')(x)

# Combine into a model
model = Model(inputs=input_layer, outputs=[stress_output, anxiety_output, depression_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'stress_output': 'mse', 'anxiety_output': 'mse', 'depression_output': 'mse'},
              metrics={'stress_output': 'mae', 'anxiety_output': 'mae', 'depression_output': 'mae'})

# Train the model
history = model.fit(
    X_train,
    {'stress_output': y_train_stress, 'anxiety_output': y_train_anxiety, 'depression_output': y_train_depression},
    validation_data=(X_test, {'stress_output': y_test_stress, 'anxiety_output': y_test_anxiety, 'depression_output': y_test_depression}),
    epochs=50, batch_size=32
)

# Save the model
model.save('stress_anxiety_depression_model_10_layers.h5')

# Define thresholds and labels
stress_thresholds = [13, 26]  # Low, Moderate, High
stress_labels = ['Low Stress', 'Moderate Stress', 'High Stress']

anxiety_thresholds = [4, 9, 14]
anxiety_labels = ['Minimal Anxiety', 'Mild Anxiety', 'Moderate Anxiety', 'Severe Anxiety']

depression_thresholds = [0, 4, 9, 14, 19]
depression_labels = ['No Depression', 'Minimal Depression', 'Mild Depression', 'Moderate Depression', 'Moderately Severe Depression', 'Severe Depression']

# Helper function to categorize continuous values into labels
def categorize(value, thresholds, labels):
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return labels[i]
    return labels[-1]

# Predict on test data
predictions = model.predict(X_test)
y_pred_stress = predictions[0].flatten()
y_pred_anxiety = predictions[1].flatten()
y_pred_depression = predictions[2].flatten()

# Convert continuous predictions to categorical labels
y_test_stress_labels = [categorize(val, stress_thresholds, stress_labels) for val in y_test_stress]
y_test_anxiety_labels = [categorize(val, anxiety_thresholds, anxiety_labels) for val in y_test_anxiety]
y_test_depression_labels = [categorize(val, depression_thresholds, depression_labels) for val in y_test_depression]

y_pred_stress_labels = [categorize(val, stress_thresholds, stress_labels) for val in y_pred_stress]
y_pred_anxiety_labels = [categorize(val, anxiety_thresholds, anxiety_labels) for val in y_pred_anxiety]
y_pred_depression_labels = [categorize(val, depression_thresholds, depression_labels) for val in y_pred_depression]

# Compute accuracy
stress_accuracy = accuracy_score(y_test_stress_labels, y_pred_stress_labels)
anxiety_accuracy = accuracy_score(y_test_anxiety_labels, y_pred_anxiety_labels)
depression_accuracy = accuracy_score(y_test_depression_labels, y_pred_depression_labels)

print(f"Stress Prediction Accuracy: {stress_accuracy:.2f}")
print(f"Anxiety Prediction Accuracy: {anxiety_accuracy:.2f}")
print(f"Depression Prediction Accuracy: {depression_accuracy:.2f}")

# Confusion matrix and classification report for each task
print("\nStress Classification Report:")
print(classification_report(y_test_stress_labels, y_pred_stress_labels))
print("Stress Confusion Matrix:")
print(confusion_matrix(y_test_stress_labels, y_pred_stress_labels))

print("\nAnxiety Classification Report:")
print(classification_report(y_test_anxiety_labels, y_pred_anxiety_labels))
print("Anxiety Confusion Matrix:")
print(confusion_matrix(y_test_anxiety_labels, y_pred_anxiety_labels))

print("\nDepression Classification Report:")
print(classification_report(y_test_depression_labels, y_pred_depression_labels))
print("Depression Confusion Matrix:")
print(confusion_matrix(y_test_depression_labels, y_pred_depression_labels))

# Example usage with new input
new_data = pd.DataFrame({
    'Age': [1],  # Encoded Age
    'Gender': [0],  # Encoded Gender
    'University': [2],  # Encoded University
    'Department': [3],  # Encoded Department
    'Academic_Year': [1],  # Encoded Academic Year
    'Current_CGPA': [4],  # Encoded CGPA
    'waiver_or_scholarship': [0],  # Encoded waiver_or_scholarship
    'PSS1': [3], 'PSS2': [4], 'PSS3': [2], 'PSS4': [3], 'PSS5': [4],
    'PSS6': [2], 'PSS7': [3], 'PSS8': [4], 'PSS9': [2], 'PSS10': [3],
    'GAD1': [2], 'GAD2': [3], 'GAD3': [4], 'GAD4': [2], 'GAD5': [3],
    'GAD6': [4], 'GAD7': [3],
    'PHQ1': [3], 'PHQ2': [4], 'PHQ3': [2], 'PHQ4': [3], 'PHQ5': [4],
    'PHQ6': [2], 'PHQ7': [3], 'PHQ8': [4], 'PHQ9': [3]
})

# Scale the new input
new_data_scaled = scaler.transform(new_data)
new_predictions = model.predict(new_data_scaled)

# Extract numerical predictions and categorize
results = {
    'Stress': {
        'Value': new_predictions[0][0][0],
        'Category': categorize(new_predictions[0][0][0], stress_thresholds, stress_labels)
    },
    'Anxiety': {
        'Value': new_predictions[1][0][0],
        'Category': categorize(new_predictions[1][0][0], anxiety_thresholds, anxiety_labels)
    },
    'Depression': {
        'Value': new_predictions[2][0][0],
        'Category': categorize(new_predictions[2][0][0], depression_thresholds, depression_labels)
    }
}

print("\nPredicted Outputs for New Data:")
print(f"Stress: Value = {results['Stress']['Value']:.2f}, Category = {results['Stress']['Category']}")
print(f"Anxiety: Value = {results['Anxiety']['Value']:.2f}, Category = {results['Anxiety']['Category']}")
print(f"Depression: Value = {results['Depression']['Value']:.2f}, Category = {results['Depression']['Category']}")
