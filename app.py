from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model (replace 'model.joblib' with your model file)
model = joblib.load('model.joblib')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data for all 45 features
    features = []
    for i in range(1, 46):  # Adjusted to collect 45 features
        feature_value = float(request.form.get(f'feature{i}', 0))  # Default to 0 if missing
        features.append(feature_value)

    # Convert features list to a numpy array
    features = np.array(features).reshape(1, -1)  # Model expects 2D array (1 sample, 45 features)

    # Predict
    prediction = model.predict(features)
    prediction_text = 'Churn' if prediction[0] == 1 else 'No Churn'

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
