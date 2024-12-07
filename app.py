from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model (replace 'model.pkl' with your model file)
model = pickle.load(open('model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthly_charges'])
    total_charges = float(request.form['total_charges'])
    
    # Model expects a 2D array, format the input
    features = np.array([[tenure, monthly_charges, total_charges]])
    
    # Predict
    prediction = model.predict(features)
    prediction_text = 'Churn' if prediction[0] == 1 else 'No Churn'
    
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
