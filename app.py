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
    gender = request.form['gender']
    senior_citizen = int(request.form['senior_citizen'])
    partner = request.form['partner']
    dependents = request.form['dependents']
    phone_service = request.form['phone_service']
    multiple_lines = request.form['multiple_lines']
    internet_service = request.form['internet_service']
    online_security = request.form['online_security']
    online_backup = request.form['online_backup']
    device_protection = request.form['device_protection']
    tech_support = request.form['tech_support']
    streaming_tv = request.form['streaming_tv']
    streaming_movies = request.form['streaming_movies']
    contract = request.form['contract']
    paperless_billing = request.form['paperless_billing']
    payment_method = request.form['payment_method']
    monthly_charges = float(request.form['monthly_charges'])
    total_charges = float(request.form['total_charges'])
    
    # Model expects a 2D array, format the input
    features = np.array([[tenure, monthly_charges, total_charges, gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
                          internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                          contract, paperless_billing, payment_method]])
    
    # Predict
    prediction = model.predict(features)
    prediction_text = 'Churn' if prediction[0] == 1 else 'No Churn'
    
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
