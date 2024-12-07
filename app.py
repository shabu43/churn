from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model (assuming it's a joblib file)
model = joblib.load('model.joblib')

# Create Flask app
app = Flask(__name__)

# This is just a placeholder for all 45 features your model expects
# You need to know what those features are from your training dataset
expected_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 
                     'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
                     'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21',
                     'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28',
                     'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
                     'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42',
                     'feature43', 'feature44', 'feature45']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        # We expect 45 features as input, but we're just taking them from the form.
        features = []
        for feature in expected_features:
            feature_value = request.form.get(feature, type=float)
            features.append(feature_value)
        
        # If the user does not provide all the features, handle it (optional: you could set defaults)
        if len(features) != len(expected_features):
            return render_template('index.html', prediction="Error: Please provide all 45 features.")

        # Convert the features to a numpy array in the shape that the model expects
        features_array = np.array([features])
        
        # Predict using the loaded model
        prediction = model.predict(features_array)
        
        # Display prediction result
        prediction_text = 'Churn' if prediction[0] == 1 else 'No Churn'
        
        return render_template('index.html', prediction=prediction_text)
    
    except Exception as e:
        # Handle potential errors like missing form data, incorrect values, etc.
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
