from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': '🚢 Titanic Survival Prediction API',
        'status': 'active',
        'endpoints': {
            '/': 'API information',
            '/predict': 'POST - Make predictions',
            '/health': 'Health check'
        },
        'required_features': {
            'Pclass': 'int (1, 2, or 3) - Passenger class',
            'Sex': 'int (0=female, 1=male)',
            'Age': 'float - Age in years',
            'SibSp': 'int - Number of siblings/spouses',
            'Parch': 'int - Number of parents/children',
            'Fare': 'float - Ticket fare'
        },
        'example_request': {
            'Pclass': 3,
            'Sex': 1,
            'Age': 22,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 7.25
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts JSON with passenger features
    Returns survival prediction and probability
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields,
                'required': required_fields
            }), 400
        
        # Create DataFrame with features in correct order
        features = pd.DataFrame([{
            'Pclass': data['Pclass'],
            'Sex': data['Sex'],
            'Age': data['Age'],
            'SibSp': data['SibSp'],
            'Parch': data['Parch'],
            'Fare': data['Fare']
        }])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'survival_status': 'Survived' if prediction == 1 else 'Not Survived',
            'probabilities': {
                'not_survived': float(probability[0]),
                'survived': float(probability[1])
            },
            'confidence': float(max(probability)),
            'input_data': data
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Accepts JSON array of passengers
    Returns predictions for all passengers
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                'error': 'Data must be a list of passengers'
            }), 400
        
        # Process each passenger
        results = []
        for idx, passenger in enumerate(data):
            features = pd.DataFrame([{
                'Pclass': passenger['Pclass'],
                'Sex': passenger['Sex'],
                'Age': passenger['Age'],
                'SibSp': passenger['SibSp'],
                'Parch': passenger['Parch'],
                'Fare': passenger['Fare']
            }])
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            results.append({
                'passenger_id': idx + 1,
                'prediction': int(prediction),
                'survival_status': 'Survived' if prediction == 1 else 'Not Survived',
                'probability': float(probability[1]),
                'input_data': passenger
            })
        
        return jsonify({
            'success': True,
            'total_passengers': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
