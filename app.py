from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

try:
    app.json.ensure_ascii = False  # Flask 2.2+
except Exception:
    app.config["JSON_AS_ASCII"] = False

with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

HTML_HOME = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Titanic Survival Prediction API</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial;max-width:900px;margin:40px auto;padding:0 16px;line-height:1.5}
    code,pre{background:#f6f8fa;padding:10px;border-radius:10px;display:block;overflow:auto}
    .card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:14px 0}
    .ok{color:#16a34a;font-weight:700}
    .muted{color:#6b7280}
  </style>
</head>
<body>
  <h1>🚢 Titanic Survival Prediction API</h1>
  <p class="ok">Status: Active</p>

  <div class="card">
    <h2>Endpoints</h2>
    <ul>
      <li><b>GET</b> / (this page)</li>
      <li><b>GET</b> /api (API info as JSON)</li>
      <li><b>GET</b> /health</li>
      <li><b>POST</b> /predict</li>
      <li><b>POST</b> /batch_predict</li>
    </ul>
  </div>

  <div class="card">
    <h2>Required Features</h2>
    <ul>
      <li>Pclass: int (1,2,3)</li>
      <li>Sex: int (0=female, 1=male)</li>
      <li>Age: float</li>
      <li>SibSp: int</li>
      <li>Parch: int</li>
      <li>Fare: float</li>
    </ul>
  </div>

  <div class="card">
    <h2>Example Request (predict)</h2>
    <pre>{
  "Pclass": 3,
  "Sex": 1,
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25
}</pre>

    <p class="muted">Tip: Use Postman / curl to send POST JSON.</p>
    <code>curl -X POST {{base}}/predict -H "Content-Type: application/json" -d '{"Pclass":3,"Sex":1,"Age":22,"SibSp":1,"Parch":0,"Fare":7.25}'</code>
  </div>
</body>
</html>
"""

@app.route('/')
def home():
    base = request.host_url.rstrip("/")
    return render_template_string(HTML_HOME, base=base)

@app.route('/api')
def api_info():
    return jsonify({
        'message': '🚢 Titanic Survival Prediction API',
        'status': 'active',
        'endpoints': {
            '/': 'HTML documentation page',
            '/api': 'API information (JSON)',
            '/predict': 'POST - Make predictions',
            '/batch_predict': 'POST - Batch predictions',
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
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields,
                'required': required_fields
            }), 400

        features = pd.DataFrame([{
            'Pclass': data['Pclass'],
            'Sex': data['Sex'],
            'Age': data['Age'],
            'SibSp': data['SibSp'],
            'Parch': data['Parch'],
            'Fare': data['Fare']
        }])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'survival_status': 'Survived' if prediction == 1 else 'Not Survived',
            'probabilities': {
                'not_survived': float(probability[0]),
                'survived': float(probability[1])
            },
            'confidence': float(max(probability)),
            'input_data': data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(force=True)

        if not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of passengers'}), 400

        results = []
        for idx, passenger in enumerate(data, start=1):
            required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
            missing = [f for f in required_fields if f not in passenger]
            if missing:
                return jsonify({
                    'error': f'Missing fields for passenger #{idx}',
                    'missing': missing
                }), 400

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
                'passenger_id': idx,
                'prediction': int(prediction),
                'survival_status': 'Survived' if prediction == 1 else 'Not Survived',
                'probability_survived': float(probability[1]),
                'input_data': passenger
            })

        return jsonify({
            'success': True,
            'total_passengers': len(results),
            'predictions': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
