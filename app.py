from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

try:
    app.json.ensure_ascii = False
except Exception:
    app.config["JSON_AS_ASCII"] = False

with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)


PREDICT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Titanic Prediction</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial;max-width:900px;margin:40px auto;padding:0 16px;line-height:1.5}
    .card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:14px 0}
    label{display:block;font-weight:700;margin:10px 0 6px}
    input,select{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:10px}
    button{padding:12px 16px;border:0;border-radius:12px;cursor:pointer;font-weight:800}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .result{padding:12px;border-radius:12px;margin-top:14px}
    .ok{background:#dcfce7;border:1px solid #16a34a}
    .bad{background:#fee2e2;border:1px solid #dc2626}
    .muted{color:#6b7280}
  </style>
</head>
<body>
  <h1>🚢 Titanic Survival Prediction</h1>
  <p class="muted">Fill the passenger data and click Predict.</p>

  <div class="card">
    <form method="POST" action="/predict_page">
      <div class="row">
        <div>
          <label>Pclass</label>
          <select name="Pclass" required>
            <option value="1">1 (First)</option>
            <option value="2">2 (Second)</option>
            <option value="3" selected>3 (Third)</option>
          </select>
        </div>

        <div>
          <label>Sex</label>
          <select name="Sex" required>
            <option value="0">Female (0)</option>
            <option value="1" selected>Male (1)</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Age</label>
          <input type="number" step="0.01" name="Age" value="22" required />
        </div>
        <div>
          <label>Fare</label>
          <input type="number" step="0.01" name="Fare" value="7.25" required />
        </div>
      </div>

      <div class="row">
        <div>
          <label>SibSp</label>
          <input type="number" step="1" name="SibSp" value="1" required />
        </div>
        <div>
          <label>Parch</label>
          <input type="number" step="1" name="Parch" value="0" required />
        </div>
      </div>

      <div style="margin-top:14px;">
        <button type="submit">Predict</button>
      </div>
    </form>

    {% if result %}
      <div class="result {{ 'ok' if result.prediction==1 else 'bad' }}">
        <h2 style="margin:0;">
          Result: {{ "Survived ✅" if result.prediction==1 else "Not Survived ❌" }}
        </h2>
        <p style="margin:8px 0 0;">
          Confidence: {{ (result.confidence * 100) | round(2) }}%
          <br/>
          Prob Survived: {{ (result.prob_survived * 100) | round(2) }}%
        </p>
      </div>
    {% endif %}
  </div>

  <div class="card">
    <b>API still available:</b>
    <ul>
      <li>POST /predict (JSON)</li>
      <li>POST /batch_predict (JSON)</li>
      <li>GET /health</li>
    </ul>
  </div>
</body>
</html>
"""

@app.route('/')
def home():
    return predict_page()

@app.route('/predict_page', methods=['GET', 'POST'])
def predict_page():
    result = None

    if request.method == 'POST':
        try:
            data = {
                'Pclass': int(request.form['Pclass']),
                'Sex': int(request.form['Sex']),
                'Age': float(request.form['Age']),
                'SibSp': int(request.form['SibSp']),
                'Parch': int(request.form['Parch']),
                'Fare': float(request.form['Fare'])
            }

            features = pd.DataFrame([data])
            pred = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            conf = float(max(proba))

            result = {
                "prediction": pred,
                "prob_survived": float(proba[1]),
                "confidence": conf
            }
        except Exception as e:
            result = {"prediction": 0, "prob_survived": 0.0, "confidence": 0.0}

    return render_template_string(PREDICT_HTML, result=result)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': 'Missing required fields', 'missing': missing}), 400

        features = pd.DataFrame([{
            'Pclass': data['Pclass'],
            'Sex': data['Sex'],
            'Age': data['Age'],
            'SibSp': data['SibSp'],
            'Parch': data['Parch'],
            'Fare': data['Fare']
        }])

        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]

        return jsonify({
            'success': True,
            'prediction': pred,
            'survival_status': 'Survived' if pred == 1 else 'Not Survived',
            'probabilities': {'not_survived': float(proba[0]), 'survived': float(proba[1])},
            'confidence': float(max(proba)),
            'input_data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, list):
            return jsonify({'error': 'Data must be a list'}), 400

        results = []
        for i, passenger in enumerate(data, start=1):
            features = pd.DataFrame([passenger])
            pred = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            results.append({
                'passenger_id': i,
                'prediction': pred,
                'survival_status': 'Survived' if pred == 1 else 'Not Survived',
                'probability_survived': float(proba[1]),
            })

        return jsonify({'success': True, 'total': len(results), 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
