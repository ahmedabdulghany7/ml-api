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

PREDICT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Titanic Survival Prediction</title>
  <style>
    :root{
      --bg1:#070b16;
      --bg2:#0b1220;
      --muted:#aab3c5;
      --text:#e8ecf6;
      --line:rgba(255,255,255,.10);
      --chip:rgba(255,255,255,.08);
      --ok:#22c55e;
      --bad:#ef4444;
      --warn:#f59e0b;
      --cardTop: rgba(255,255,255,.07);
      --cardBot: rgba(255,255,255,.03);
    }

    /* ✅ Important: full height + no weird scroll seams */
    html, body { height: 100%; }
    body{
      margin:0;
      min-height:100vh;
      font-family: system-ui, -apple-system, Segoe UI, Arial;
      color:var(--text);
      background: linear-gradient(180deg, var(--bg1), var(--bg2));
      overflow-x:hidden;
      position: relative;
    }

    /* ✅ Background layer 1 (fixed) */
    body::before{
      content:"";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events:none;
      background:
        radial-gradient(900px 600px at 12% 0%, rgba(56,189,248,.22), transparent 60%),
        radial-gradient(800px 520px at 92% 18%, rgba(34,197,94,.16), transparent 58%),
        radial-gradient(900px 650px at 50% 120%, rgba(168,85,247,.10), transparent 62%);
      background-repeat: no-repeat;
    }

    /* ✅ Background layer 2 overlay (fixed) */
    body::after{
      content:"";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events:none;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), transparent 30%, rgba(0,0,0,.30));
      background-repeat: no-repeat;
    }

    /* Content above background */
    .wrap{
      position: relative;
      z-index: 1;
      max-width:980px;
      margin:40px auto;
      padding:0 16px;
    }

    .header{
      display:flex;align-items:center;justify-content:space-between;gap:12px;
      margin-bottom:18px;
    }
    .title h1{margin:0;font-size:28px;letter-spacing:.2px}
    .title p{margin:6px 0 0;color:var(--muted)}
    .badge{
      padding:10px 12px;border:1px solid var(--line);border-radius:999px;
      background:rgba(255,255,255,.05);color:var(--muted);font-weight:700;
      white-space:nowrap;
    }

    .grid{display:grid;grid-template-columns:1.2fr .8fr;gap:14px}
    @media (max-width: 900px){ .grid{grid-template-columns:1fr;} }

    .card{
      background: linear-gradient(180deg, var(--cardTop), var(--cardBot));
      border:1px solid var(--line);
      border-radius:18px;
      padding:18px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
      /* backdrop-filter: blur(10px); */
    }

    .card h2{margin:0 0 10px;font-size:16px}
    .muted{color:var(--muted)}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    @media (max-width: 520px){ .row{grid-template-columns:1fr;} }
    label{display:block;margin:10px 0 6px;font-weight:800;color:#dbe3ff}

    input,select{
      width:100%;
      padding:12px 12px;
      border-radius:12px;
      border:1px solid var(--line);
      background: rgba(0,0,0,.26);
      color: var(--text);
      outline:none;
    }
    input::placeholder{color:rgba(232,236,246,.45)}
    input:focus, select:focus{
      border-color:rgba(56,189,248,.55);
      box-shadow:0 0 0 4px rgba(56,189,248,.12)
    }

    .actions{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
    button{
      padding:12px 16px;border:0;border-radius:14px;
      background: linear-gradient(90deg, rgba(56,189,248,.95), rgba(34,197,94,.90));
      color:#07101e;font-weight:900;cursor:pointer;
    }
    .ghost{
      background:transparent;border:1px solid var(--line);color:var(--text);
      padding:12px 16px;border-radius:14px;font-weight:800;cursor:pointer;
    }
    .hint{color:var(--muted);font-size:13px}

    .result{
      border-radius:16px;
      padding:14px;
      border:1px solid var(--line);
      background: rgba(255,255,255,.04);
    }
    .ok{border-color:rgba(34,197,94,.45); background: rgba(34,197,94,.10)}
    .bad{border-color:rgba(239,68,68,.45); background: rgba(239,68,68,.10)}
    .err{border-color:rgba(245,158,11,.55); background: rgba(245,158,11,.10)}
    .big{font-size:22px;font-weight:1000;margin:0}
    .small{margin:8px 0 0;color:var(--muted)}
    .kvs{margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .kv{background:var(--chip);border:1px solid var(--line);border-radius:14px;padding:10px}
    .kv b{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
    .kv span{font-weight:900}

    .footer{margin-top:14px;color:var(--muted);font-size:12px}
    a{color:#8be9ff;text-decoration:none}
    code{background:rgba(255,255,255,.06);padding:2px 6px;border-radius:8px;border:1px solid var(--line)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="title">
        <h1>🚢 Titanic Survival Prediction</h1>
        <p>Academic ML demo using your trained model (<code>titanic_model.pkl</code>).</p>
      </div>
      <div class="badge">Academic Project</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Passenger Information</h2>
        <p class="muted" style="margin-top:0">Please enter the passenger features below.</p>

        <form method="POST" action="/predict_page" novalidate>
          <div class="row">
            <div>
              <label for="Pclass">Passenger Class</label>
              <select id="Pclass" name="Pclass" required>
                <option value="1" {% if form.Pclass == 1 %}selected{% endif %}>1st Class</option>
                <option value="2" {% if form.Pclass == 2 %}selected{% endif %}>2nd Class</option>
                <option value="3" {% if form.Pclass == 3 %}selected{% endif %}>3rd Class</option>
              </select>
            </div>

            <div>
              <label for="Sex">Sex</label>
              <select id="Sex" name="Sex" required>
                <option value="0" {% if form.Sex == 0 %}selected{% endif %}>Female</option>
                <option value="1" {% if form.Sex == 1 %}selected{% endif %}>Male</option>
              </select>
            </div>
          </div>

          <div class="row">
            <div>
              <label for="Age">Age (years)</label>
              <input id="Age" type="number" step="0.01" min="0" name="Age"
                     value="{{ form.Age }}" placeholder="e.g., 22" required />
            </div>

            <div>
              <label for="Fare">Ticket Fare</label>
              <input id="Fare" type="number" step="0.01" min="0" name="Fare"
                     value="{{ form.Fare }}" placeholder="e.g., 7.25" required />
            </div>
          </div>

          <div class="row">
            <div>
              <label for="SibSp">Number of Siblings/Spouses Aboard</label>
              <input id="SibSp" type="number" step="1" min="0" name="SibSp"
                     value="{{ form.SibSp }}" placeholder="e.g., 1" required />
            </div>

            <div>
              <label for="Parch">Number of Parents/Children Aboard</label>
              <input id="Parch" type="number" step="1" min="0" name="Parch"
                     value="{{ form.Parch }}" placeholder="e.g., 0" required />
            </div>
          </div>

          <div class="actions">
            <button type="submit">Predict Survival</button>
            <button type="button" class="ghost" onclick="window.location.href='/predict_page'">Reset</button>
            <span class="hint">API: POST <code>/predict</code> · POST <code>/batch_predict</code></span>
          </div>
        </form>

        <div class="footer">
          Health check: <a href="/health">/health</a>
        </div>
      </div>

      <div class="card">
        <h2>Prediction Output</h2>

        {% if error %}
          <div class="result err">
            <p class="big">⚠️ Error</p>
            <p class="small">{{ error }}</p>
          </div>
        {% elif result %}
          <div class="result {{ 'ok' if result.prediction==1 else 'bad' }}">
            <p class="big">
              {{ "Survived ✅" if result.prediction==1 else "Not Survived ❌" }}
            </p>
            <p class="small">
              Confidence: <b>{{ (result.confidence * 100) | round(2) }}%</b><br/>
              Probability (Survived): <b>{{ (result.prob_survived * 100) | round(2) }}%</b>
            </p>

            <div class="kvs">
              <div class="kv"><b>Passenger Class</b><span>{{ form.Pclass }}</span></div>
              <div class="kv"><b>Sex</b><span>{{ "Female" if form.Sex==0 else "Male" }}</span></div>
              <div class="kv"><b>Age</b><span>{{ form.Age }}</span></div>
              <div class="kv"><b>Fare</b><span>{{ form.Fare }}</span></div>
              <div class="kv"><b>SibSp</b><span>{{ form.SibSp }}</span></div>
              <div class="kv"><b>Parch</b><span>{{ form.Parch }}</span></div>
            </div>
          </div>
        {% else %}
          <div class="result">
            <p class="big">—</p>
            <p class="small">No prediction yet. Fill the form and click <b>Predict Survival</b>.</p>
          </div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""

@app.route('/')
def home():
    return predict_page()

@app.route('/predict_page', methods=['GET', 'POST'])
def predict_page():
    form = {
        "Pclass": 3,
        "Sex": 1,
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25
    }

    result = None
    error = None

    if request.method == 'POST':
        try:
            form["Pclass"] = int(request.form.get("Pclass", form["Pclass"]))
            form["Sex"]   = int(request.form.get("Sex", form["Sex"]))
            form["Age"]   = float(request.form.get("Age", form["Age"]))
            form["SibSp"] = int(request.form.get("SibSp", form["SibSp"]))
            form["Parch"] = int(request.form.get("Parch", form["Parch"]))
            form["Fare"]  = float(request.form.get("Fare", form["Fare"]))

            if form["Pclass"] not in (1, 2, 3):
                raise ValueError("Passenger Class must be 1, 2, or 3.")
            if form["Sex"] not in (0, 1):
                raise ValueError("Sex must be Female or Male.")
            if form["Age"] < 0 or form["Fare"] < 0:
                raise ValueError("Age and Fare must be >= 0.")
            if form["SibSp"] < 0 or form["Parch"] < 0:
                raise ValueError("SibSp and Parch must be >= 0.")

            features = pd.DataFrame([{
                "Pclass": form["Pclass"],
                "Sex": form["Sex"],
                "Age": form["Age"],
                "SibSp": form["SibSp"],
                "Parch": form["Parch"],
                "Fare": form["Fare"]
            }])

            pred = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            conf = float(max(proba))

            result = {
                "prediction": pred,
                "prob_survived": float(proba[1]),
                "confidence": conf
            }

        except Exception as e:
            error = str(e)

    return render_template_string(PREDICT_HTML, result=result, error=error, form=form)

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
