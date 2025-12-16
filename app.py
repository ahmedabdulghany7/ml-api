"""
Sentiment Analysis API - Ready for Render Deployment
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import os

app = Flask(__name__)

# Load model
try:
    model = joblib.load('sentiment_model.pkl')
    print("✅ Model loaded!")
except:
    model = None
    print("⚠️ Model not found!")

# Home page HTML
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis API</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
        }
        h1 { color: #333; margin-bottom: 20px; text-align: center; }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            margin-bottom: 15px;
            min-height: 100px;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            display: none;
        }
        .positive { background: #d4edda; color: #155724; }
        .negative { background: #f8d7da; color: #721c24; }
        .result-text { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Sentiment Analysis</h1>
        <textarea id="text" placeholder="Enter text to analyze..."></textarea>
        <button onclick="analyze()">Analyze Sentiment</button>
        <div id="result" class="result">
            <div class="result-text" id="sentiment"></div>
            <div id="confidence"></div>
        </div>
    </div>
    <script>
        async function analyze() {
            const text = document.getElementById('text').value;
            if (!text) { alert('Please enter text'); return; }
            
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result ' + data.sentiment;
            
            const emoji = data.sentiment === 'positive' ? '😊' : '😞';
            document.getElementById('sentiment').textContent = emoji + ' ' + data.sentiment.toUpperCase();
            document.getElementById('confidence').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text'}), 400
    
    text = data['text']
    prediction = model.predict([text])[0]
    confidence = float(max(model.predict_proba([text])[0]))
    
    return jsonify({
        'sentiment': prediction,
        'confidence': confidence,
        'text': text
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
