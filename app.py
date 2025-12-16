"""
MNIST Digit Classification API
Flask application for serving digit recognition predictions (Keras CNN)
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os
import base64
from io import BytesIO

# Keras / TensorFlow
from keras.models import load_model

app = Flask(__name__)

# =========================
# Load Keras CNN model
# =========================
MODEL_PATH = "mnist_cnn_model.keras"

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Keras CNN model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_PATH}: {e}")
    model = None


# =========================
# HTML template (same UI)
# =========================
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Classifier API</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height:100vh; color:white; padding:20px;
        }
        .container { max-width:900px; margin:0 auto; }
        h1 { text-align:center; margin-bottom:10px; font-size:2.5em; }
        .subtitle { text-align:center; color:#888; margin-bottom:30px; }
        .main-content { display:grid; grid-template-columns:1fr 1fr; gap:30px; }
        @media (max-width:768px) { .main-content { grid-template-columns:1fr; } }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius:20px; padding:25px; backdrop-filter: blur(10px);
        }
        .card h2 { margin-bottom:20px; font-size:1.3em; }
        canvas {
            background:white; border-radius:10px; cursor:crosshair;
            display:block; margin:0 auto 20px;
        }
        .btn-group { display:flex; gap:10px; justify-content:center; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color:white; border:none; padding:12px 25px; font-size:14px;
            border-radius:8px; cursor:pointer; font-weight:600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102,126,234,0.4); }
        button.secondary { background: rgba(255,255,255,0.2); }
        .result { margin-top:20px; text-align:center; display:none; }
        .digit-result {
            font-size:5em; font-weight:bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        }
        .confidence { color:#888; margin-top:10px; }
        .probabilities { margin-top:20px; }
        .prob-bar { display:flex; align-items:center; margin:5px 0; }
        .prob-label { width:30px; font-weight:bold; }
        .prob-fill {
            height:20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius:4px; transition: width 0.3s;
        }
        .prob-value { margin-left:10px; color:#888; font-size:0.9em; }
        .api-docs { margin-top:30px; }
        .api-docs h3 { margin-bottom:15px; }
        code {
            background: rgba(0,0,0,0.3); padding:3px 8px; border-radius:5px;
            font-family:'Courier New', monospace;
        }
        pre {
            background: rgba(0,0,0,0.3); padding:15px; border-radius:10px;
            overflow-x:auto; margin:10px 0; font-size:0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 MNIST Digit Classifier</h1>
        <p class="subtitle">Draw a digit (0-9) and let AI predict it!</p>

        <div class="main-content">
            <div class="card">
                <h2>✏️ Draw a Digit</h2>
                <canvas id="canvas" width="280" height="280"></canvas>
                <div class="btn-group">
                    <button onclick="predict()">🔮 Predict</button>
                    <button class="secondary" onclick="clearCanvas()">🗑️ Clear</button>
                </div>

                <div id="result" class="result">
                    <div class="digit-result" id="digit"></div>
                    <div class="confidence" id="confidence"></div>
                </div>
            </div>

            <div class="card">
                <h2>📊 Prediction Probabilities</h2>
                <div id="probabilities" class="probabilities">
                    <p style="color:#888; text-align:center;">Draw a digit to see probabilities</p>
                </div>
            </div>
        </div>

        <div class="card api-docs">
            <h3>📡 API Documentation</h3>
            <p><strong>Endpoint:</strong> <code>POST /predict</code></p>
            <p><strong>Request Body:</strong></p>
            <pre>{
  "pixels": [0.0, 0.0, ..., 0.5, 0.8, ...]  // 784 values (0-1 normalized)
}</pre>
            <p><strong>Response:</strong></p>
            <pre>{
  "digit": 5,
  "confidence": 0.95,
  "probabilities": {"0": 0.01, "1": 0.02, ..., "9": 0.01}
}</pre>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        function startDrawing(e) { isDrawing = true; draw(e); }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        function stopDrawing() { isDrawing = false; ctx.beginPath(); }

        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').style.display = 'none';
            document.getElementById('probabilities').innerHTML =
                '<p style="color:#888; text-align:center;">Draw a digit to see probabilities</p>';
        }

        async function predict() {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, 28, 28);

            const imageData = tempCtx.getImageData(0, 0, 28, 28);
            const pixels = [];

            for (let i = 0; i < imageData.data.length; i += 4) {
                const r = imageData.data[i];
                const g = imageData.data[i + 1];
                const b = imageData.data[i + 2];
                const gray = (r + g + b) / 3;
                pixels.push((255 - gray) / 255); // invert (white->0, black->1)
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pixels: pixels })
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.message || data.error || "Request failed");

                document.getElementById('result').style.display = 'block';
                document.getElementById('digit').textContent = data.digit;
                document.getElementById('confidence').textContent =
                    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

                let probHtml = '';
                for (let i = 0; i < 10; i++) {
                    const prob = data.probabilities[String(i)] || 0;
                    const width = prob * 100;
                    probHtml += `
                        <div class="prob-bar">
                            <span class="prob-label">${i}</span>
                            <div class="prob-fill" style="width:${width}%"></div>
                            <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `;
                }
                document.getElementById('probabilities').innerHTML = probHtml;

            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        window.predict = predict;
        window.clearCanvas = clearCanvas;
    </script>
</body>
</html>
"""


# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_type": "Keras CNN"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "message": f"Please ensure {MODEL_PATH} exists"
        }), 500

    data = request.get_json(silent=True)
    if not data or "pixels" not in data:
        return jsonify({
            "error": "Invalid request",
            "message": 'Please provide JSON with "pixels" array (784 values)'
        }), 400

    pixels = data["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 784:
        return jsonify({
            "error": "Invalid pixels",
            "message": f"Expected 784 pixel values, got {len(pixels) if isinstance(pixels, list) else 'invalid type'}"
        }), 400

    try:
        x = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1)

        probs = model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return jsonify({
            "digit": digit,
            "confidence": confidence,
            "probabilities": {str(i): float(probs[i]) for i in range(10)}
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "error": "Invalid request",
            "message": 'Please provide JSON with "image" field (base64 encoded)'
        }), 400

    try:
        from PIL import Image

        image_data = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_data)).convert("L").resize((28, 28))

        arr = np.array(image, dtype=np.float32) / 255.0

        # invert if mostly white background
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        x = arr.reshape(1, 28, 28, 1)

        probs = model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))

        return jsonify({
            "digit": digit,
            "confidence": float(np.max(probs)),
            "probabilities": {str(i): float(probs[i]) for i in range(10)}
        })

    except ImportError:
        return jsonify({
            "error": "PIL not installed",
            "message": "pip install pillow (or use /predict with pixels)"
        }), 500
    except Exception as e:
        return jsonify({"error": "Image prediction failed", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
