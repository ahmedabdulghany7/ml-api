"""
MNIST Digit Classification API
Flask application with SMART MODEL LOADING
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os
import base64
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = "mnist_cnn_model.keras"
model = None

def train_model():
    """Train MNIST CNN model from scratch"""
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
    from keras.datasets import mnist
    from keras.utils import to_categorical
    
    print("🚀 Training new model...")
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("🏋️ Training (3 epochs)...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=3, batch_size=128, verbose=1)
    
    try:
        model.save(MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    return model

def load_or_train_model():
    """Smart model loading with fallback"""
    global model
    
    # Try loading existing model
    if os.path.exists(MODEL_PATH):
        print(f"📂 Found model file: {MODEL_PATH}")
        try:
            # Try modern Keras 3.x loading
            import keras
            model = keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded with Keras {keras.__version__}")
            return model
        except Exception as e1:
            print(f"⚠️ Keras 3.x loading failed: {e1}")
            
            try:
                # Try legacy TensorFlow/Keras 2.x loading
                from tensorflow import keras as tf_keras
                model = tf_keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded with TensorFlow Keras")
                return model
            except Exception as e2:
                print(f"⚠️ TensorFlow Keras loading failed: {e2}")
                print("🔄 Will train new model...")
    
    # Train new model if loading failed or file doesn't exist
    print("🆕 No compatible model found, training new one...")
    model = train_model()
    return model

# Load model on startup
try:
    model = load_or_train_model()
    if model is None:
        raise Exception("Model loading/training failed")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    model = None


HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Classifier</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family:'Segoe UI', Tahoma, sans-serif;
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
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button.secondary { background: rgba(255,255,255,0.2); }
        .result { margin-top:20px; text-align:center; display:none; }
        .digit-result {
            font-size:5em; font-weight:bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
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

        canvas.addEventListener('mousedown', (e) => { isDrawing = true; draw(e); });
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
        canvas.addEventListener('mouseout', () => { isDrawing = false; ctx.beginPath(); });

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
                const gray = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
                pixels.push((255 - gray) / 255);
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pixels: pixels })
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.message || "Failed");

                document.getElementById('result').style.display = 'block';
                document.getElementById('digit').textContent = data.digit;
                document.getElementById('confidence').textContent =
                    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

                let probHtml = '';
                for (let i = 0; i < 10; i++) {
                    const prob = data.probabilities[String(i)] || 0;
                    probHtml += `
                        <div class="prob-bar">
                            <span class="prob-label">${i}</span>
                            <div class="prob-fill" style="width:${prob * 100}%"></div>
                            <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `;
                }
                document.getElementById('probabilities').innerHTML = probHtml;

            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "pixels" not in data:
        return jsonify({"error": "Invalid request"}), 400

    pixels = data["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 784:
        return jsonify({"error": f"Expected 784 pixels, got {len(pixels)}"}), 400

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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
