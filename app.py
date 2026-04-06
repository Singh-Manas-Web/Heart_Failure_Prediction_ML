# app.py
# Flask backend for CHF Heart Failure Prediction
# ─────────────────────────────────────────────────
# HOW TO RUN:
#   1. Put app.py, best_model.pkl, scaler.pkl in same folder
#   2. pip install flask
#   3. python app.py
#   4. Open browser at http://localhost:5000
# ─────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model and scaler once when app starts ──
print("Loading model...")

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Model loaded successfully!")


# ── Read the HTML file ──
def get_html():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Route 1: Home page ──
@app.route("/")
def home():
    return get_html()


# ── Route 2: Prediction API ──
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Get data sent from the frontend form
        data = request.get_json()

        # Step 2: Extract all 12 features in correct order
        age                     = float(data["age"])
        anaemia                 = float(data["anaemia"])
        creatinine_phosphokinase = float(data["creatinine_phosphokinase"])
        diabetes                = float(data["diabetes"])
        ejection_fraction       = float(data["ejection_fraction"])
        high_blood_pressure     = float(data["high_blood_pressure"])
        platelets               = float(data["platelets"])
        serum_creatinine        = float(data["serum_creatinine"])
        serum_sodium            = float(data["serum_sodium"])
        sex                     = float(data["sex"])
        smoking                 = float(data["smoking"])
        time                    = float(data["time"])

        # Step 3: Put into array (same order as training)
        features = np.array([[
            age, anaemia, creatinine_phosphokinase, diabetes,
            ejection_fraction, high_blood_pressure, platelets,
            serum_creatinine, serum_sodium, sex, smoking, time
        ]])

        # Step 4: Scale the features (same scaler used in training)
        features_scaled = scaler.transform(features)

        # Step 5: Add engineered features (same as notebook Cell 9)
        low_ejection    = 1 if ejection_fraction < 40 else 0
        is_elderly      = 1 if age > 65 else 0
        high_creatinine = 1 if serum_creatinine > 1.5 else 0
        low_sodium      = 1 if serum_sodium < 135 else 0

        features_final = np.append(features_scaled, [
            [low_ejection, is_elderly, high_creatinine, low_sodium]
        ], axis=1)

        # Step 6: Make prediction
        prediction  = model.predict(features_final)[0]
        probability = model.predict_proba(features_final)[0][1]

        # Step 7: Send result back to frontend
        result = {
            "prediction": int(prediction),
            "label": "HIGH RISK " if prediction == 1 else "LOW RISK ",
            "probability": round(float(probability) * 100, 2),
            "status": "success"
        }
        return jsonify(result)

    except Exception as e:
        # If anything goes wrong, send error message
        return jsonify({"status": "error", "message": str(e)}), 400


# ── Route 3: Health check (optional) ──
@app.route("/health")
def health():
    return jsonify({"status": "running", "model": str(type(model).__name__)})


# ── Start the app ──
if __name__ == "__main__":
    print("Starting Flask server...")
    print("Open browser at: http://localhost:5000")
    app.run(debug=True, port=5000)