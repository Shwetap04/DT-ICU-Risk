from flask import Flask, request, jsonify, make_response
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json
import logging

# Optional: Suppress Flask logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SCALER_PATH = r"C:/Users/Shweta/Documents/icu_dashboard_project/model/scaler.pkl"
MODEL_PATH  = r"C:/Users/Shweta/Documents/icu_dashboard_project/model/cnn_icu_model.h5"

scaler = joblib.load(SCALER_PATH)
model  = load_model(MODEL_PATH)
print("Model input shape:", model.input_shape)  # (None, 228, 1)

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "API is live"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "No input features provided"}), 400

    try:
        # Convert to DataFrame
        X = pd.DataFrame(features)
        print("Received input with shape:", X.shape)

        if X.shape[1] != scaler.mean_.shape[0]:
            return jsonify({"error": f"Expected 227 features, got {X.shape[1]}"}), 400

        # Scale features
        X_scaled = scaler.transform(X.iloc[:, :scaler.mean_.shape[0]])

        # Pad or truncate
        target_timesteps = model.input_shape[1]
        if X_scaled.shape[1] < target_timesteps:
            pad_width = target_timesteps - X_scaled.shape[1]
            X_padded = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
        else:
            X_padded = X_scaled[:, :target_timesteps]

        # Reshape
        X_input = X_padded.reshape((1, target_timesteps, 1))

        # Predict
        proba = model.predict(X_input).ravel()[0]

        # Fix potential NaN or inf
        safe_proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
        safe_proba = float(min(max(safe_proba, 0.0), 1.0))  # clip between 0-1

        pred = int(safe_proba >= 0.5)

        # Final JSON response using Flask’s jsonify (no manual json.dumps!)
        return jsonify({
            "predicted_class": pred,
            "probability": safe_proba
        }), 200

    except Exception as e:
        print("Exception during prediction:", e)
        return jsonify({"error": "Internal prediction error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
