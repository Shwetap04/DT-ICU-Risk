# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_data

app = Flask(__name__)

# Load the CNN model
model = load_model("model/cnn_icu_model.h5")

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(message="API is running")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        
        if 'PATIENT_VISIT_IDENTIFIER' not in df.columns:
            return jsonify(error="Missing 'PATIENT_VISIT_IDENTIFIER' column"), 400
        
        # Preprocess
        X_input, patient_ids = preprocess_data(df)

        # Predict
        icu_risk = model.predict(X_input).ravel()
        response = []
        for pid, risk in zip(patient_ids, icu_risk):
            response.append({
                "patient_id": pid,
                "icu_risk": round(float(risk), 4)
            })
        return jsonify(predictions=response)

    except Exception as e:
        return jsonify(error=str(e)), 500
import requests
import pandas as pd

df = pd.read_excel("data/original_data.xlsx")  # or synthetic_data.xlsx
data_json = df.to_dict(orient="records")

response = requests.post("http://127.0.0.1:5000/predict", json=data_json)
print(response.json())
