import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

SCALER_PATH = r"C:/Users/Shweta/Documents/icu_dashboard_project/model/scaler.pkl"
MODEL_PATH  = r"C:/Users/Shweta/Documents/icu_dashboard_project/model/cnn_icu_model.h5"

scaler = joblib.load(SCALER_PATH)
model  = load_model(MODEL_PATH)

def predict_icu_risk(df):
    results = []

    for patient_id in df["PATIENT_VISIT_IDENTIFIER"].unique():
        patient_df = df[df["PATIENT_VISIT_IDENTIFIER"] == patient_id]

        if len(patient_df) < 5:
            continue  # Skip if we don’t have a full sequence

        # Sort and trim to 5 windows
        patient_df = patient_df.sort_values("WINDOW").iloc[:5]

        # ✅ Ensure consistent features
        required_features = scaler.feature_names_in_
        features = patient_df[required_features]

        # ✅ Scale
        features_scaled = scaler.transform(features)

        # ✅ Reshape for model: (1, 5 * n_features)
        input_vector = features_scaled.flatten().reshape(1, -1)

        # ✅ Predict
        prediction = model.predict(input_vector)[0][0]

        # ✅ Store results
        results.append({
            "patient_id": patient_id,
            "icu_risk": prediction,
            "window_axis": patient_df["WINDOW"].tolist(),
            "vitals": features_scaled.tolist()
        })

    return results
