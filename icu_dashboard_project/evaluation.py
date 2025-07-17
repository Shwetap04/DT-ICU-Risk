# evaluation.py

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load model and scaler ===
model = load_model("model/cnn_icu_model.h5")
scaler = joblib.load("model/scaler.pkl")

# === Load and preprocess data ===
df = pd.read_excel("data/original_data.xlsx")
X = df.drop(columns=["ICU", "PATIENT_VISIT_IDENTIFIER", "WINDOW"], errors='ignore')
X = X.select_dtypes(include='number').fillna(0)

y = df["ICU"].values

# === Pad features if needed ===
X_array = X.values
expected_features = model.input_shape[1]
current_features = X_array.shape[1]
padding_required = expected_features - current_features

if padding_required > 0:
    print(f"🔧 Padding input with {padding_required} zero columns...")
    X_array = np.pad(X_array, ((0, 0), (0, padding_required)), mode='constant')
elif padding_required < 0:
    raise ValueError(f"❌ Too many features. Reduce from {current_features} to {expected_features}")

# === Scale using fitted scaler ===
X_scaled = scaler.transform(X_array[:, :scaler.mean_.shape[0]])

# === Re-pad after scaling if needed ===
if padding_required > 0:
    X_scaled = np.pad(X_scaled, ((0, 0), (0, padding_required)), mode='constant')

# === Reshape for CNN ===
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# === Predict ===
predicted_probs = model.predict(X_reshaped).flatten()
predicted_classes = (predicted_probs > 0.5).astype(int)

# === Classification report ===
print("\n--- Classification Report ---")
print(classification_report(y, predicted_classes))

# === Confusion Matrix ===
cm = confusion_matrix(y, predicted_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()  # 👈 Popup plot

# === ROC-AUC ===
roc_auc = roc_auc_score(y, predicted_probs)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y, predicted_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()  # 👈 Popup plot

# === Add predictions to DataFrame ===
df["Predicted_Prob"] = predicted_probs
df["Predicted_Class"] = predicted_classes

# === Show misclassified ICU=1 patients ===
print("\n--- Misclassified High-Risk Patients (Actual=1, Predicted=0) ---")
misclassified = df[(df["ICU"] == 1) & (df["Predicted_Class"] == 0)]
if misclassified.empty:
    print("✅ No misclassifications.")
else:
    print(misclassified[["PATIENT_VISIT_IDENTIFIER", "WINDOW", "Predicted_Prob"]])
