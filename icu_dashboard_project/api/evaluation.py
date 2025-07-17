# evaluation.py

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Create output directory ===
os.makedirs("evaluation", exist_ok=True)

# === Load model and scaler ===
model = load_model("model/cnn_icu_model.h5")
scaler = joblib.load("model/scaler.pkl")

# === Load and preprocess data ===
df = pd.read_excel("data/original_data.xlsx")
X = df.drop(columns=["ICU", "PATIENT_VISIT_IDENTIFIER", "WINDOW"], errors="ignore")
X = X.select_dtypes(include="number").fillna(0)
y = df["ICU"].values

# === Scale and reshape ===
X_scaled = scaler.transform(X)
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
plt.savefig("evaluation/confusion_matrix.png")
plt.show()

# === ROC-AUC Score ===
roc_auc = roc_auc_score(y, predicted_probs)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y, predicted_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation/roc_curve.png")
plt.show()

# === Save predictions with patient IDs and time windows ===
df["Predicted_Prob"] = predicted_probs
df["Predicted_Class"] = predicted_classes
df.to_excel("evaluation/predictions_with_risk.xlsx", index=False)
print("\n✅ Predictions saved to 'evaluation/predictions_with_risk.xlsx'")
print("✅ Confusion matrix saved to 'evaluation/confusion_matrix.png'")
print("✅ ROC curve saved to 'evaluation/roc_curve.png'")

# === Optional: print misclassified high-risk patients ===
print("\n--- Misclassified High-Risk Patients (Actual ICU=1, Predicted=0) ---")
misclassified = df[(df["ICU"] == 1) & (df["Predicted_Class"] == 0)]
if misclassified.empty:
    print("✅ No high-risk patients misclassified.")
else:
    print(misclassified[["PATIENT_VISIT_IDENTIFIER", "WINDOW", "Predicted_Prob"]])
