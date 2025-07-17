# train_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data (use original_data.xlsx)
df = pd.read_excel("data/original_data.xlsx")

# Features and labels
features = df.drop(columns=["ICU"])  # Adjust if ICU column has a different name
labels = df["ICU"]

# Group data by patient ID if needed and take last 5 rows (assuming already grouped correctly)
# For simplicity, assuming input is reshaped to (num_samples, 228)
X = features.to_numpy().reshape(-1, 228)
y = labels.to_numpy()[::5]  # pick 1 ICU label per patient (assuming 5 rows per patient)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "C:/Users/Shweta/Documents/icu_dashboard_project/model/scaler.pkl")

# Reshape for CNN input: (samples, timesteps, features)
X_cnn = X_scaled.reshape(-1, 228, 1)

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(228, 1)),
    Dropout(0.2),
    Conv1D(64, kernel_size=2, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_cnn, y, epochs=10, batch_size=32, validation_split=0.2,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# Save the trained model
model.save("models/cnn_icu_model.h5")
