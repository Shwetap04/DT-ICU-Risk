#dashboard.py                                                                                                                import streamlit as st 
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="AI Digital Twin ICU Risk Monitor", layout="centered")
# --- Spacey Blue Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stRadio > div {
            color: white !important;
        }
        .stSidebar {
            background-color: #1e293b;
        }
        .css-1d391kg {  /* widget label */
            color: white !important;
        }
        .css-ffhzg2 {  /* radio text color fix */
            color: white !important;
        }
        .css-1r6slb0 {  /* selectbox text */
            color: white !important;
        }
        .stButton > button {
            background-color: #2563eb;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
API_URL = "http://127.0.0.1:5000/predict"
EXPECTED_FEATURE_LENGTH = 227

st.title("🧠 AI Digital Twin - ICU Risk Monitoring Dashboard")
st.markdown("📤 **Upload patient data (.xlsx with 5 rows per patient)**")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Validate required columns
    if "PATIENT_VISIT_IDENTIFIER" not in df.columns or "WINDOW" not in df.columns:
        st.error("Excel must contain 'PATIENT_VISIT_IDENTIFIER' and 'WINDOW' columns.")
    else:
        patient_ids = df["PATIENT_VISIT_IDENTIFIER"].unique()
        num_to_select = st.number_input("How many patients do you want to visualize?", min_value=1, max_value=len(patient_ids), value=1)
        selected_ids = st.multiselect("Select Patient Visit IDs", options=patient_ids, default=patient_ids[:num_to_select])

        if st.button("Predict ICU Risk"):
            fig, ax = plt.subplots()

            # Colors and markers for better distinction
            colors = plt.cm.tab10.colors
            markers = ['o', 's', '^', 'D', '*', 'X', 'P', 'v', '<', '>']

            for idx, selected_id in enumerate(selected_ids):
                patient_data = df[df["PATIENT_VISIT_IDENTIFIER"] == selected_id].sort_values("WINDOW")
                icu_risks = []

                for _, row in patient_data.iterrows():
                    raw_features = row.drop(labels=["PATIENT_VISIT_IDENTIFIER", "ICU", "WINDOW"], errors="ignore")
                    features_list = []
                    for val in raw_features:
                        try:
                            features_list.append(float(val))
                        except:
                            features_list.append(0.0)

                    if len(features_list) < EXPECTED_FEATURE_LENGTH:
                        features_list += [0.0] * (EXPECTED_FEATURE_LENGTH - len(features_list))
                    elif len(features_list) > EXPECTED_FEATURE_LENGTH:
                        features_list = features_list[:EXPECTED_FEATURE_LENGTH]

                    json_data = {"features": [features_list]}
                    try:
                        response = requests.post(API_URL, json=json_data)
                        result = response.json()
                        prob = float(result.get("probability", 0.0))
                        if not np.isfinite(prob):
                            prob = 0.0
                    except:
                        prob = 0.0

                    icu_risks.append(prob)

                # Add jitter if all values are the same to prevent overlap
                if len(set(icu_risks)) == 1:
                    icu_risks = [val + (0.005 * idx) for val in icu_risks]
                    
                # Add x-axis jitter for visual separation
                x_vals = np.arange(len(patient_data["WINDOW"]))
                x_jittered = x_vals + (idx * 0.05)  # shift each patient's line slightly

                ax.plot(
                    x_jittered,
                    icu_risks,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    label=f"Patient {selected_id}"
                )

                # Replace x-ticks with WINDOW labels
                ax.set_xticks(np.arange(len(patient_data["WINDOW"])))
                ax.set_xticklabels(patient_data["WINDOW"].values)


            ax.set_xlabel("Time Window")
            ax.set_ylabel("Predicted ICU Risk")
            ax.set_title("ICU Risk Over Time")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)     
            if selected_ids:
                st.markdown("---")
                st.subheader("📊 Sample Prediction Analysis")

                sample_id = selected_ids[0]
                sample_data = df[df["PATIENT_VISIT_IDENTIFIER"] == sample_id].sort_values("WINDOW")

                icu_risks_sample = []
                for _, row in sample_data.iterrows():
                    raw_features = row.drop(labels=["PATIENT_VISIT_IDENTIFIER", "ICU", "WINDOW"], errors="ignore")
                    features_list = []
                    for val in raw_features:
                        try:
                            features_list.append(float(val))
                        except:
                            features_list.append(0.0)

                    if len(features_list) < EXPECTED_FEATURE_LENGTH:
                        features_list += [0.0] * (EXPECTED_FEATURE_LENGTH - len(features_list))
                    elif len(features_list) > EXPECTED_FEATURE_LENGTH:
                        features_list = features_list[:EXPECTED_FEATURE_LENGTH]

                    json_data = {"features": [features_list]}
                    try:
                        response = requests.post(API_URL, json=json_data)
                        result = response.json()
                        prob = float(result.get("probability", 0.0))
                        if not np.isfinite(prob):
                            prob = 0.0
                    except:
                        prob = 0.0

                    icu_risks_sample.append(round(prob, 3))

                icu_ground_truth = sample_data["ICU"].values if "ICU" in sample_data.columns else ["N/A"] * len(sample_data)

                analysis_df = pd.DataFrame({
                    "Time Window": sample_data["WINDOW"].values,
                    "Predicted ICU Risk": icu_risks_sample,
                    "ICU Ground Truth": icu_ground_truth,
                    "Alert (Risk > 0.8)": ["🚨 Yes" if r > 0.8 else "No" for r in icu_risks_sample]
                })

                st.write(f"### Patient ID: {sample_id}")
                st.dataframe(analysis_df)

                high_alerts = analysis_df[analysis_df["Alert (Risk > 0.8)"] == "🚨 Yes"]
                if not high_alerts.empty:
                    st.success(f"✅ Alert triggered in {len(high_alerts)} out of {len(analysis_df)} time windows.")
                else:
                    st.info("No high ICU risk predicted for this patient.")

                # Optional download
                csv = analysis_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analysis as CSV",
                    data=csv,
                    file_name=f"icu_risk_analysis_patient_{sample_id}.csv",
                    mime='text/csv',
                )