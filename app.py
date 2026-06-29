import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_risiko_v1.joblib"
SCALER_PATH = MODELS_DIR / "scaler_risiko_v1.joblib"


st.set_page_config(page_title="AI Risk Simulator M16", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        font-family: Arial, sans-serif;
    }
    div[data-testid="stVerticalBlock"] > div {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error(f"Model tidak ditemukan. Pastikan file ada di: {MODELS_DIR}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model_ml, scaler = load_pipeline()

TRAIN_MEAN = np.array([80.0, 6.0], dtype=float)
TRAIN_STD = np.array([15.0, 3.0], dtype=float)
MODEL_RMSE = 4.25

st.title("⚙️ AI Risk Simulator - Optimized")
st.caption("Integrated Machine Learning + XAI + MCDM SAW + MLOps")
st.info("Aplikasi ini memprediksi risiko kegagalan mesin berdasarkan suhu dan getaran, lalu memberi rekomendasi prioritas tindakan.")

st.divider()

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🧪 What-If Intervention")
    suhu = st.number_input("Suhu Mesin (°C)", min_value=0.0, max_value=200.0, value=85.0, step=0.5)
    getaran = st.slider("Getaran Mesin", 0.0, 50.0, 7.0, 0.1)

    if suhu > 120 or suhu < 10:
        st.warning("⚠️ Input berada di luar rentang training. Risiko drift meningkat.")

    st.divider()

    st.subheader("🔐 Data Privacy")
    operator = "BUDI"
    operator_mask = operator[0] + "***"
    st.info(f"Operator : {operator_mask}\nStatus : Anonymous")

input_data = np.array([[suhu, getaran]], dtype=float)
scaled_data = scaler.transform(input_data)
prediction = float(np.clip(model_ml.predict(scaled_data)[0], 0, 100))

# Drift detection
if np.max(np.abs((input_data[0] - TRAIN_MEAN) / TRAIN_STD)) > 2:
    drift_status = "⚠️ Data Drift Detected"
else:
    drift_status = "✅ Data Stable"

if prediction < 30:
    risk_label = "Low Risk"
    risk_color = "success"
    action_text = "Pemeliharaan preventif ringan cukup untuk menjaga performa mesin."
elif prediction < 70:
    risk_label = "Medium Risk"
    risk_color = "warning"
    action_text = "Lakukan pemeriksaan rutin dan cek komponen yang berpotensi aus."
else:
    risk_label = "High Risk"
    risk_color = "error"
    action_text = "Prioritaskan inspeksi segera dan pertimbangkan tindakan korektif."

with right_col:
    st.subheader("📊 Prediction Result")
    st.progress(prediction / 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediksi Risiko", f"{prediction:.1f}%")
    col2.metric("Status", risk_label)
    col3.metric("Rentang Kepercayaan", f"{prediction - MODEL_RMSE:.1f}% - {prediction + MODEL_RMSE:.1f}%")

    if risk_color == "success":
        st.success(f"{risk_label} : {prediction:.2f}%")
    elif risk_color == "warning":
        st.warning(f"{risk_label} : {prediction:.2f}%")
    else:
        st.error(f"{risk_label} : {prediction:.2f}%")

    st.caption(f"RMSE Model : {MODEL_RMSE}")
    st.info(drift_status)
    st.write(action_text)

    drift_values = np.abs((input_data[0] - TRAIN_MEAN) / TRAIN_STD)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(["Suhu", "Getaran"], drift_values, color=["#4f46e5", "#0ea5e9"])
    ax.set_ylim(0, max(3.5, float(drift_values.max()) * 1.1))
    ax.set_title("Indikator Drift terhadap Data Training")
    ax.set_ylabel("Skor Drift")
    st.pyplot(fig)

    st.divider()
    st.subheader("🏆 Recommendation Ranking")

    alternatif = ["Mesin A", "Mesin B", "Mesin C"]
    matrix = np.array([
        [prediction, 85],
        [35, 75],
        [15, 60],
    ], dtype=float)

    risk_norm = matrix[:, 0].min() / matrix[:, 0]
    eff_norm = matrix[:, 1] / matrix[:, 1].max()
    weights = np.array([0.6, 0.4])
    score = risk_norm * weights[0] + eff_norm * weights[1]

    result = pd.DataFrame({
        "Alternatif": alternatif,
        "Risiko ML": matrix[:, 0],
        "Efisiensi": matrix[:, 1],
        "SAW Score": score,
    })
    result = result.sort_values("SAW Score", ascending=False).reset_index(drop=True)
    result["Prioritas"] = [
        "Utama" if score >= result["SAW Score"].quantile(0.67) else "Menengah" if score >= result["SAW Score"].median() else "Rendah"
        for score in result["SAW Score"]
    ]

    st.dataframe(result, use_container_width=True)
    best_option = result.iloc[0]["Alternatif"]
    st.success(f"Rekomendasi utama: {best_option} dengan skor SAW terbaik.")



    # ==================================
    # XAI
    # ==================================


    st.subheader(
    "🔍 Explainable AI"
    )


    coef=model_ml.coef_



    contribution = (

        scaled_data[0]
        *
        coef

    )



    fig,ax=plt.subplots(
    figsize=(7,3)
    )


    ax.barh(

    ["Suhu","Getaran"],

    contribution

    )


    ax.set_xlabel(
    "Feature Contribution"
    )



    st.pyplot(fig)
