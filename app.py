import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & THEME CUSTOMIZATION
# ==========================================

st.set_page_config(
    page_title="AI Risk Simulator M15-M16",
    page_icon="⚙️",
    layout="wide"
)

# Kustomisasi CSS untuk tema warna pastel yang lebih estetik dan modern
st.markdown("""
<style>
    /* Background utama aplikasi */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FBEFEF;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #4A3B3B;
    }

    /* Efek Kartu Modern (Floating Cards) untuk Kolom */
    div[data-testid="column"] {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #FFE2E2;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    /* Percantik tampilan dataframe */
    .stDataFrame {
        border: 1px solid #F5CBCB;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD MODEL (MLOPS)
# ==========================================

@st.cache_resource
def load_pipeline():
    try:
        model = joblib.load("models/model_risiko_v1.joblib")
        scaler = joblib.load("models/scaler_risiko_v1.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Model gagal dimuat: {e}")
        st.stop()

model_ml, scaler = load_pipeline()


# ==========================================
# TRAINING REFERENCE & CONSTANTS
# ==========================================

TRAIN_MEAN = np.array([80, 6])
TRAIN_STD = np.array([15, 3])
MODEL_RMSE = 4.25


# ==========================================
# 3. FUNCTION MODULES
# ==========================================

def validate_input(suhu, getaran):
    warning = []
    if suhu < 0 or suhu > 250:
        warning.append("⚠️ Suhu di luar batas normal operasional!")
    if getaran > 50:
        warning.append("⚠️ Tingkat getaran terlalu tinggi dan berbahaya!")
    return warning


def predict_risk(data):
    data_scaled = scaler.transform(data)
    prediction = model_ml.predict(data_scaled)[0]
    prediction = np.clip(prediction, 0, 100)
    return prediction, data_scaled


def check_drift(data):
    drift_score = np.abs((data[0] - TRAIN_MEAN) / TRAIN_STD)
    if np.max(drift_score) > 2:
        status = "⚠️ Data Drift Detected (Karakteristik data berubah)"
    else:
        status = "✅ Data Stable (Karakteristik data sesuai training)"
    return drift_score, status


def uncertainty_range(prediction, rmse):
    lower = max(0, prediction - rmse)
    upper = min(100, prediction + rmse)
    return lower, upper


def anonymize_data(suhu, getaran):
    raw = pd.DataFrame({
        "Nama_Operator": ["Budi Santoso"],
        "NIK_Petugas": ["3273012345"],
        "Suhu": [suhu],
        "Getaran": [getaran]
    })
    clean = raw.drop(columns=["Nama_Operator", "NIK_Petugas"])
    return raw, clean


def calculate_saw(matrix):
    # Urutan kolom di matrix: [Risiko (Cost), Efisiensi (Benefit)]
    risk_norm = matrix[:, 0].min() / matrix[:, 0]
    efficiency_norm = matrix[:, 1] / matrix[:, 1].max()
    
    weight = np.array([0.6, 0.4])
    score = (risk_norm * weight[0]) + (efficiency_norm * weight[1])
    return score


def explain_prediction(scaled_data):
    coef = model_ml.coef_
    contribution = scaled_data[0] * coef
    return contribution


# ==========================================
# 4. HEADER
# ==========================================

st.title("⚙️ AI Risk Simulator - Final UAS")
st.caption("Integrated Machine Learning + XAI + MCDM SAW + MLOps Dashboard")
st.divider()

# Layout utama menggunakan pembagian kolom otomatis yang responsif
left, right = st.columns([1, 1.4], gap="large")


# ==========================================
# 5. INPUT & WHAT-IF INTERVENTION (LEFT)
# ==========================================

with left:
    st.subheader("🧪 What-If Intervention")
    
    suhu = st.slider(
        "🌡️ Suhu Mesin (°C)",
        min_value=0.0,
        max_value=250.0,
        value=85.0,
        step=1.0
    )

    getaran = st.slider(
        "🔩 Getaran Mesin (mm/s)",
        min_value=0.0,
        max_value=50.0,
        value=7.0,
        step=0.1
    )

    # Validasi input input secara real-time
    warning = validate_input(suhu, getaran)
    for w in warning:
        st.warning(w)

    st.divider()

    st.subheader("🔐 Data Privacy & Anonymization")
    raw_data, clean_data = anonymize_data(suhu, getaran)

    # Menggunakan tabs agar layout data privacy terlihat lebih rapi dan ringkas
    tab1, tab2 = st.tabs(["📄 Raw Data Log", "✨ Anonymized Data"])
    with tab1:
        st.dataframe(raw_data, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(clean_data, use_container_width=True, hide_index=True)


# ==========================================
# 6. PROCESSING PIPELINE
# ==========================================

input_data = np.array([[suhu, getaran]])
prediction, scaled_data = predict_risk(input_data)
lower, upper = uncertainty_range(prediction, MODEL_RMSE)
drift, drift_status = check_drift(input_data)


# ==========================================
# 7. SAW MCDM CALCULATION
# ==========================================

alternatif = ["Mesin A", "Mesin B", "Mesin C"]
matrix = np.array([
    [prediction, 85],  # Nilai prediksi risiko real-time disuntikkan ke Mesin A
    [35, 75],
    [15, 60]
])

score = calculate_saw(matrix)
result = pd.DataFrame({
    "Alternatif": alternatif,
    "Risiko ML (%)": matrix[:, 0],
    "Efisiensi (%)": matrix[:, 1],
    "SAW Score": score
}).sort_values("SAW Score", ascending=False)


# ==========================================
# 8. OUTPUT & VISUALIZATION (RIGHT)
# ==========================================

with right:
    st.subheader("📊 Prediction Result")
    
    # Progress bar risiko mesin
    st.progress(prediction / 100)

    # Status Alert Maker yang dinamis berdasarkan level risiko
    if prediction < 30:
        st.success(f"**Low Risk Zone**: {prediction:.2f}%")
    elif prediction < 70:
        st.warning(f"**Medium Risk Zone**: {prediction:.2f}%")
    else:
        st.error(f"**High Risk Critical Zone**: {prediction:.2f}%")

    # Informasi metrik performa model & interval kepercayaan dalam bentuk kolom ringkas
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Confidence Interval", value=f"{lower:.1f}% - {upper:.1f}%")
    with m2:
        st.metric(label="Model Error (RMSE)", value=f"{MODEL_RMSE:.2f}%")

    # Status Data Drift dari sisi MLOps
    st.info(drift_status)
    st.divider()

    # Tampilan Rangking Rekomendasi SAW
    st.subheader("🏆 Multi-Criteria Recommendation (SAW)")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.divider()

    # Visualisasi Kontribusi Fitur Menggunakan Tema Warna Selaras
    st.subheader("🔍 Explainable AI (Feature Contribution)")
    contribution = explain_prediction(scaled_data)

    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor='white')
    ax.set_facecolor('white')
    
    # Menggunakan warna #F5CBCB untuk kontribusi positif dan abu-abu lembut untuk negatif
    colors = ['#F5CBCB' if c >= 0 else '#D1C4C4' for c in contribution]
    
    bars = ax.barh(["Suhu", "Getaran"], contribution, color=colors, height=0.55)
    
    # Merapikan dekorasi grafik agar terlihat clean & minimalis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4A3B3B')
    ax.spines['bottom'].set_color('#4A3B3B')
    ax.tick_params(colors='#4A3B3B')
    ax.set_xlabel("Contribution Score", color='#4A3B3B', fontsize=10)
    ax.axvline(0, color='#4A3B3B', linewidth=0.8, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)
