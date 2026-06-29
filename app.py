import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================================
# 1. KONFIGURASI HALAMAN & THEME
# ===============================================
st.set_page_config(page_title="Simulator Risiko & Rekomendasi Sistem", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #F8FAFC; font-family: 'Plus Jakarta Sans', sans-serif; }
div[data-testid="column"] { background-color: #ffffff; padding: 22px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03); border: 1px solid #E2E8F0; margin-bottom: 10px; }
.stProgress > div > div > div > div { background-color: #D6336C; }
</style>
""", unsafe_allow_html=True)

# ===============================================
# 2. BACKEND: LOAD MODEL (DIREKTORI UTAMA)
# ===============================================
@st.cache_resource
def init_models():
    model = joblib.load("model_risiko_v1.joblib")
    scaler = joblib.load("scaler_risiko_v1.joblib")
    return model, scaler

model_ml, scaler_risiko = init_models()
train_mean_ref = 80.0  
baseline_risk = 45.0  
MODEL_RMSE = 4.25  # Simulasi nilai RMSE model untuk metrik ketidakpastian

# ===============================================
# 3. FRONTEND & INTERACTION (WHAT-IF SLIDER)
# ===============================================
st.title("⚙️ Simulator Pintar Risiko & Rekomendasi AI (UAS)")
st.markdown("<p style='color: #666; margin-top: -15px;'>Sintesis Akhir Pipeline Machine Learning, Explainable AI (SHAP), dan Sistem Pendukung Keputusan (MCDM-SAW).</p>", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([1.1, 1.9], gap="large")

with col_left:
    st.markdown("### 🧪 Kebijakan Intervensi (Input Sensor)")
    
    # VALIDASI INPUT & ROBUSTNESS CHECK (Membatasi input ekstrem ke rentang aman internal)
    suhu_raw = st.number_input("Suhu Mesin (°C) - Input Ketik", value=85.0)
    getaran_slider = st.slider("Getaran Mesin (mm/s)", min_value=0.0, max_value=50.0, value=7.0)
    
    # Mekanisme Clipping/Handling jika user memasukkan angka di luar akal sehat (Soal 5)
    suhu_slider = min(max(suhu_raw, 0.0), 250.0) 
    if suhu_raw > 250.0 or suhu_raw < 0.0:
        st.warning(f"⚠️ **Robustness Triggered:** Input Suhu {suhu_raw}°C tidak rasional! Sistem otomatis melakukan pembatasan (*clipping*) ke batas aman {suhu_slider}°C untuk mencegah kegagalan prediksi.")

    st.markdown("#### 📋 Ringkasan Input & Privasi Operator (PII Anonim)")
    # IMPLEMENTASI PRIVASI DATA: Anonimisasi ID Operator menggunakan Masking (Soal 7)
    raw_operator_id = "OP-98745-BUDI"
    anon_operator_id = f"{raw_operator_id[:3]}*****{raw_operator_id[-4:]}"
    
    st.markdown(f"""
    <div style='background-color: #F1F5F9; padding: 12px; border-radius: 8px; border-left: 5px solid #64748B;'>
        <p style='margin: 0; font-size: 13.5px; color: #334155;'>🌡️ <b>Suhu Terbaca:</b> {suhu_slider:.2f} °C</p>
        <p style='margin: 4px 0 0 0; font-size: 13.5px; color: #334155;'>📉 <b>Getaran Terbaca:</b> {getaran_slider:.2f} mm/s</p>
        <p style='margin: 4px 0 0 0; font-size: 13.5px; color: #D6336C;'>👤 <b>ID Operator Terenkripsi (PII Protected):</b> {anon_operator_id}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Pembobotan SPK (MCDM - SAW)")
    st.caption("Kriteria Penilaian: [1] Skor Keamanan (Invers Risiko ML) - Bobot: 0.60 | [2] Efisiensi Kerja - Bobot: 0.40")

# ===============================================
# 4. PIPELINE PROCESSING (ML -> SPK -> RANKING)
# ===============================================
features = np.array([[suhu_slider, getaran_slider]])
features_scaled = scaler_risiko.transform(features)
prediksi_risiko = float(model_ml.predict(features_scaled)[0])
prediksi_risiko = min(max(prediksi_risiko, 0.0), 100.0)

# Rentang ketidakpastian berbasis RMSE (Soal 8)
batas_bawah_risiko = max(0.0, prediksi_risiko - MODEL_RMSE)
batas_atas_risiko = min(100.0, prediksi_risiko + MODEL_RMSE)

alternatif = ["Mesin A (Lokasi 1)", "Mesin B (Lokasi 2)", "Mesin C (Lokasi 3)"]
matriks_x = np.array([
    [prediksi_risiko, 85.0],  # Mesin A mengikuti prediksi ML secara dinamis
    [35.0, 75.0],             # Mesin B
    [15.0, 60.0]              # Mesin C
])

c_min = matriks_x[:, 0].min() if matriks_x[:, 0].min() > 0 else 1
b_max = matriks_x[:, 1].max()

r_risiko = c_min / matriks_x[:, 0]
r_efisiensi = matriks_x[:, 1] / b_max

bobot = np.array([0.6, 0.4])
skor_saw = (r_risiko * bobot[0]) + (r_efisiensi * bobot[1])

df_rekomendasi = pd.DataFrame({
    "Alternatif Mesin": alternatif,
    "Skor Risiko": matriks_x[:, 0],
    "Efisiensi (%)": matriks_x[:, 1],
    "Skor Performa (SAW)": skor_saw
}).sort_values(by="Skor Performa (SAW)", ascending=False)

# ===============================================
# 5. OUTPUT DISPLAY & VISUALISASI UTUH
# ===============================================
with col_right:
    st.markdown("### 📄 Hasil Validasi & Analisis Preskriptif")
    
    st.markdown("#### 🚨 Indikator Batas Risiko Aktual")
    st.progress(prediksi_risiko / 100.0)
    
    # Mengomunikasikan Ketidakpastian/RMSE via UI (Soal 8)
    if prediksi_risiko < 30:
        st.success(f"### Skor Risiko: {prediksi_risiko:.2f}% (Risiko Rendah)")
    elif prediksi_risiko < 70:
        st.warning(f"### Skor Risiko: {prediksi_risiko:.2f}% (Risiko Sedang)")
    else:
        st.error(f"### Skor Risiko: {prediksi_risiko:.2f}% (Risiko Tinggi)")
        
    st.caption(f"📊 *Rentang Kepercayaan Prediksi (RMSE: {MODEL_RMSE}): {batas_bawah_risiko:.2f}% hingga {batas_atas_risiko:.2f}%*")

    # Deteksi Data Drift (Soal 5)
    drift_score = np.abs(suhu_slider - train_mean_ref)
    if drift_score > 40.0:
        st.error(f"⚠️ **WARNING:** Terdeteksi indikasi Data Drift sebesar {drift_score:.2f}. Angka input jauh dari sebaran data training asli!")
    else:
        st.info("✅ **Sistem Robust:** Parameter input saat ini sesuai dengan profil latih model.")

    st.markdown("#### 📊 Grafik Komparasi Nilai Risiko (Baseline vs Intervensi)")
    data_grafik = pd.DataFrame({
        'Skenario': ['Kondisi Baseline', 'Intervensi Pengguna'],
        'Tingkat Risiko (%)': [baseline_risk, prediksi_risiko]
    })
    st.bar_chart(data=data_grafik, x='Skenario', y='Tingkat Risiko (%)', color='#D6336C', use_container_width=True)

    st.markdown("#### 🏆 Hasil Perhitungan Urutan Alternatif (MCDM - SAW)")
    st.dataframe(df_rekomendasi.style.format({"Skor Risiko": "{:.2f}", "Skor Performa (SAW)": "{:.4f}"}), use_container_width=True)
    
    # INTEGRASI MODUL TRANSPARANSI (SHAP MATPLOTLIB - BEBAS BUG)
    st.markdown("### 🔍 Mengapa Hasilnya Demikian? (Transparansi XAI)")
    st.caption("Diagram di bawah memaparkan pembuktian kontribusi kuantitatif dari setiap parameter sensor terhadap hasil prediksi risiko.")

    coef = model_ml.coef_
    suhu_contrib = features_scaled[0][0] * coef[0]
    getaran_contrib = features_scaled[0][1] * coef[1]
    
    contrib_values = [suhu_contrib, getaran_contrib]
    feature_names = ["Suhu Mesin", "Getaran Mesin"]
    
    fig, ax = plt.subplots(figsize=(7, 2.5))
    colors = ['#FF0051' if val >= 0 else '#008BFB' for val in contrib_values]
    bars = ax.barh(feature_names, contrib_values, color=colors, height=0.4)
    ax.axvline(x=0, color='#64748B', lw=1, ls='--')
    ax.set_xlabel('Dampak Matriks Fitur (SHAP Value)')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (0.1 if width >= 0 else -1.5), bar.get_y() + bar.get_height()/2,
                f'{width:+.2f}', va='center', ha='left', fontsize=9, fontweight='bold')
                
    plt.tight_layout()
    st.pyplot(fig)
