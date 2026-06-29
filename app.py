import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Risk Simulator M15-M16",
    page_icon="⚙️",
    layout="wide"
)


st.markdown("""
<style>

/* =========================================
   LIGHT MODE (default)
   Palette: #FBEFEF / #FFE2E2 / #F5CBCB
   ========================================= */

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #FBEFEF !important;
    font-family: Arial;
}

[data-testid="stHeader"] {
    background-color: #FBEFEF !important;
}

[data-testid="stSidebar"] {
    background-color: #FFE2E2 !important;
}

div[data-testid="column"] {
    background: #FFE2E2 !important;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #F5CBCB;
}

/* General text */
html body p,
html body label,
html body span,
html body div {
    color: #3B1A1E;
}

/* Subheaders & headings */
h1, h2, h3, h4, h5, h6 {
    color: #7B2D35 !important;
}

/* Title */
[data-testid="stTitle"] {
    color: #7B2D35 !important;
}

/* Caption */
[data-testid="stCaptionContainer"],
.stCaption {
    color: #9E5560 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #7B2D35;
}

/* Divider */
hr {
    border-color: #F5CBCB !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #F5CBCB;
    border-radius: 8px;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Sliders thumb */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #C0535A !important;
}

/* Progress bar fill */
[data-testid="stProgress"] > div > div > div {
    background-color: #C0535A !important;
}

/* Primary buttons */
button[kind="primary"] {
    background-color: #C0535A !important;
    border-color: #C0535A !important;
    color: white !important;
}

/* Checkbox and radio */
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #7B2D35;
}


/* =========================================
   DARK MODE
   Uses deeper/muted rose tones so palette
   stays consistent but readable on dark bg
   ========================================= */

@media (prefers-color-scheme: dark) {

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #2A1117 !important;
    }

    [data-testid="stHeader"] {
        background-color: #2A1117 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #3D1820 !important;
    }

    div[data-testid="column"] {
        background: #3D1820 !important;
        border: 1px solid #6B3039 !important;
    }

    /* Text */
    html body p,
    html body label,
    html body span,
    html body div {
        color: #F5CBCB !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFB8BE !important;
    }

    [data-testid="stTitle"] {
        color: #FFB8BE !important;
    }

    [data-testid="stCaptionContainer"],
    .stCaption {
        color: #E8848A !important;
    }

    [data-testid="stMetricLabel"] {
        color: #FFB8BE !important;
    }

    hr {
        border-color: #6B3039 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #6B3039 !important;
        background-color: #3D1820 !important;
    }

    /* Slider & Progress keep rose accent */
    [data-testid="stSlider"] > div > div > div > div {
        background-color: #E8848A !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background-color: #E8848A !important;
    }

    button[kind="primary"] {
        background-color: #E8848A !important;
        border-color: #E8848A !important;
        color: #2A1117 !important;
    }

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #F5CBCB !important;
    }

}

</style>
""",
unsafe_allow_html=True)


# ==========================================
# HELPER: detect theme for matplotlib
# ==========================================

def get_mpl_colors():
    """Return chart colors based on Streamlit theme."""
    theme = st.get_option("theme.base")
    if theme == "dark":
        return {
            "fig_bg":   "#3D1820",
            "ax_bg":    "#2A1117",
            "bar1":     "#E8848A",
            "bar2":     "#C0535A",
            "edge":     "#6B3039",
            "text":     "#F5CBCB",
            "spine":    "#6B3039",
        }
    else:
        return {
            "fig_bg":   "#FFE2E2",
            "ax_bg":    "#FBEFEF",
            "bar1":     "#C0535A",
            "bar2":     "#E8848A",
            "edge":     "#F5CBCB",
            "text":     "#7B2D35",
            "spine":    "#F5CBCB",
        }


# ==========================================
# 2. LOAD MODEL (MLOPS)
# ==========================================

@st.cache_resource
def load_pipeline():

    try:

        model = joblib.load(
            "models/model_risiko_v1.joblib"
        )

        scaler = joblib.load(
            "models/scaler_risiko_v1.joblib"
        )

        return model, scaler

    except Exception as e:

        st.error(
            f"Model gagal dimuat : {e}"
        )

        st.stop()


model_ml, scaler = load_pipeline()


# ==========================================
# TRAINING REFERENCE
# ==========================================

TRAIN_MEAN = np.array([80, 6])
TRAIN_STD  = np.array([15, 3])
MODEL_RMSE = 4.25


# ==========================================
# 3. FUNCTION MODULE
# ==========================================

def validate_input(suhu, getaran):
    warning = []
    if suhu < 0 or suhu > 250:
        warning.append("⚠️ Suhu diluar batas normal")
    if getaran > 50:
        warning.append("⚠️ Getaran terlalu tinggi")
    return warning


def predict_risk(data):
    data_scaled = scaler.transform(data)
    prediction  = model_ml.predict(data_scaled)[0]
    prediction  = np.clip(prediction, 0, 100)
    return prediction, data_scaled


def check_drift(data):
    drift_score = np.abs((data[0] - TRAIN_MEAN) / TRAIN_STD)
    status = (
        "⚠️ Data Drift Detected"
        if np.max(drift_score) > 2
        else "✅ Data Stable"
    )
    return drift_score, status


def uncertainty_range(prediction, rmse):
    lower = max(0,   prediction - rmse)
    upper = min(100, prediction + rmse)
    return lower, upper


def anonymize_data(suhu, getaran):
    raw = pd.DataFrame({
        "Nama_Operator": ["Budi Santoso"],
        "NIK_Petugas":   ["3273012345"],
        "Suhu":          [suhu],
        "Getaran":       [getaran],
    })
    clean = raw.drop(columns=["Nama_Operator", "NIK_Petugas"])
    return raw, clean


def calculate_saw(matrix):
    risk_norm       = matrix[:, 0].min() / matrix[:, 0]
    efficiency_norm = matrix[:, 1] / matrix[:, 1].max()
    weight          = np.array([0.6, 0.4])
    score           = risk_norm * weight[0] + efficiency_norm * weight[1]
    return score


def explain_prediction(scaled_data):
    coef         = model_ml.coef_
    contribution = scaled_data[0] * coef
    return contribution


# ==========================================
# 4. HEADER
# ==========================================

st.title("⚙️ AI Risk Simulator - Final UAS")
st.caption("Integrated Machine Learning + XAI + MCDM SAW + MLOps")
st.divider()

left, right = st.columns([1, 2])


# ==========================================
# 5. INPUT WHAT-IF
# ==========================================

with left:

    st.subheader("🧪 What-If Intervention")

    suhu = st.slider(
        "🌡️ Suhu Mesin (°C)",
        min_value=0.0, max_value=250.0, value=85.0, step=1.0
    )

    getaran = st.slider(
        "🔩 Getaran Mesin (mm/s)",
        min_value=0.0, max_value=50.0, value=7.0, step=0.1
    )

    for w in validate_input(suhu, getaran):
        st.warning(w)

    st.divider()

    st.subheader("🔐 Data Privacy")

    raw_data, clean_data = anonymize_data(suhu, getaran)

    st.write("Data sebelum anonymization")
    st.dataframe(raw_data)

    st.write("Data setelah anonymization")
    st.dataframe(clean_data)


# ==========================================
# 6. PIPELINE ML
# ==========================================

input_data           = np.array([[suhu, getaran]])
prediction, scaled_data = predict_risk(input_data)
lower, upper         = uncertainty_range(prediction, MODEL_RMSE)
drift, drift_status  = check_drift(input_data)


# ==========================================
# 7. SAW MCDM
# ==========================================

alternatif = ["Mesin A", "Mesin B", "Mesin C"]

matrix = np.array([
    [prediction, 85],
    [35,         75],
    [15,         60],
])

score = calculate_saw(matrix)

result = pd.DataFrame({
    "Alternatif": alternatif,
    "Risiko ML":  matrix[:, 0],
    "Efisiensi":  matrix[:, 1],
    "SAW Score":  score,
}).sort_values("SAW Score", ascending=False)


# ==========================================
# 8. OUTPUT
# ==========================================

with right:

    st.subheader("📊 Prediction Result")

    st.progress(prediction / 100)

    if prediction < 30:
        st.success(f"Low Risk : {prediction:.2f}%")
    elif prediction < 70:
        st.warning(f"Medium Risk : {prediction:.2f}%")
    else:
        st.error(f"High Risk : {prediction:.2f}%")

    st.caption(
        f"Model RMSE : {MODEL_RMSE}  |  "
        f"Confidence Range : {lower:.2f}% – {upper:.2f}%"
    )

    st.info(drift_status)

    st.divider()

    st.subheader("🏆 Recommendation Ranking")
    st.dataframe(result, use_container_width=True)

    st.divider()

    # ==================================
    # XAI — chart adapts to theme
    # ==================================

    st.subheader("🔍 Explainable AI")

    contribution = explain_prediction(scaled_data)
    c = get_mpl_colors()

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(c["fig_bg"])
    ax.set_facecolor(c["ax_bg"])

    ax.barh(
        ["Suhu", "Getaran"],
        contribution,
        color=[c["bar1"], c["bar2"]],
        edgecolor=c["edge"],
        linewidth=0.8,
    )

    ax.set_xlabel("Feature Contribution", color=c["text"])
    ax.tick_params(colors=c["text"])

    for spine in ax.spines.values():
        spine.set_color(c["spine"])

    fig.tight_layout()
    st.pyplot(fig)
