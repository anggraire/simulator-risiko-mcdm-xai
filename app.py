import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


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

html, body, [data-testid="stAppViewContainer"]{
background-color:#F8FAFC;
font-family:Arial;
}

div[data-testid="column"]{
background:white;
padding:20px;
border-radius:15px;
border:1px solid #ddd;
}

</style>

""", unsafe_allow_html=True)



# ==========================================
# 2. LOAD MODEL (MLOPS)
# ==========================================

@st.cache_resource
def load_pipeline():

    model = joblib.load(
        "models/model_risiko_v1.joblib"
    )

    scaler = joblib.load(
        "models/scaler_risiko_v1.joblib"
    )

    return model, scaler



model_ml, scaler = load_pipeline()



# baseline training reference M15

TRAIN_MEAN = np.array([
    80,
    6
])


TRAIN_STD = np.array([
    15,
    3
])



MODEL_RMSE = 4.25



# ==========================================
# 3. HEADER
# ==========================================

st.title(
"⚙️ AI Risk Simulator - Final UAS"
)


st.caption(
"Integrated Machine Learning + XAI + MCDM SAW + MLOps"
)



st.divider()



left,right = st.columns(
[1,2]
)




# ==========================================
# 4. INPUT SENSOR
# ==========================================

with left:


    st.subheader(
    "🧪 What-If Intervention"
    )


    suhu = st.number_input(
        "Suhu Mesin (°C)",
        value=85.0
    )


    getaran = st.slider(
        "Getaran Mesin",
        0.0,
        50.0,
        7.0
    )



    # robustness check M16

    if suhu > 120 or suhu < 10:

        st.warning(
        "⚠️ Input diluar distribusi training. Risiko drift meningkat."
        )



    st.divider()



    # anonymization M15

    st.subheader(
    "🔐 Data Privacy"
    )


    operator = "BUDI"


    operator_mask = (
        operator[0]
        +
        "***"
    )


    st.info(
    f"""
    Operator :
    {operator_mask}

    Status :
    Anonymous
    """
    )



# ==========================================
# 5. ML INFERENCE
# ==========================================


input_data = np.array(
[
[suhu,getaran]
]
)



scaled_data = scaler.transform(
input_data
)



prediction = model_ml.predict(
scaled_data
)[0]



prediction = np.clip(
prediction,
0,
100
)



# ==========================================
# 6. DATA DRIFT MONITORING
# ==========================================


drift = np.abs(
(input_data[0]-TRAIN_MEAN)
/
TRAIN_STD
)



if np.max(drift)>2:


    drift_status = (
    "⚠️ Data Drift Detected"
    )


else:


    drift_status = (
    "✅ Data Stable"
    )



# ==========================================
# 7. MCDM SAW
# ==========================================


alternatif=[
"Mesin A",
"Mesin B",
"Mesin C"
]



matrix=np.array(
[
[prediction,85],
[35,75],
[15,60]
]
)



# normalisasi benefit & cost

risk_norm = (
matrix[:,0].min()
/
matrix[:,0]
)


eff_norm = (
matrix[:,1]
/
matrix[:,1].max()
)



weights=np.array(
[
0.6,
0.4
]
)



score = (
risk_norm*weights[0]
+
eff_norm*weights[1]
)



result=pd.DataFrame({

"Alternatif":alternatif,

"Risiko ML":matrix[:,0],

"Efisiensi":matrix[:,1],

"SAW Score":score

})


result=result.sort_values(
"SAW Score",
ascending=False
)



# ==========================================
# 8. OUTPUT
# ==========================================


with right:


    st.subheader(
    "📊 Prediction Result"
    )


    st.progress(
    prediction/100
    )



    if prediction <30:

        st.success(
        f"Low Risk : {prediction:.2f}%"
        )


    elif prediction<70:

        st.warning(
        f"Medium Risk : {prediction:.2f}%"
        )


    else:

        st.error(
        f"High Risk : {prediction:.2f}%"
        )



    st.caption(

    f"""
    RMSE Model : {MODEL_RMSE}

    Confidence Range :
    {prediction-MODEL_RMSE:.2f}
    -
    {prediction+MODEL_RMSE:.2f}

    """

    )



    st.info(
    drift_status
    )



    st.divider()



    st.subheader(
    "🏆 Recommendation Ranking"
    )



    st.dataframe(
        result,
        use_container_width=True
    )



    st.divider()



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


