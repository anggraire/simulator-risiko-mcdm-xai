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
""",
unsafe_allow_html=True)



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


TRAIN_MEAN = np.array(
[
80,
6
]
)


TRAIN_STD = np.array(
[
15,
3
]
)



MODEL_RMSE = 4.25




# ==========================================
# 3. FUNCTION MODULE
# ==========================================


# ---------- INPUT VALIDATION ----------

def validate_input(
    suhu,
    getaran
):

    warning=[]


    if suhu < 0 or suhu > 250:

        warning.append(
            "⚠️ Suhu diluar batas normal"
        )


    if getaran > 50:

        warning.append(
            "⚠️ Getaran terlalu tinggi"
        )


    return warning




# ---------- ML PREDICTION ----------


def predict_risk(data):


    data_scaled = scaler.transform(
        data
    )


    prediction = model_ml.predict(
        data_scaled
    )[0]


    prediction = np.clip(
        prediction,
        0,
        100
    )


    return prediction, data_scaled





# ---------- DRIFT CHECK ----------


def check_drift(data):


    drift_score = np.abs(

        (data[0]-TRAIN_MEAN)
        /
        TRAIN_STD

    )


    if np.max(drift_score) > 2:


        status = (
            "⚠️ Data Drift Detected"
        )


    else:


        status = (
            "✅ Data Stable"
        )


    return drift_score,status





# ---------- UNCERTAINTY ----------


def uncertainty_range(
    prediction,
    rmse
):


    lower=max(
        0,
        prediction-rmse
    )


    upper=min(
        100,
        prediction+rmse
    )


    return lower,upper





# ---------- ANONYMIZATION ----------


def anonymize_data(
    suhu,
    getaran
):


    raw=pd.DataFrame({

        "Nama_Operator":
        ["Budi Santoso"],


        "NIK_Petugas":
        ["3273012345"],


        "Suhu":
        [suhu],


        "Getaran":
        [getaran]


    })



    clean=raw.drop(

        columns=[

            "Nama_Operator",
            "NIK_Petugas"

        ]

    )


    return raw,clean





# ---------- SAW ----------


def calculate_saw(matrix):


    risk_norm = (

        matrix[:,0].min()
        /
        matrix[:,0]

    )


    efficiency_norm=(

        matrix[:,1]
        /
        matrix[:,1].max()

    )


    weight=np.array(
    [
        0.6,
        0.4
    ])




    score=(

        risk_norm*weight[0]

        +

        efficiency_norm*weight[1]

    )


    return score





# ---------- XAI ----------


def explain_prediction(
    scaled_data
):


    coef=model_ml.coef_


    contribution=(

        scaled_data[0]
        *
        coef

    )


    return contribution





# ==========================================
# 4. HEADER
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
# 5. INPUT WHAT-IF
# ==========================================

with left:


    st.subheader(
    "🧪 What-If Intervention"
    )


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



    st.markdown(
    """
    **Parameter Simulasi**
    
    - Suhu normal training: 60-100°C
    - Getaran normal training: 2-10 mm/s
    """
    )



    # robustness check

    warning = validate_input(
        suhu,
        getaran
    )


    for w in warning:

        st.warning(w)



    st.divider()



    st.subheader(
    "🔐 Data Privacy"
    )


    raw_data,clean_data = anonymize_data(

        suhu,

        getaran

    )



    st.write(
    "Data sebelum anonymization"
    )


    st.dataframe(
        raw_data,
        use_container_width=True
    )



    st.write(
    "Data setelah anonymization"
    )


    st.dataframe(
        clean_data,
        use_container_width=True
    )




    # robustness

    warning = validate_input(
        suhu,
        getaran
    )



    for w in warning:

        st.warning(w)



    st.divider()



    st.subheader(
    "🔐 Data Privacy"
    )



    raw_data,clean_data = anonymize_data(

        suhu,
        getaran

    )



    st.write(
    "Data sebelum anonymization"
    )


    st.dataframe(
        raw_data
    )



    st.write(
    "Data setelah anonymization"
    )


    st.dataframe(
        clean_data
    )







# ==========================================
# 6. PIPELINE ML
# ==========================================


input_data=np.array(

[
[suhu,getaran]
]

)



prediction,scaled_data = predict_risk(

    input_data

)



lower,upper = uncertainty_range(

    prediction,

    MODEL_RMSE

)





# Drift

drift,drift_status = check_drift(

    input_data

)





# ==========================================
# 7. SAW MCDM
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



score=calculate_saw(

    matrix

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


    elif prediction <70:


        st.warning(

        f"Medium Risk : {prediction:.2f}%"

        )


    else:


        st.error(

        f"High Risk : {prediction:.2f}%"

        )




    st.caption(

f"""
Model RMSE : {MODEL_RMSE}

Confidence Range :

{lower:.2f} %

-

{upper:.2f} %

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



    contribution = explain_prediction(

        scaled_data

    )



    fig,ax=plt.subplots(

        figsize=(7,3)

    )



    ax.barh(

        [
        "Suhu",
        "Getaran"
        ],

        contribution

    )



    ax.set_xlabel(

        "Feature Contribution"

    )


    st.pyplot(fig)
