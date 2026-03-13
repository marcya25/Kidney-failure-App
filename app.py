import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="CKD Prediction System", page_icon="🩺", layout="centered")

# Custom styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.main-title{
    font-size:100px;        /* Huge title */
    color:#0A4FB3;          /* Blue */
    text-align:center;
    font-weight:900;
    margin-bottom:40px;
}

.sub-title{
    font-size:60px;         /* Very visible subtitle */
    color:#1F618D;
    font-weight:800;
    text-align:center;
    margin-bottom:30px;
}

label{
    font-size:22px !important; /* Larger input labels */
    font-weight:600;
}

.stButton>button{
    background-color:#28B463;
    color:white;
    font-size:24px;          /* Bigger button text */
    border-radius:12px;
    padding:15px 40px;
}

.stButton>button:hover{
    background-color:#239B56;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown('<p class="main-title">🩺 Chronic Kidney Disease Prediction</p>', unsafe_allow_html=True)

# Accuracy
model_accuracy = 0.94
st.success(f"Model Accuracy: {model_accuracy*100:.2f}%")

st.markdown("---")

st.markdown('<p class="sub-title">Enter Patient Medical Information</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    bp = st.number_input("Blood Pressure")
    sc = st.number_input("Serum Creatinine")
    al = st.number_input("Albumin Level")
    hemo = st.number_input("Hemoglobin")

with col2:
    dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
    htn = st.selectbox("Hypertension", ["No", "Yes"])
    appet = st.selectbox("Appetite", ["Poor", "Good"])
    ane = st.selectbox("Anemia", ["No", "Yes"])

# Convert Yes/No and Poor/Good to 0/1 for model
dm_val = 1 if dm == "Yes" else 0
htn_val = 1 if htn == "Yes" else 0
ane_val = 1 if ane == "Yes" else 0
appet_val = 1 if appet == "Good" else 0

# Prediction
if st.button("🔍 Predict CKD Risk"):

    input_data = np.array([[sc, al, hemo, bp, dm_val, htn_val, age, appet_val, ane_val]])
    input_scaled = scaler.transform(input_data)

    # Probability prediction
    risk_prob = model.predict_proba(input_scaled)[0][1]  # probability of CKD = 1
    risk_percent = risk_prob * 100

    st.markdown("---")

    if risk_percent >= 50:
        st.error(f"⚠️ High Risk of CKD: {risk_percent:.2f}%")
    else:
        st.success(f"✅ Low Risk of CKD: {risk_percent:.2f}%")
