import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="CKD Prediction System", page_icon="🩺", layout="centered")

# Custom styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">

<style>
/* Global font and background */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(to right, #f0f8ff, #e6f2ff); /* Light blue gradient */
}

/* Main Title */
.main-title {
    font-size:60px;
    color:white;
    background: linear-gradient(90deg, #0A4FB3, #003d99);
    text-align:center;
    font-weight:800;
    padding:20px;
    border-radius:15px;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* Sub Title */
.sub-title {
    font-size:32px;
    color:#003366;
    font-weight:700;
    text-align:center;
    background-color: #d9e6ff; /* Light blue background */
    padding: 15px;
    border-radius: 12px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Input labels */
label {
    font-size:20px !important;
    font-weight:500;
    color:#003366;
}

/* Buttons */
.stButton>button {
    background-color:#0A4FB3;
    color:white;
    font-size:22px;
    border-radius:12px;
    padding:12px 30px;
    font-weight:600;
}

.stButton>button:hover {
    background-color:#003d99;
    transform: scale(1.05);
    transition: all 0.3s ease;
}

/* Success/Error messages */
.stAlert {
    font-size:20px;
    font-weight:600;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

# Main Title
st.markdown('<p class="main-title">🩺 Chronic Kidney Disease Prediction</p>', unsafe_allow_html=True)

# Model Accuracy
model_accuracy = 0.94
st.success(f"Model Accuracy: {model_accuracy*100:.2f}%")

st.markdown("---")

# Sub-title
st.markdown('<p class="sub-title">Enter Patient Medical Information</p>', unsafe_allow_html=True)

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    bp = st.number_input("Blood Pressure")
    sc = st.number_input("Serum Creatinine")
    al = st.number_input("Albumin Level")
    hemo = st.number_input("Hemoglobin")

with col2:
    dm = st.selectbox("Diabetes Mellitus", [0,1])
    htn = st.selectbox("Hypertension", [0,1])
    appet = st.selectbox("Appetite (0 = Poor, 1 = Good)", [0,1])
    ane = st.selectbox("Anemia", [0,1])

st.markdown("")

# Prediction
if st.button("🔍 Predict CKD Risk"):

    input_data = np.array([[sc, al, hemo, bp, dm, htn, age, appet, ane]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Chronic Kidney Disease")
    else:
        st.success("✅ Low Risk of Chronic Kidney Disease")