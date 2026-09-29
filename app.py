"""
STEP 4: Streamlit frontend for placement prediction.

Run:  streamlit run app.py   (start the API first)
"""

import os
import requests
import streamlit as st

# Uses Render API link when deployed, localhost when running locally
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("🎓 Student Placement Predictor")
st.write("Enter student details to predict placement (Random Forest model).")

col1, col2 = st.columns(2)
with col1:
    iq = st.number_input("IQ", min_value=50, max_value=200, value=110)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    comm = st.slider("Communication Skills (1-10)", 1, 10, 7)
with col2:
    tenth = st.number_input("10th Marks (%)", min_value=0.0, max_value=100.0, value=80.0)
    twelfth = st.number_input("12th Marks (%)", min_value=0.0, max_value=100.0, value=78.0)

if st.button("Predict"):
    payload = {
        "IQ": iq,
        "CGPA": cgpa,
        "tenth_marks": tenth,
        "twelfth_marks": twelfth,
        "Communication_Skills": comm,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        result = response.json()
        if result["placed"] == 1:
            st.success(f"{result['result']}  (probability: {result['probability']:.0%})")
        else:
            st.error(f"{result['result']}  (placement probability: {result['probability']:.0%})")
    except requests.exceptions.RequestException:
        st.error("Cannot reach the API. If deployed on Render, it may be waking up — wait a minute and try again.")
