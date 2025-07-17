import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
final_model = joblib.load('final_xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit page setup
st.set_page_config(page_title="Cervical Cancer Prediction", layout="centered")
st.title("Cervical Cancer Prediction")

st.subheader("Enter Patient Details")

# Input fields (integers only)
age = st.number_input("Age", min_value=0, step=1)
partners = st.number_input("Number of sexual partners", min_value=0, step=1)
vaginal_condylo = st.number_input("STDs: vaginal condylomatosis (0 or 1)", min_value=0, max_value=1, step=1)
first_diag_time = st.number_input("STDs: Time since first diagnosis", min_value=0, step=1)
dx_cin = st.number_input("Dx: CIN (enter 0 or 1)", min_value=0, max_value=1, step=1)
schiller = st.number_input("Schiller (e.g., 0 or 1)", min_value=0, step=1)

# Predict button
if st.button("Predict Diagnosis"):
    # Convert input to 2D array
    input_data = np.array([[age, partners, vaginal_condylo, first_diag_time, dx_cin, schiller]])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = final_model.predict(input_scaled)[0]
    predicted_label = "Positive" if prediction == 1 else "Negative"

    # Output result
    st.write(f"Predicted Diagnosis: {predicted_label}")
