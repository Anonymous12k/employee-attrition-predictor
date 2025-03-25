import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Predictor")

# Create input form
st.sidebar.header("Enter Employee Details:")

# Example input fields (Add more based on your model features)
age = st.sidebar.number_input("Age", min_value=18, max_value=60, step=1)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, step=1)
years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, step=1)
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
distance_from_home = st.sidebar.number_input("Distance from Home (km)", min_value=1, max_value=50, step=1)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=50000, step=500)

if st.sidebar.button("Predict"):
    # Convert overtime to numeric if needed
    overtime_val = 1 if overtime == "Yes" else 0

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company],
        'OverTime': [overtime_val],
        'DistanceFromHome': [distance_from_home],
        'MonthlyIncome': [monthly_income]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ The employee is likely to leave (Attrition Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ The employee is likely to stay (Attrition Risk: {probability*100:.2f}%)")

st.write("Note: This prediction is based on the model you trained. Please make sure your model file is in the same folder (employee_attrition_model.pkl).")
