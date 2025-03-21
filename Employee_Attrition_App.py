import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('attrition_model.pkl')

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👩‍💼", layout="centered")

st.title("👩‍💼 Employee Attrition Prediction App")
st.write("Enter employee details below to predict whether they are likely to leave the company.")

# Input fields
age = st.slider("Age", 18, 60, 30)
daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
department = st.selectbox("Department", ["Sales", "R&D", "HR"])
job_satisfaction = st.slider("Job Satisfaction (1-Low, 4-High)", 1, 4, 3)
overtime = st.radio("OverTime", ["Yes", "No"])
years_at_company = st.slider("Years at Company", 0, 40, 5)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=25000, value=5000)

# Convert categorical fields to numbers (example encoding)
dept_map = {"Sales": 0, "R&D": 1, "HR": 2}
overtime_map = {"Yes": 1, "No": 0}

# Predict button
if st.button("Predict Attrition"):
    input_data = np.array([
        age,
        daily_rate,
        dept_map[department],
        job_satisfaction,
        overtime_map[overtime],
        years_at_company,
        monthly_income
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. Probability: {probability:.2f}%")
    else:
        st.success(f"✅ This employee is likely to stay. Probability of leaving: {probability:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
