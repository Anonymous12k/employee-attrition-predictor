import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")

st.title("📊 Employee Attrition Predictor")
st.write("Fill in the employee details below and predict the likelihood of attrition.")

# List all input features (from your dataset minus 'Attrition')
features = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Input form with user-friendly widgets
with st.form("attrition_form"):
    input_data = {}
    for feature in features:
        if feature in ['Gender', 'OverTime']:
            input_data[feature] = st.selectbox(f"{feature}", ['0', '1'])  # Already encoded
        elif feature in ['Department', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']:
            input_data[feature] = st.number_input(f"{feature} (encoded value)", min_value=0, step=1)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    # Convert inputs to dataframe
    input_df = pd.DataFrame([input_data])

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    # Display result
    st.success(f"The predicted attrition: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Predicted probability of leaving: {probability:.2f}%")

    st.write("---")
    st.write("✅ *Model trained and deployed with Python, Scikit-learn, and Streamlit.*")

st.sidebar.write("Developed by: L KISHORE")
