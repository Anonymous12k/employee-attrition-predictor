import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the trained model
model = joblib.load("attrition_model.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Employee Attrition Predictor Dashboard", layout="wide")

# Custom banner
def add_banner():
    st.markdown("""
        <div style="background-color:#1f77b4;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">🏢 Employee Attrition Predictor & Insights</h1>
        </div>
        """, unsafe_allow_html=True)

add_banner()

# Sidebar UI
st.sidebar.header("🔎 Enter Employee Details:")
age = st.sidebar.slider("Age", min_value=18, max_value=60, step=1)
job_satisfaction = st.sidebar.radio("Job Satisfaction Level (1 - Very Dissatisfied, 4 - Very Satisfied)", [1, 2, 3, 4])
years_at_company = st.sidebar.slider("Years at Company", min_value=0, max_value=40, step=1)
overtime = st.sidebar.radio("Overtime", ["Yes", "No"])
distance_from_home = st.sidebar.slider("Distance from Home (km)", min_value=1, max_value=50, step=1)
monthly_income = st.sidebar.slider("Monthly Income (₹)", min_value=10000, max_value=100000, step=5000)

# Prediction button
if st.sidebar.button("🚀 Predict Attrition"):
    overtime_val = 1 if overtime == "Yes" else 0

    input_data = pd.DataFrame({
        'Age': [age],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company],
        'OverTime': [overtime_val],
        'DistanceFromHome': [distance_from_home],
        'MonthlyIncome': [monthly_income]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ The employee is likely to leave (Attrition Risk: {probability:.2f}%)")
    else:
        st.success(f"✅ The employee is likely to stay (Attrition Risk: {probability:.2f}%)")

    st.info("**Model considers:** Age, Job Satisfaction, Years at Company, Overtime, Distance from Home, and Monthly Income.")

st.write("---")

# Example data visualization
try:
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    st.subheader("📊 Attrition Distribution")
    labels = ['Stay', 'Leave']
    values = data['Attrition'].value_counts().values

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        pull=[0, 0.1],
        marker=dict(colors=['#00cc66', '#ff5050']),
        textinfo='label+percent',
        rotation=45
    )])
    fig_pie.update_layout(title_text="Attrition Distribution (Interactive Pie Chart)")
    st.plotly_chart(fig_pie)
except Exception as e:
    st.warning("Data for visualization not found. Upload 'WA_Fn-UseC_-HR-Employee-Attrition.csv' to see charts.")

st.sidebar.write("Developed by: L KISHORE | SIMATS Engineering")
