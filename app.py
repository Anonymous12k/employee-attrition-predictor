import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go

# Load the trained model
model = joblib.load("attrition_model.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Employee Attrition Predictor Dashboard", layout="wide")

# Add a custom banner
def add_banner():
    st.markdown("""
        <div style="background-color:#1f77b4;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">🏢 Employee Attrition Predictor & Insights</h1>
        </div>
        """, unsafe_allow_html=True)

add_banner()

# Sidebar form inputs
st.sidebar.header("🔎 Enter Employee Details:")
age = st.sidebar.number_input("Age", min_value=18, max_value=60, step=1)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-5)", min_value=1, max_value=5, step=1)
years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, step=1)
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
distance_from_home = st.sidebar.number_input("Distance from Home (km)", min_value=1, max_value=50, step=1)
monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=10000, max_value=100000, step=5000)

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
        st.error(f"⚠️ The employee is likely to leave (Attrition Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ The employee is likely to stay (Attrition Probability: {probability:.2f}%)")

    st.info("**Key Influencing Factors Used by the Algorithm:**")
    st.write("- Age and years of experience at the company")
    st.write("- Job satisfaction level")
    st.write("- Overtime frequency (converted into binary: 1 for yes, 0 for no)")
    st.write("- Distance from home and monthly income")
    st.write("\nThe model (Logistic Regression or Random Forest) calculates the weighted influence of each feature based on learned coefficients or decision trees during training.")

    # Add SHAP explanation (if explainer available)
    try:
        explainer = shap.Explainer(model, input_data)
        shap_values = explainer(input_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("🔎 Model Explainability (SHAP Waterfall Plot)")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')
    except:
        st.warning("SHAP explanation not available for this model type.")

st.write("---")

# Load dataset for visualization
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Interactive 3D-Style Pie Chart for Attrition Distribution
st.subheader("📊 Interactive 3D-Style Pie Chart for Attrition Distribution")
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
fig_pie.update_layout(title_text="Attrition Distribution (Interactive 3D Effect)")
st.plotly_chart(fig_pie)

st.sidebar.write("Developed by: L KISHORE | SIMATS Engineering")
st.sidebar.markdown("""
<div style='text-align:center;'>
    <img src='https://raw.githubusercontent.com/Anonymous12k/employee-attrition-predictor/main/your_logo.png' width='100'/>
</div>
""", unsafe_allow_html=True)
