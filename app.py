import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mpl_toolkits.mplot3d import Axes3D

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
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, step=1)
years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, step=1)
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
distance_from_home = st.sidebar.number_input("Distance from Home (km)", min_value=1, max_value=50, step=1)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=50000, step=500)

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

    st.info("**Key Influencing Factors:**")
    st.write("- Age and experience in the company")
    st.write("- Job satisfaction level")
    st.write("- Overtime frequency")
    st.write("- Distance from home and monthly income")

    # Add SHAP explanation (if explainer available)
    try:
        explainer = shap.Explainer(model, input_data)
        shap_values = explainer(input_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("🔎 Model Explainability (SHAP)")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')
    except:
        st.warning("SHAP explanation not available for this model type.")

st.write("---")

# Load dataset for visualization
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 3D Pie Chart visualization
st.subheader("📊 3D Pie Chart for Overall Attrition Distribution")
attrition_counts = data['Attrition'].value_counts()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.pie(attrition_counts, labels=['Stay', 'Leave'], autopct='%1.1f%%', colors=['#00cc66', '#ff5050'])
ax.set_title('3D Attrition Distribution (Note: 3D simulated with tilt)')
st.pyplot(fig)

st.sidebar.write("Developed by: L KISHORE | SIMATS Engineering")
st.sidebar.markdown("""
<div style='text-align:center;'>
    <img src='https://raw.githubusercontent.com/Anonymous12k/employee-attrition-predictor/main/your_logo.png' width='100'/>
</div>
""", unsafe_allow_html=True)
