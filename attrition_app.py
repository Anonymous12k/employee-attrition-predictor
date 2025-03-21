import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for visuals
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

st.set_page_config(page_title="Employee Attrition Predictor & Visuals", page_icon="📊", layout="wide")

# Add custom company branding banner
def add_banner():
    st.markdown("""
        <div style="background-color:#2c3e50;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">🏢 STRAK Indus - Employee Attrition Predictor Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)

add_banner()

st.write("Fill in the employee details below and predict the likelihood of attrition.")

# Sidebar Input fields with styled headers
st.sidebar.header("🔎 Enter Employee Details:")
age = st.sidebar.slider("Age", 18, 60, 30)
overtime = st.sidebar.radio("OverTime", ["Yes", "No"])
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 50000, 5000)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
distance_from_home = st.sidebar.slider("Distance from Home (km)", 1, 50, 10)

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

    st.markdown("## 📊 Prediction Output")

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. Probability: {probability:.2f}%")
    else:
        st.success(f"✅ This employee is likely to stay. Probability of leaving: {probability:.2f}%")

    # Visual insights on prediction
    st.info("**Key Factors influencing prediction:**")
    st.write("- Age: Younger employees may have higher turnover.")
    st.write("- Job Satisfaction: Low satisfaction increases risk.")
    st.write("- OverTime: Frequent overtime can increase attrition.")
    st.write("- Monthly Income: Lower income groups show higher attrition.")

st.write("---")

# Visualizations
st.subheader("📈 Dataset Insights")

# Pie Chart for overall attrition distribution
attrition_counts = data['Attrition'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(attrition_counts, labels=['Stay', 'Leave'], autopct='%1.1f%%', colors=['#00cc66', '#ff5050'])
ax1.set_title('Overall Attrition Distribution')
st.pyplot(fig1)

# Bar chart: Average monthly income by Job Role
st.subheader("💰 Average Monthly Income by Job Role")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x='JobRole', y='MonthlyIncome', data=data, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_ylabel("Monthly Income")
ax2.set_xlabel("Job Role")
st.pyplot(fig2)

# Correlation heatmap using only numeric columns
st.subheader("📊 Feature Correlation Heatmap")
numeric_data = data.select_dtypes(include=['int64', 'float64'])
fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=False, cmap="coolwarm", ax=ax3)
ax3.set_title("Correlation Between Features")
st.pyplot(fig3)

st.sidebar.write("Developed by: L KISHORE | SIMATS Engineering")
