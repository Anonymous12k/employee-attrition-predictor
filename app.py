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
        <h1 style="color:white;text-align:center;">🏢 STRAK Industries - Employee Attrition Predictor Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)

add_banner()

st.write("Fill in the employee details below to predict the likelihood of attrition.")

# Sidebar Input fields with styled headers
st.sidebar.header("🔎 Enter Employee Details:")
age = st.sidebar.slider("Age", 18, 60, 30)
overtime = st.sidebar.radio("OverTime", ["Yes", "No"])
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 100000, 5000)
job_satisfaction = st.sidebar.radio("Job Satisfaction (1-4)", [1, 2, 3, 4])
distance_from_home = st.sidebar.slider("Distance from Home (km)", 1, 50, 10)
education_field = st.sidebar.selectbox("Education Field", data['EducationField'].unique())
job_role = st.sidebar.selectbox("Job Role", data['JobRole'].unique())

if st.sidebar.button("🚀 Predict Attrition"):
    overtime_val = 1 if overtime == "Yes" else 0
    input_data = pd.DataFrame({
        'Age': [age],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company],
        'OverTime': [overtime_val],
        'DistanceFromHome': [distance_from_home],
        'MonthlyIncome': [monthly_income],
        'EducationField': [education_field],
        'JobRole': [job_role]
    })

    # Perform one-hot encoding for new categorical columns
    input_encoded = pd.get_dummies(input_data)
    model_columns = joblib.load("model_columns.pkl")  # Contains columns model was trained on
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1] * 100

    st.markdown("## 📊 Prediction Output")

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. Probability: {probability:.2f}%")
    else:
        st.success(f"✅ This employee is likely to stay. Probability of leaving: {probability:.2f}%")

    st.info("**Key Factors influencing prediction:**")
    st.write("- Age: Younger employees may have higher turnover.")
    st.write("- Job Satisfaction: Low satisfaction increases risk.")
    st.write("- OverTime: Frequent overtime can increase attrition.")
    st.write("- Monthly Income: Lower income groups show higher attrition.")
    st.write("- Job Role and Education Field: Certain roles and fields show higher tendencies.")

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

# Countplot: Attrition by Education Field
st.subheader("🎓 Attrition Count by Education Field")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.countplot(x='EducationField', hue='Attrition', data=data, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.set_ylabel("Count")
ax3.set_xlabel("Education Field")
st.pyplot(fig3)

# Correlation heatmap using only numeric columns
st.sub
