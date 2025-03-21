import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1. Load the dataset
data = pd.read_csv(r"C:\Users\LENOVO\Downloads\employee_attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Preprocessing
# Encode categorical features
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# 3. Feature and target selection
X = data.drop('Attrition', axis=1)
y = data['Attrition']  # Attrition is already encoded (0 or 1 after LabelEncoder)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train models and compare
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# 7. Save the best model and scaler
joblib.dump(best_model, 'attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest model saved as 'attrition_model.pkl' with accuracy {best_accuracy}")
