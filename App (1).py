import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load
model = joblib.load("ٌrf_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Loan Risk Prediction 💰")

# =========================
# INPUTS
# =========================

age = st.number_input("Age")
income = st.number_input("Income")
loan = st.number_input("Loan Amount")
credit = st.number_input("Credit Score")
exp = st.number_input("Years Experience")

education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
city = st.selectbox("City", ["Urban", "Semiurban", "Rural","Houston"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])

# =========================
# ENCODING
# =========================

education_map = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

education_encoded = education_map[education]

# =========================
# CREATE DATA
# =========================

data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan],
    "CreditScore": [credit],
    "YearsExperience": [exp],
    "Education": [education_encoded],
    "City": [city],
    "EmploymentType": [employment]
})

# One-Hot للباقي
data = pd.get_dummies(data, columns=["City", "EmploymentType"], drop_first=True)

# align columns
data = data.reindex(columns=columns, fill_value=0)

# scaling
data_scaled = scaler.transform(data)

# =========================
# PREDICT
# =========================

if st.button("Predict"):
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]

    if pred == 1:
        st.success(f"✅ Approved ({proba:.2f})")
    else:
        st.error(f"❌ Rejected ({1-proba:.2f})")