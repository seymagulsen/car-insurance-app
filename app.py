import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# === Visual Header Section ===

st.set_page_config(
    page_title="Car Insurance Claim Prediction | ADS 542 Project",
    page_icon="🚗",
    layout="wide"
)

logo_col, title_col = st.columns([1, 5])
with logo_col:
    tedu_logo = Image.open("tedu_logo.png")
    st.image(tedu_logo, width=120)
with title_col:
    st.markdown("""
        <h1 style='color:#003366; margin-bottom:0;'>Car Insurance Claim Prediction</h1>
        <h3 style='color:#666;'>ADS 542 Statistical Learning | Final Project</h3>
        <h5 style='color:#999;'>Created by: <b>Şeyma Gülşen Akkuş</b></h5>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### 🧠 About This App
This interactive tool is designed for predicting the likelihood that a customer will purchase car insurance based on their demographic and driving-related features.

The model has been trained on a structured dataset using machine learning techniques, and it evaluates features such as credit score, vehicle type, driving experience, and more to produce a probability score.

To use the app:
1. Fill in the customer information fields below.
2. Click the **Predict** button.
3. View the prediction result, visual interpretation, and top contributing features.
""")

st.markdown("---")

# === Custom Classes ===

class ModelBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rf_credit = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_mileage = RandomForestRegressor(n_estimators=100, random_state=42)
        self.credit_features = ['AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
                                'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
                                'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
        self.mileage_features = self.credit_features + ['CREDIT_SCORE']
        self.label_encoders = {}

    def _encode(self, df, fit=False):
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
        return df_encoded

    def fit(self, X, y=None):
        X_ = X.copy()
        credit_train = X_[X_['CREDIT_SCORE'].notnull()]
        credit_train_encoded = self._encode(credit_train[self.credit_features], fit=True)
        self.rf_credit.fit(credit_train_encoded, credit_train['CREDIT_SCORE'])

        mileage_train = X_[X_['ANNUAL_MILEAGE'].notnull()]
        mileage_train_encoded = self._encode(mileage_train[self.mileage_features], fit=True)
        self.rf_mileage.fit(mileage_train_encoded, mileage_train['ANNUAL_MILEAGE'])

        return self

    def transform(self, X):
        X_ = X.copy()

        credit_missing = X_[X_['CREDIT_SCORE'].isnull()]
        if not credit_missing.empty:
            credit_encoded = self._encode(credit_missing[self.credit_features])
            predicted_credit = self.rf_credit.predict(credit_encoded)
            X_.loc[credit_missing.index, 'CREDIT_SCORE'] = predicted_credit

        mileage_missing = X_[X_['ANNUAL_MILEAGE'].isnull()]
        if not mileage_missing.empty:
            mileage_encoded = self._encode(mileage_missing[self.mileage_features])
            predicted_mileage = self.rf_mileage.predict(mileage_encoded)
            X_.loc[mileage_missing.index, 'ANNUAL_MILEAGE'] = predicted_mileage

        return X_

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['RISK_SCORE'] = (
            X_['SPEEDING_VIOLATIONS'] +
            X_['DUIS'] * 3 +
            X_['PAST_ACCIDENTS'] * 2
        )
        mileage_median = X_['ANNUAL_MILEAGE'].median()
        X_['HIGH_MILEAGE'] = (X_['ANNUAL_MILEAGE'] > mileage_median).astype(int)
        return X_

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols if drop_cols else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols, errors='ignore')

# === Load pipeline ===
model = pickle.load(open('car_insurance_pipeline.pkl', 'rb'))

# === Session state handling for reset ===
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if st.button("Start New Prediction"):
    st.session_state.submitted = False
    st.experimental_rerun()

# === Input Form ===
with st.form("input_form"):
    st.markdown("## 📌 Customer Information Form")

    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'], help="Customer's age group (e.g., 26-39)")
        gender = st.selectbox("Gender", ['male', 'female'], help="Gender of the customer")
        race = st.selectbox("Race", ['majority', 'minority'], help="Ethnic group classification")
        driving = st.selectbox("Driving Experience", ['0-9y', '10-19y', '20-29y', '30y+'], help="Years of driving experience")

    with col2:
        education = st.selectbox("Education Level", ['none', 'high school', 'university'], help="Highest education level completed")
        income = st.selectbox("Income Bracket", ['poverty', 'working class', 'middle class', 'upper class'], help="Income category of the customer")
        credit_score = st.number_input("Credit Score", min_value=0.05, max_value=0.96, help="Credit score between 0.05 and 0.96")
        mileage = st.number_input("Annual Mileage", min_value=2000.0, max_value=22000.0, help="Miles driven annually (2000–22000)")

    col3, col4 = st.columns(2)
    with col3:
        vehicle_ownership = st.checkbox("Owns Vehicle", help="Check if customer owns a vehicle")
        vehicle_year = st.selectbox("Vehicle Year", ['before 2015', 'after 2015'], help="Year range of the customer's vehicle")
        married = st.checkbox("Married", help="Check if customer is married")
        children = st.checkbox("Has Children", help="Check if customer has children")

    with col4:
        vehicle_type = st.selectbox("Vehicle Type", ['sedan', 'sports car'], help="Type of vehicle")
        speeding = st.number_input("Speeding Violations", min_value=0, max_value=22, help="Number of speeding violations (0–22)")
        duis = st.number_input("DUIs", min_value=0, max_value=6, help="Number of DUI incidents (0–6)")
        accidents = st.number_input("Past Accidents", min_value=0, max_value=15, help="Number of past accidents (0–15)")

    submitted = st.form_submit_button("Predict")

    if submitted:
        st.session_state.submitted = True
        st.success("✅ Form submitted successfully!")
        st.balloons()
        st.experimental_rerun()
    else:
        st.session_state.submitted = False
        st.warning("⚠️ Please fill in all fields before submitting.")   
        
if st.session_state.submitted:
    data = pd.DataFrame([{ 
        'AGE': age,
        'GENDER': gender,
        'RACE': race,
        'DRIVING_EXPERIENCE': driving,
        'EDUCATION': education,
        'INCOME': income,
        'CREDIT_SCORE': credit_score,
        'VEHICLE_OWNERSHIP': vehicle_ownership,
        'VEHICLE_YEAR': vehicle_year,
        'MARRIED': married,
        'CHILDREN': children,
        'ANNUAL_MILEAGE': mileage,
        'VEHICLE_TYPE': vehicle_type,
        'SPEEDING_VIOLATIONS': speeding,
        'DUIS': duis,
        'PAST_ACCIDENTS': accidents
    }])

    probability = model.predict_proba(data)[0][1]
    prediction = model.predict(data)[0]

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**🧾 Probability of Purchasing Insurance:** `{probability:.2%}`")

    if prediction == 1:
        st.success("🟢 This customer is likely to purchase car insurance.")
        st.markdown("💡 _This customer shows strong interest in car insurance. Consider personalized offers._")
    else:
        st.warning("🟠 This customer is unlikely to purchase car insurance.")
        st.markdown("💡 _This customer may require a different marketing approach._")

    st.progress(int(probability * 100))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Purchase Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prediction == 1 else "orange"}
        }
    ))
    st.plotly_chart(fig)

    st.subheader("🔬 Feature Importances")
    try:
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)

        st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.error(f"⚠️ Feature importance visualization failed: {e}")

    with st.expander("🔎 See Input Data"):
        st.write(data)

# === Footer ===
st.markdown("---")
st.markdown("— Made by Şeyma Gülşen Akkuş", unsafe_allow_html=True)