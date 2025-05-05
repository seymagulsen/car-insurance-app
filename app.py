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

# Set page config
st.set_page_config(
    page_title="Car Insurance Claim Prediction | ADS 542 Project",
    page_icon="🚗",
    layout="wide"
)

# TEDU Logo and Title
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


# === Sidebar Input ===
st.sidebar.header("Customer Information")

def user_input():
    data = {
        'AGE': st.sidebar.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'], help="Customer's age range"),
        'GENDER': st.sidebar.selectbox("Gender", ['male', 'female'], help="Customer's gender"),
        'RACE': st.sidebar.selectbox("Race", ['majority', 'minority'], help="Ethnic group identification"),
        'DRIVING_EXPERIENCE': st.sidebar.selectbox("Driving Experience", ['0-9y', '10-19y', '20-29y', '30y+'], help="Years of driving experience"),
        'EDUCATION': st.sidebar.selectbox("Education Level", ['none', 'high school', 'university'], help="e.g., university degree"),
        'INCOME': st.sidebar.selectbox("Income Bracket", ['poverty', 'working class', 'middle class', 'upper class'], help="Customer's income category"),
        'CREDIT_SCORE': st.sidebar.number_input("Credit Score", min_value=0.0, help="e.g., 600.0"),
        'VEHICLE_OWNERSHIP': st.sidebar.checkbox("Owns Vehicle", help="Check if owns a vehicle"),
        'VEHICLE_YEAR': st.sidebar.selectbox("Vehicle Year", ['before 2015', 'after 2015'], help="Year category of the car"),
        'MARRIED': st.sidebar.checkbox("Married", help="Check if married"),
        'CHILDREN': st.sidebar.checkbox("Has Children", help="Check if has children"),
        'ANNUAL_MILEAGE': st.sidebar.number_input("Annual Mileage (in miles)", min_value=0.0, help="e.g., 12000"),
        'VEHICLE_TYPE': st.sidebar.selectbox("Vehicle Type", ['sedan', 'sports car'], help="Type of vehicle owned"),
        'SPEEDING_VIOLATIONS': st.sidebar.number_input("Speeding Violations", min_value=0, help="Total number of speeding violations"),
        'DUIS': st.sidebar.number_input("DUIs", min_value=0, help="Total DUI offenses"),
        'PAST_ACCIDENTS': st.sidebar.number_input("Past Accidents", min_value=0, help="Total past accidents")
    }
    return pd.DataFrame([data])

input_df = user_input()

# === Prediction Output ===
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**🧮 Probability of Purchasing Insurance:** `{probability:.2%}`")

    if prediction == 1:
        st.success("🟢 This customer is likely to purchase car insurance.")
        st.markdown("💡 _This customer shows strong interest in car insurance. Consider personalized offers._")
    else:
        st.warning("🟠 This customer is unlikely to purchase car insurance.")
        st.markdown("💡 _This customer may require a different marketing approach._")

    # Progress bar
    st.progress(int(probability * 100))


    # Gauge chart
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

    # Feature importances
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
        st.write(input_df)
        
# === Footer ===
st.markdown("---")
st.markdown("— Made by Şeyma Gülşen Akkuş", unsafe_allow_html=True)