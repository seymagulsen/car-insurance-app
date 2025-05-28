# Car Insurance Claim Prediction

**ADS 542 - Statistical Learning Final Project**  
**Author:** Şeyma Gülşen Akkuş  
**App:** [car-insurance-app.streamlit.app](https://car-insurance-app.streamlit.app/)  

---

## Overview

This project aims to build a robust classification model to predict whether a car insurance policyholder will file a claim. The prediction is based on customer demographics, driving behavior, vehicle characteristics, and credit information.

The project is implemented in Python and includes:

- Data cleaning and preprocessing with imputation models  
- Feature engineering and transformation  
- Classification using multiple machine learning algorithms  
- Hyperparameter tuning  
- Full pipeline integration  
- Deployment using Streamlit  

---

## Dataset

- **Source:** Synthetic dataset (`Car_Insurance_Claim.csv`) with 10,000 entries and 19 columns  
- **Target Variable:** `OUTCOME` (1: Claim filed, 0: No claim)  
- **Missing Data:** `CREDIT_SCORE` (9.8%) and `ANNUAL_MILEAGE` (9.6%) were imputed using model-based Random Forest Regressors  

---

## Data Preprocessing

- **Missing Value Imputation:** Model-based imputation (RandomForest)  
- **Encoding:** Ordinal and One-Hot encoding  
- **Feature Scaling:** StandardScaler  
- **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)  

---

## Feature Engineering

- `RISK_SCORE`: Composite feature combining DUI, speeding violations, and past accidents  
- `HIGH_MILEAGE`: Binary feature indicating above-median annual mileage  

---

## Model Training & Selection

Four models were compared:

| Model               | F1 Score | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression | 0.74     | 0.91    |
| Random Forest       | 0.74     | 0.89    |
| MLP Neural Network  | 0.68     | 0.85    |
| XGBoost             | 0.74     | 0.89    |

**Final Model: Tuned XGBoost Classifier**  
After hyperparameter tuning, the final model achieved:
- **F1 Score:** 0.76  
- **ROC-AUC:** 0.90  

---

## Full Pipeline

The project includes a complete pipeline integrating:
- Column dropping  
- Model-based imputation  
- Feature engineering  
- Column transformations  
- SMOTE oversampling  
- XGBoost classification  

The entire pipeline is saved as a `.pkl` file and used in the deployed Streamlit app.

---

## Web Application

📍 **Live App**: [car-insurance-app.streamlit.app](https://car-insurance-app.streamlit.app/)  

### App Features
- Interactive form to input customer features  
- Real-time claim probability prediction  
- Visual result with confidence gauge  
- Bar chart of top contributing features  
- Ability to reset and start a new prediction  

---
## Key Libraries

- `pandas`, `numpy`, `scikit-learn`  
- `xgboost`, `imbalanced-learn`  
- `matplotlib`, `seaborn`, `plotly`  
- `streamlit`  

---

## Author

**Şeyma Gülşen Akkuş**  
Graduate Student, Applied Data Science  
TED University  
