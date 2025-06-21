import streamlit as st
import numpy as np
import pandas as pd
import joblib
from src.spend_analyzer import load_transactions, calculate_category_spending, calculate_monthly_spending, detect_spending_spike
import shap

# Load data
transactions = load_transactions()

# Fix timestamp issue
transactions['Timestamp'] = pd.to_datetime(transactions['Timestamp'], errors='coerce')
transactions = transactions.dropna(subset=['Timestamp'])

# Load merchant risk data
merchants = pd.read_csv('data/merchants_with_ml_risk.csv')

# Load ML model
model = joblib.load('model/merchant_risk_model.pkl')

st.title("Spend Guard: Smart Spending & Merchant Risk Analyzer")

# Sidebar user selection
user_id = st.sidebar.selectbox("Select User ID", sorted(transactions['UserID'].unique()))

st.header(f"Spending Summary for User {user_id}")

# Category Spending
category_spending = calculate_category_spending(transactions, user_id)
st.subheader("Spending by Category")
st.bar_chart(category_spending.set_index('Category'))

# Monthly Spending
monthly_spending = calculate_monthly_spending(transactions, user_id)
st.subheader("Monthly Spending Trend")
st.line_chart(monthly_spending.set_index('Month'))

# Spending Spikes
spikes = detect_spending_spike(transactions, user_id)
if not spikes.empty:
    st.warning(f"Spending Spike Detected in: {', '.join([str(month) for month in spikes.index])}")
else:
    st.success("No unusual spending spikes detected!")

# Merchant ML Risk Table
st.subheader("Merchant ML Risk Profile (ML-Predicted)")
st.dataframe(merchants[['MerchantName', 'ML_RiskLevel']].sort_values(by='ML_RiskLevel', ascending=False))

# Real-Time Merchant Simulation
st.subheader("Simulate a New Transaction with ML Prediction")

merchant_choice = st.selectbox("Select Merchant", merchants['MerchantName'])
selected_merchant = merchants[merchants['MerchantName'] == merchant_choice]

merchant_features = selected_merchant[['FraudReports', 'RefundRate', 'AvgUserRating']]
predicted_risk = model.predict(merchant_features)[0]

# Convert prediction to risk label
risk_label = 'High' if predicted_risk == 2 else ('Medium' if predicted_risk == 1 else 'Low')

if risk_label == 'High':
    st.error(f"ML Alert: {merchant_choice} is classified as HIGH-RISK!")
elif risk_label == 'Medium':
    st.warning(f"ML Alert: {merchant_choice} is classified as MEDIUM-RISK!")
else:
    st.success(f"ML Alert: {merchant_choice} is classified as LOW-RISK.")

# SHAP Explainability
st.subheader("SHAP Explainability for Selected Merchant")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(merchant_features)

sample_index = 0  # Only one row input

# Handle multiclass SHAP safely
if isinstance(shap_values, list):
    shap_values_for_prediction = shap_values[predicted_risk][sample_index]
else:
    shap_values_for_prediction = shap_values[sample_index]

st.write("Feature Contributions:")
for feature, value, shap_val in zip(merchant_features.columns, merchant_features.values[0], shap_values_for_prediction):
    # Ensure shap_val is scalar
    if isinstance(shap_val, (np.ndarray, list)):
        shap_val = np.array(shap_val).flatten()[0]  # Force flatten and pick first element safely
    st.write(f"{feature}: Value = {value}, SHAP Contribution = {shap_val:.2f}")


# -------------------------------------
# Real-Time Transaction Entry with Log
# -------------------------------------
st.subheader("Simulate User Transaction Entry")

# Initialize session state to store simulated transactions
if 'simulated_transactions' not in st.session_state:
    st.session_state.simulated_transactions = []

# Select user and merchant
sim_user_id = st.selectbox("Select User for Transaction", sorted(transactions['UserID'].unique()), key='user_sim')
new_merchant = st.selectbox("Select Merchant", merchants['MerchantName'], key='merchant_sim')
new_amount = st.number_input("Enter Transaction Amount", min_value=1.0, step=1.0, key='amount_sim')

if st.button("Process Transaction"):
    selected_merchant = merchants[merchants['MerchantName'] == new_merchant]
    merchant_features = selected_merchant[['FraudReports', 'RefundRate', 'AvgUserRating']]
    predicted_risk = model.predict(merchant_features)[0]

    # Risk label conversion
    risk_label = 'High' if predicted_risk == 2 else ('Medium' if predicted_risk == 1 else 'Low')

    # Show alert based on risk
    if risk_label == 'High':
        st.error(f"ALERT: Transaction BLOCKED! {new_merchant} is HIGH-RISK.")
    elif risk_label == 'Medium':
        st.warning(f"ALERT: Caution! {new_merchant} is MEDIUM-RISK. Please verify.")
    else:
        st.success(f"Transaction APPROVED. {new_merchant} is LOW-RISK.")

    # Save transaction in session log
    st.session_state.simulated_transactions.append({
        'UserID': sim_user_id,
        'Merchant': new_merchant,
        'Amount': new_amount,
        'RiskLevel': risk_label
    })

    # SHAP Explanation
    st.subheader("SHAP Explanation for This Transaction")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(merchant_features)

    sample_index = 0  # Only one row input
    if isinstance(shap_values, list):
        shap_values_for_prediction = shap_values[predicted_risk][sample_index]
    else:
        shap_values_for_prediction = shap_values[sample_index]

    st.write("Feature Contributions:")
    for feature, value, shap_val in zip(merchant_features.columns, merchant_features.values[0], shap_values_for_prediction):
        if isinstance(shap_val, (np.ndarray, list)):
            shap_val = np.array(shap_val).flatten()[0]
        st.write(f"{feature}: Value = {value}, SHAP Contribution = {shap_val:.2f}")

# Display transaction log
if st.session_state.simulated_transactions:
    st.subheader("Simulated Transaction Log")
    st.dataframe(pd.DataFrame(st.session_state.simulated_transactions))
