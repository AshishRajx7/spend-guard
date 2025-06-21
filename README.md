# Spend Guard: Smart Spending & Merchant Risk Analyzer

Spend Guard is a smart financial dashboard that helps users track their spending patterns, detect unusual spikes, and assess merchant risk in real-time using machine learning.

##  Key Features
- Smart spending summaries by category and timeline
- Automatic spending spike detection
- Real-time merchant risk profiling using ML
- SHAP explainability for model transparency
- Real-time transaction simulation with dynamic risk alerts
- Simulated transaction log for user activity tracking

##  Technologies Used
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- SHAP for Explainable AI
- Joblib for model serialization

## Project Structure

```
spend-guard/
│
├── README.md               # Project description and instructions
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit main app
│
├── data/                   # Simulated datasets
│   ├── merchants.csv
│   ├── merchants_with_ml_risk.csv
│   └── transactions.csv
│
├── model/                  # Trained ML model
│   └── merchant_risk_model.pkl
│
├── src/                    # Python backend code
│   ├── __init__.py
│   ├── spend_analyzer.py
│   ├── risk_engine.py
│   └── ml_model.py
│
└── simulation/             # Data simulation code
    └── data_simulation.py
```
##  Real-Time Transaction Simulation
The app allows users to:

- Select merchants
- Enter transaction amounts
- Get immediate risk classification
- Visualize SHAP explainability for each transaction
- View a live transaction log

  ##  Demo Screenshots
  ![image](https://github.com/user-attachments/assets/b50af946-f202-4dd3-8b92-ca6570bd78d9)



```
