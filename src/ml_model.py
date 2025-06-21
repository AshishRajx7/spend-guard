import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.risk_engine import load_merchants, prepare_merchant_features

def label_risk(row):
    if row['FraudReports'] > 40 or row['RefundRate'] > 0.30:
        return 2  # High Risk
    elif row['FraudReports'] > 20 or row['RefundRate'] > 0.15:
        return 1  # Medium Risk
    else:
        return 0  # Low Risk

def train_risk_model(df, labels):
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'model/merchant_risk_model.pkl')
    print("\n Model saved successfully!")

    return model

def assign_ml_risk_labels(model, feature_df, original_df):
    predictions = model.predict(feature_df)
    original_df['ML_Prediction'] = predictions
    original_df['ML_RiskLevel'] = original_df['ML_Prediction'].apply(lambda x: 'High' if x == 2 else ('Medium' if x == 1 else 'Low'))
    return original_df

if __name__ == "__main__":
    # Load merchant data
    merchant_df = load_merchants()

    # Multiclass labels: 0 = Low, 1 = Medium, 2 = High
    y = merchant_df.apply(label_risk, axis=1)

    # Prepare features
    X = prepare_merchant_features(merchant_df)

    # Train model
    model = train_risk_model(X[['FraudReports', 'RefundRate', 'AvgUserRating']], y)

    # Assign ML-based labels
    ml_scored_merchants = assign_ml_risk_labels(model, X[['FraudReports', 'RefundRate', 'AvgUserRating']], merchant_df)

    # Save updated merchant file
    ml_scored_merchants.to_csv('data/merchants_with_ml_risk.csv', index=False)
    print("\n ML-labeled merchant file saved as merchants_with_ml_risk.csv")
