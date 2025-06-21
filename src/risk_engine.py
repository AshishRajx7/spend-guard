
import pandas as pd

def load_merchants():
    return pd.read_csv('data/merchants.csv')

def prepare_merchant_features(df):
    # Prepare features for ML
    return df[['MerchantName', 'FraudReports', 'RefundRate', 'AvgUserRating']]

if __name__ == "__main__":
    df = load_merchants()
    features = prepare_merchant_features(df)
    print("\n📊 Merchant Features Prepared:")
    print(features.head())