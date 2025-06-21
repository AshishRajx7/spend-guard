import pandas as pd
import random
from datetime import datetime, timedelta

# Sample categories and merchants
CATEGORIES = ['Food', 'Travel', 'Shopping', 'Groceries', 'Entertainment', 'Utilities']
MERCHANTS = ['Zomato', 'Amazon', 'Flipkart', 'Swiggy', 'Uber', 'Myntra', 'BigBasket', 'Netflix', 'Dominos', 'Recharge']

def simulate_transactions(num_users=10, num_transactions=500):
    transactions = []

    for txn_id in range(1, num_transactions + 1):
        user_id = random.randint(1, num_users)
        merchant = random.choice(MERCHANTS)
        amount = round(random.uniform(50, 5000), 2)
        category = random.choice(CATEGORIES)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 60))  # Past 2 months
        transactions.append([txn_id, user_id, merchant, amount, category, timestamp])

    df = pd.DataFrame(transactions, columns=['TransactionID', 'UserID', 'MerchantName', 'Amount', 'Category', 'Timestamp'])
    df.to_csv('data/transactions.csv', index=False)
    print("User transaction data generated!")

def simulate_merchants():
    profiles = []

    for merchant in MERCHANTS:
        fraud_reports = random.randint(0, 50)
        refund_rate = round(random.uniform(0, 0.3), 2)  # 0% to 30% refunds
        avg_rating = round(random.uniform(2.0, 5.0), 1)
        profiles.append([merchant, fraud_reports, refund_rate, avg_rating])

    df = pd.DataFrame(profiles, columns=['MerchantName', 'FraudReports', 'RefundRate', 'AvgUserRating'])
    df.to_csv('data/merchants.csv', index=False)
    print("Merchant profile data generated!")


if __name__ == "__main__":
    simulate_transactions()
    simulate_merchants()
