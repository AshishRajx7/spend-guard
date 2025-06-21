import pandas as pd

def load_transactions():
    return pd.read_csv('data/transactions.csv', parse_dates=['Timestamp'])

def calculate_category_spending(df, user_id):
    user_df = df[df['UserID'] == user_id]
    category_summary = user_df.groupby('Category')['Amount'].sum().reset_index()
    return category_summary

def calculate_monthly_spending(df, user_id):
    user_df = df[df['UserID'] == user_id].copy()
    user_df['Month'] = pd.to_datetime(user_df['Timestamp']).dt.to_period('M')
    monthly_summary = user_df.groupby('Month')['Amount'].sum().reset_index()
    return monthly_summary

def detect_spending_spike(df, user_id, threshold=1.5):
    user_df = df[df['UserID'] == user_id].copy()
    user_df['Month'] = pd.to_datetime(user_df['Timestamp']).dt.to_period('M')
    monthly_spend = user_df.groupby('Month')['Amount'].sum()

    mean_spend = monthly_spend.mean()
    spikes = monthly_spend[monthly_spend > threshold * mean_spend]

    return spikes

if __name__ == "__main__":
    df = load_transactions()
    print(calculate_category_spending(df, user_id=1))
    print(calculate_monthly_spending(df, user_id=1))
    print(detect_spending_spike(df, user_id=1))
