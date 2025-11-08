import os
import pandas as pd

def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'retail_sales.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def aggregate_daily_sales(df):
    df = df.copy()
    df = df.groupby('date', as_index=False)['units_sold'].sum()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def create_lag_features(df, lags=[1,7,14,30]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['units_sold'].shift(lag)
    df['rolling_7'] = df['units_sold'].rolling(7).mean()
    df = df.dropna()
    return df
