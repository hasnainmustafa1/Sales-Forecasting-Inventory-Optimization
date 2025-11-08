import os
from flask import Flask, jsonify
import joblib
import pandas as pd
from data_preprocessing import aggregate_daily_sales, create_lag_features

app = Flask(__name__)
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, 'models', 'rf_forecast_model.pkl')
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)

@app.route('/health')
def health():
    return jsonify({'status':'ok'})

@app.route('/forecast')
def forecast():
    load_model()
    df_raw = pd.read_csv(os.path.join(BASE, 'data', 'retail_sales.csv'), parse_dates=['date'])
    df = aggregate_daily_sales(df_raw)
    df_feat = create_lag_features(df)
    last_row = df_feat.iloc[-1:]
    preds = []
    current = last_row.copy()
    for i in range(30):
        X = current.drop(columns=['units_sold'])
        pred = model.predict(X)[0]
        preds.append(float(pred))
        # update lags
        for lag in [30,14,7,1]:
            current[f'lag_{lag}'] = current[f'lag_{lag}'].shift(-1)
        current['lag_1'] = pred
    return jsonify({'forecast_30_days': preds})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
