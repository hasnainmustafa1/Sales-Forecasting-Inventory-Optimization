import os
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from data_preprocessing import load_data, aggregate_daily_sales, create_lag_features


def evaluate():
    df_raw = load_data()
    df = aggregate_daily_sales(df_raw)
    df_feat = create_lag_features(df)
    X = df_feat.drop(columns=['units_sold'])
    y = df_feat['units_sold']
    test_size = 90
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rf_forecast_model.pkl'))
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    with open(os.path.join(results_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f'MAPE:{mape}\nRMSE:{rmse}\n')
    print('Evaluation report saved.')

if __name__ == '__main__':
    evaluate()
