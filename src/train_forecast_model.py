import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from data_preprocessing import load_data, aggregate_daily_sales, create_lag_features


def train_models():
    df_raw = load_data()
    df = aggregate_daily_sales(df_raw)
    df_feat = create_lag_features(df)
    X = df_feat.drop(columns=['units_sold'])
    y = df_feat['units_sold']
    test_size = 90
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(models_dir, 'rf_forecast_model.pkl'))
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, preds, label='Predicted')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'forecast_plot.png'))
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f'MAPE:{mape}\nRMSE:{rmse}\n')
    print('Training complete. Model and results saved.')

if __name__ == '__main__':
    train_models()
