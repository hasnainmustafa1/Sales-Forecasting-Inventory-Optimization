import os
import pandas as pd

def compute_reorder(reorder_days=30, safety_days=7, lead_time_days=7):
    base = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(base, 'data', 'retail_sales.csv'), parse_dates=['date'])
    df = df.groupby('date')['units_sold'].sum().reset_index()
    df = df.set_index('date').resample('D').sum().fillna(0)
    avg_daily = df['units_sold'][-reorder_days:].mean()
    reorder_qty = int((avg_daily * reorder_days) + (avg_daily * safety_days))
    reorder_point = int(avg_daily * lead_time_days)
    report = pd.DataFrame({
        'average_daily_demand': [avg_daily],
        'reorder_quantity': [reorder_qty],
        'reorder_point': [reorder_point]
    })
    results_dir = os.path.join(base, 'results')
    os.makedirs(results_dir, exist_ok=True)
    report.to_csv(os.path.join(results_dir, 'inventory_report.csv'), index=False)
    print('Inventory report saved to results/inventory_report.csv')

if __name__ == '__main__':
    compute_reorder()
