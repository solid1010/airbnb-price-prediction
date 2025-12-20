import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_price(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
    return float(x)

def diagnose():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    sub = pd.read_csv("submissions/submission_refactored_pipeline.csv")
    
    # Process Train Price
    train['price_num'] = train['price'].apply(clean_price)
    
    # Stats
    print("\n--- Statistics Comparison ---")
    print(f"{'Metric':<15} {'Train Actual':<15} {'Test Prediction':<15} {'Diff':<10}")
    print("-" * 60)
    
    metrics = {
        'Mean': (train['price_num'].mean(), sub['TARGET'].mean()),
        'Median': (train['price_num'].median(), sub['TARGET'].median()),
        'Min': (train['price_num'].min(), sub['TARGET'].min()),
        'Max': (train['price_num'].max(), sub['TARGET'].max()),
        'Std': (train['price_num'].std(), sub['TARGET'].std())
    }
    
    for m, (t, s) in metrics.items():
        print(f"{m:<15} {t:<15.2f} {s:<15.2f} {s-t:<10.2f}")
        
    print("\n--- Extreme Outlier Check ---")
    high_preds = sub[sub['TARGET'] > 10000]
    print(f"Predictions > 10,000 TL: {len(high_preds)} (Train has {len(train[train['price_num'] > 10000])})")
    print(high_preds.head())
    
    low_preds = sub[sub['TARGET'] < 100]
    print(f"Predictions < 100 TL: {len(low_preds)} (Train has {len(train[train['price_num'] < 100])})")

if __name__ == "__main__":
    diagnose()
