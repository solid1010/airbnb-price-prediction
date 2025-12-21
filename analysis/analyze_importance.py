import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def analyze_importance():
    model_path = "models/lightgbm_refactored_last_fold.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Get feature importance from the booster object
    booster = model.booster_
    importance_df = pd.DataFrame({
        'feature': booster.feature_name(),
        'importance_gain': booster.feature_importance(importance_type='gain'),
        'importance_split': booster.feature_importance(importance_type='split')
    })

    # Sort by Gain (usually more representative of predictive power)
    importance_df = importance_df.sort_values(by='importance_gain', ascending=False)

    print("\n--- TOP 30 FEATURES (by Gain) ---")
    print(importance_df.head(30))

    # Plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance_gain', y='feature', data=importance_df.head(40))
    plt.title('LightGBM Feature Importance (Gain) - Top 40')
    plt.tight_layout()
    plt.savefig('analysis/feature_importance_gain.png')
    print("\nPlot saved to analysis/feature_importance_gain.png")

    # Check specific features
    print("\n--- Specific Feature Category Check ---")
    categories = {
        "Target Encoding": "encoded",
        "K-Means": "geo_cluster",
        "NLP Title": "title_",
        "NLP Description": "desc_",
        "Amenities": "amenity_"
    }

    for cat, prefix in categories.items():
        sub_df = importance_df[importance_df['feature'].str.contains(prefix)]
        total_gain = sub_df['importance_gain'].sum()
        total_split = sub_df['importance_split'].sum()
        print(f"{cat:<20}: Count={len(sub_df):<4} | Total Gain={total_gain:<12.2f} | Total Split={total_split:<6}")

if __name__ == "__main__":
    analyze_importance()
