import pandas as pd
import numpy as np
import os

def analyze_reviews():
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
            
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # Identify review columns
    review_cols = [c for c in df.columns if 'review' in c]
    print(f"\nFound {len(review_cols)} review-related columns:")
    print(review_cols)
    
    print("\n--- Analysis ---")
    stats = []
    for col in review_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            median = df[col].median()
            mean = df[col].mean()
            stats.append({
                "Column": col,
                "Missing": missing,
                "Missing (%)": f"{missing_pct:.2f}%",
                "Median": median,
                "Mean": f"{mean:.2f}"
            })
        else:
            print(f"Skipping non-numeric: {col}")
            
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    analyze_reviews()
