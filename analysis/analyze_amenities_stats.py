import pandas as pd
import numpy as np
import os
import ast
from collections import Counter
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold

def parse_amenities_list(x):
    if pd.isna(x): return []
    s = str(x)
    s = s.strip("{}[]")
    return [p.strip().lower() for p in s.split(",") if p.strip()]

def clean_price(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
    return float(x)

def analyze_stats():
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
            
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # Clean target
    df['price_num'] = df['price'].apply(clean_price)
    df = df.dropna(subset=['price_num'])
    df['log_price'] = np.log1p(df['price_num'])
    
    print("Parsing amenities...")
    df['amenities_list'] = df['amenities'].apply(parse_amenities_list)
    
    # Get all distinct amenities
    all_amenities = [item for sublist in df['amenities_list'] for item in sublist]
    counter = Counter(all_amenities)
    
    # Let's analyze top 150 most frequent ones to capture enough variance
    top_150 = [x[0] for x in counter.most_common(150)]
    
    print(f"Constructing binary matrix for {len(top_150)} amenities...")
    
    # Efficiently create binary dataframe
    amenity_data = {}
    for amenity in top_150:
        amenity_data[amenity] = df['amenities_list'].apply(lambda x: 1 if amenity in x else 0)
        
    X_amenities = pd.DataFrame(amenity_data)
    y = df['log_price']
    
    print("\n--- Statistical Analysis ---")
    
    # 1. Variance
    # Var(X) = p(1-p) for binary. Max is 0.25 (p=0.5).
    variances = X_amenities.var()
    
    # 2. Mutual Information
    # Measures dependency between Amenity and Price
    print("Calculating Mutual Information (this may take a moment)...")
    mi_scores = mutual_info_regression(X_amenities, y, discrete_features=True, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_amenities.columns)
    
    # Combine results
    stats = pd.DataFrame({
        "Variance": variances,
        "Mutual Info": mi_series,
        "Frequency": X_amenities.mean()
    })
    
    # Sort by Mutual Info (Predictive Power)
    stats = stats.sort_values(by="Mutual Info", ascending=False)
    
    print(f"\n{'Amenity':<40} {'Mut. Info':<10} {'Variance':<10} {'Freq (%)':<10}")
    print("-" * 80)
    
    for amenity, row in stats.head(40).iterrows():
        print(f"{amenity:<40} {row['Mutual Info']:.4f}     {row['Variance']:.4f}     {row['Frequency']*100:.1f}%")

    print("-" * 80)
    print("Interpretation:")
    print("- High Mutual Info: Best predictors of price.")
    print("- High Variance: Good spread (not too rare, not too common).")
    
if __name__ == "__main__":
    analyze_stats()
