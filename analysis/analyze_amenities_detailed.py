import pandas as pd
import numpy as np
import os
import ast
import re
from collections import Counter

# Copied from src/features.py to ensure consistency
def parse_amenities_list(x):
    """Parse amenities into list."""
    if pd.isna(x):
        return []
    s = str(x)

    # try python literal (list-like)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, set, tuple)):
            return [str(v).strip().lower() for v in obj]
    except Exception:
        pass

    # fallback: split
    s = s.strip("{}[]")
    return [p.strip().lower() for p in s.split(",") if p.strip()]

def clean_price(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
    return float(x)

def analyze():
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
            
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # Clean price
    df['price_num'] = df['price'].apply(clean_price)
    # Remove extreme outliers for valid comparison (e.g. > $10,000)
    df = df[df['price_num'] < 10000]
    
    print("Parsing amenities...")
    df['amenities_list'] = df['amenities'].apply(parse_amenities_list)
    
    # Flatten list to get counts
    all_amenities = [item for sublist in df['amenities_list'] for item in sublist]
    counter = Counter(all_amenities)
    
    # Get Top 80 candidates by frequency
    top_candidates = [x[0] for x in counter.most_common(80)]
    
    print("\n--- Calculating Price Impact (Top 80 Frequent Amenities) ---")
    print(f"{'Amenity':<40} {'Count':<8} {'Avg Price (With)':<16} {'Avg Price (w/o)':<16} {'Diff ($)':<10}")
    print("-" * 100)
    
    results = []
    
    for amenity in top_candidates:
        # Create temporary boolean mask
        has_amenity = df['amenities_list'].apply(lambda x: amenity in x)
        
        price_with = df.loc[has_amenity, 'price_num'].mean()
        price_without = df.loc[~has_amenity, 'price_num'].mean()
        count = has_amenity.sum()
        
        diff = price_with - price_without
        
        results.append({
            "amenity": amenity,
            "count": count,
            "with": price_with,
            "without": price_without,
            "diff": diff
        })
        
    # Sort by Price Difference (Value Boosters)
    results.sort(key=lambda x: x['diff'], reverse=True)
    
    for r in results:
        print(f"{r['amenity']:<40} {r['count']:<8} {r['with']:<16.2f} {r['without']:<16.2f} {r['diff']:<+10.2f}")
    
    # Also check keywords
    keywords = ["view", "jacuzzi", "pool", "gym", "sauna", "security", "doorman"]
    print("\n--- Keyword Impact ---")
    for kw in keywords:
        mask = df['amenities_list'].apply(lambda lst: any(kw in item for item in lst))
        p_with = df.loc[mask, 'price_num'].mean()
        p_without = df.loc[~mask, 'price_num'].mean()
        count = mask.sum()
        print(f"Contains '{kw}': Count={count}, Price With=${p_with:.0f}, Without=${p_without:.0f}, Diff=${p_with-p_without:.0f}")

if __name__ == "__main__":
    analyze()
