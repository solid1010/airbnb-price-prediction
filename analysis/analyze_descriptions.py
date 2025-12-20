import pandas as pd
import numpy as np
import os
import re
from collections import Counter

def clean_price(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
    return float(x)

def tokenize(text):
    if pd.isna(text): return []
    # Lowercase and remove non-alphanumeric
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split
    tokens = text.split()
    return tokens

def analyze_descriptions():
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
            
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # Clean price
    df['price_num'] = df['price'].apply(clean_price)
    df = df.dropna(subset=['price_num'])
    # Remove extreme outliers for steady analysis
    df = df[df['price_num'] < 10000]
    
    print("Tokenizing descriptions (this may take a moment)...")
    # Subsample if too large, but 24k is fine
    df['tokens'] = df['description'].fillna("").apply(tokenize)
    
    # Flatten
    all_tokens = [t for sublist in df['tokens'] for t in sublist]
    
    # Stopwords (Standard + Extended for Description filler)
    stopwords = {
        'in', 'at', 'the', 'with', 'and', 'a', 'of', 'for', 'to', 'from', 'on', 'by',
        'is', 'it', 'center', 'central', 'location', 'heart', 'near',
        'apartment', 'flat', 'room', 'home', 'house', 'place', 'stay',
        'istanbul', 'turkey', 'taksim', 'beyoglu', 'galata', # Locations checked elsewhere
        '1', '2', '3', '4', 'bedroom', 'bedrooms', 'bath', 'baths',
        'min', 'minutes', 'walk', 'walking', 'distance', 'away',
        'very', 'are', 'you', 'your', 'we', 'our', 'will', 'have', 'has', 'be', 'as',
        'this', 'that', 'there', 'all', 'can', 'just', 's', 't', 'm', 'd', 're', 've', 'll',
        'guests', 'guest', 'equipped', 'fully', 'kitchen', 'living', 'area', 'space',
        'located', 'building', 'floor', 'street', 'city', 'neighborhood', 'district',
        'enjoy', 'welcome', 'perfect', 'great', 'nice', 'comfortable', 'cozy', 'lovely', # Generic adjectives
        'check', 'time', 'available', 'also', 'plenty', 'many', 'lots', 'close', 'easy', 'access'
    }
    
    filtered_tokens = [t for t in all_tokens if t not in stopwords and len(t) > 3]
    counter = Counter(filtered_tokens)
    
    # Use top 200 most frequent tokens for analysis
    common_tokens = [x[0] for x in counter.most_common(200)]
    
    print("\n--- Calculating Price Impact of Description Keywords ---")
    print(f"{'Keyword':<20} {'Count':<8} {'Avg Price (With)':<18} {'Avg Price (w/o)':<18} {'Diff ($)':<10}")
    print("-" * 80)
    
    results = []
    
    for token in common_tokens:
        mask = df['tokens'].apply(lambda x: token in x)
        
        price_with = df.loc[mask, 'price_num'].mean()
        price_without = df.loc[~mask, 'price_num'].mean()
        count = mask.sum()
        
        diff = price_with - price_without
        
        results.append({
            "token": token,
            "count": count,
            "with": price_with,
            "without": price_without,
            "diff": diff
        })
        
    # Sort by Diff (Positive Impact)
    results.sort(key=lambda x: x['diff'], reverse=True)
    
    print("TOP 20 POSITIVE Keywords:")
    for r in results[:20]:
        print(f"{r['token']:<20} {r['count']:<8} {r['with']:<18.2f} {r['without']:<18.2f} {r['diff']:<+10.2f}")

    print("\nTOP 20 NEGATIVE Keywords:")
    for r in results[-20:]:
        print(f"{r['token']:<20} {r['count']:<8} {r['with']:<18.2f} {r['without']:<18.2f} {r['diff']:<+10.2f}")

if __name__ == "__main__":
    analyze_descriptions()
