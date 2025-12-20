import pandas as pd
import numpy as np
import os
import ast
from collections import Counter

def parse_amenities_list(x):
    if pd.isna(x): return []
    s = str(x)
    s = s.strip("{}[]")
    return [p.strip().lower() for p in s.split(",") if p.strip()]

def analyze_synonyms():
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
            
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    print("Parsing amenities...")
    df['amenities_list'] = df['amenities'].apply(parse_amenities_list)
    
    # Flatten list
    all_amenities = [item for sublist in df['amenities_list'] for item in sublist]
    counter = Counter(all_amenities)
    
    unique_amenities = len(counter)
    print(f"\nTotal Unique Amenity Strings: {unique_amenities}")
    
    # 1. Look for variations of common terms
    concepts = [
        "wifi", "tv", "air conditioning", "heating", 
        "parking", "kitchen", "coffee", "shampoo", "view",
        "sound", "baby", "children", "pet", "breakfast"
    ]
    
    print(f"\n{'Concept':<20} {'Exact Match Count':<20} {'Total Variations Count':<25} {'Examples of Variations'}")
    print("-" * 120)
    
    for concept in concepts:
        # Exact match
        exact_count = counter[concept]
        
        # Variations (contains concept)
        variations = {k: v for k, v in counter.items() if concept in k}
        total_var_count = sum(variations.values())
        
        # Top 3 variations (excluding the exact match itself)
        top_vars = sorted([(k, v) for k, v in variations.items() if k != concept], key=lambda x: x[1], reverse=True)[:3]
        examples = ", ".join([f"{k}({v})" for k, v in top_vars])
        
        print(f"{concept:<20} {exact_count:<20} {total_var_count:<25} {examples}")

    # 2. Look for completely unique but frequent items that we might missed
    # (Items that don't match our 'concepts' list but are frequent)
    print("\n\n--- Frequent Amenities NOT in Concept List ---")
    seen_in_concepts = set()
    for concept in concepts:
        for k in counter.keys():
            if concept in k:
                seen_in_concepts.add(k)
                
    remaining = {k: v for k, v in counter.items() if k not in seen_in_concepts}
    top_remaining = sorted(remaining.items(), key=lambda x: x[1], reverse=True)[:40]
    
    for k, v in top_remaining:
        print(f"{k}: {v}")

if __name__ == "__main__":
    analyze_synonyms()
