import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

def inspect():
    df = pd.read_csv('data/train.csv')
    
    # Identify columns we probably haven't used deeply
    used_keywords = ['amenities', 'price', 'name', 'latitude', 'longitude', 
                     'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
                     'review_scores', 'availability', 'host_response', 'host_acceptance']
                     
    print(f"Total Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))
    
    print("\n--- Sample of potentially unused text/categorical columns ---")
    potential_cols = ['description', 'neighborhood_overview', 'host_about', 'host_verifications', 'property_type', 'licence']
    
    for c in potential_cols:
        if c in df.columns:
            print(f"\nCOLUMN: {c}")
            print(df[c].head(5))
            print(f"Null count: {df[c].isnull().sum()}")
            if df[c].dtype == 'object':
                print(f"Unique values (first 10): {df[c].dropna().unique()[:10]}")

if __name__ == "__main__":
    inspect()
