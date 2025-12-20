import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze():
    print("Loading raw data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Check Price distribution in Train
    print("\n--- Price Stats (Train) ---")
    print(train['price'].describe())
    
    # Check Dates
    print("\n--- Date Distributions ---")
    for col in ['last_review', 'host_since']:
        if col in train.columns:
            train[col] = pd.to_datetime(train[col], errors='coerce')
            test[col] = pd.to_datetime(test[col], errors='coerce')
            
            print(f"\n{col}:")
            print(f"Train: min {train[col].min()}, max {train[col].max()}")
            print(f"Test : min {test[col].min()}, max {test[col].max()}")
            
            # Check overlap
            train_latest = train[col].max()
            test_earliest = test[col].min()
            
            if test_earliest > train_latest:
                print("!! STRICT TEMPORAL SPLIT DETECTED !!")
            else:
                print("Dates overlap.")

    # Check Key Categoricals/New Features
    print("\n--- New Feature Stats ---")
    common_host_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in common_host_cols:
        if col in train.columns:
            # Simple conversion for quick check
            t_val = train[col].astype(str).str.replace('%', '').astype(float)
            te_val = test[col].astype(str).str.replace('%', '').astype(float)
            
            print(f"\n{col} Mean:")
            print(f"Train: {t_val.mean():.4f}")
            print(f"Test : {te_val.mean():.4f}")
            
analyze()
