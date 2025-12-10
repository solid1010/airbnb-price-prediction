import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(data_dir="../data"):
    """
    Loads processed data from joblib files.
    Constructs X, y from processed_train.joblib and X_test from processed_test.joblib.
    Also loads raw data for geospatial alignment.
    """
    print("Loading processed data...")
    try:
        # Load main processed files
        df_train = joblib.load(os.path.join(data_dir, "processed_train.joblib"))
        df_test = joblib.load(os.path.join(data_dir, "processed_test.joblib"))
        
        # Load raw data for geospatial calculations
        train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"))
        
        # Prepare X and y
        # Target variable
        if 'log_price' in df_train.columns:
            y = df_train['log_price']
        elif 'price' in df_train.columns:
            # If log_price missing but price present, create it
            # Ensure price is numeric first (it should be processed already)
            y = np.log1p(df_train['price'])
        else:
            raise ValueError("Target 'log_price' or 'price' not found in processed_train.joblib")
            
        # Features X
        # Drop target and non-feature columns
        cols_to_drop = ['price', 'log_price', 'id']
        X_team = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
        
        # Features X_test
        X_test_team = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
        
        return X_team, y, X_test_team, train_raw, test_raw
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure '01_EDA&FeatureEng.ipynb' has been run and data files are in '../data/'.")
        exit(1)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance (km) between two coordinates."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_geo_features(df):
    """Generates geospatial features based on distance to key Istanbul locations."""
    locations = {
        "Taksim": (41.0370, 28.9851),
        "Sultanahmet": (41.0054, 28.9768),
        "Besiktas": (41.0422, 29.0060),
        "Kadikoy": (40.9901, 29.0254),
        "Airport": (41.2811, 28.7533)
    }
    
    geo_data = pd.DataFrame()
    for loc, (lat, lon) in locations.items():
        geo_data[f"dist_{loc}"] = haversine_distance(df["latitude"], df["longitude"], lat, lon)
    
    # Distance to the nearest city center (excluding airport)
    centers = [c for c in geo_data.columns if "Airport" not in c]
    geo_data["min_dist_center"] = geo_data[centers].min(axis=1)
    return geo_data

def clean_col_names(df):
    """Cleans column names to be compatible with LightGBM."""
    new_cols = []
    seen_cols = {}
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', str(col))
        if new_col in seen_cols:
            seen_cols[new_col] += 1
            new_col = f"{new_col}_{seen_cols[new_col]}"
        else:
            seen_cols[new_col] = 1
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def main():
    # 1. Load Data
    data_dir = "data" # Run from project root
    # Check if we are in the correct directory, if not adjust path
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
        else:
            print("Error: 'data' directory not found.")
            return

    X_team, y, X_test_team, train_raw, test_raw = load_data(data_dir)

    # 2. Align Data (Filter outliers as done in EDA)
    print("Aligning and filtering data...")
    # NOTE: The filtering logic below mimics the notebook. 
    # Important: X_team and y are ALREADY processed/aligned in the previous steps (EDA notebook).
    # However, train_raw needs to be aligned to X_team if rows were dropped.
    # In the notebook logic, joblib files X and y are already aligned. 
    # train_raw is just used for 'latitude' and 'longitude'.
    # We must ensure train_raw matches X_team indices.
    
    # Re-applying the filter logic to train_raw to match dimensions if needed, 
    # OR assuming X_team was created FROM the filtered version.
    # The notebook code: train_filtered = train_filtered.iloc[:X_team.shape[0]] 
    # suggests a simple truncation was used if sizes didn't match, which is risky but we follow the notebook logic for fidelity.
    
    # Robust alignment strategy:
    # It takes raw train, cleans price, clips outliers.
    train_raw["price"] = train_raw["price"].replace({"\$": "", ",": ""}, regex=True).astype(float)
    train_raw = train_raw.dropna(subset=["price"])
    q_high = train_raw["price"].quantile(0.99)
    train_filtered = train_raw[train_raw["price"] <= q_high].reset_index(drop=True)
    
    if train_filtered.shape[0] != X_team.shape[0]:
        print(f"Warning: Raw filtered shape {train_filtered.shape} != Processed shape {X_team.shape}")
        print("Truncating filtered raw data to match processed data length (as per notebook logic).")
        train_filtered = train_filtered.iloc[:X_team.shape[0]]

    # 3. Feature Engineering (Geospatial)
    print("Generating geospatial features...")
    X_geo_train = get_geo_features(train_filtered)
    X_geo_test = get_geo_features(test_raw)

    # 4. Merge & Clean
    print("Merging features...")
    X_final = pd.concat([X_team.reset_index(drop=True), X_geo_train.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_team.reset_index(drop=True), X_geo_test.reset_index(drop=True)], axis=1)

    X_final = clean_col_names(X_final)
    X_test_final = clean_col_names(X_test_final)

    # Filter for numeric/bool types only (LightGBM requirement)
    print("Filtering non-numeric columns...")
    X_final = X_final.select_dtypes(include=['number', 'bool'])
    X_test_final = X_test_final.select_dtypes(include=['number', 'bool'])
    
    # Align columns between train and test (ensure test has same columns as train)
    # Get common columns
    common_cols = X_final.columns.intersection(X_test_final.columns)
    X_final = X_final[common_cols]
    X_test_final = X_test_final[common_cols]
    
    print(f"Final feature count: {X_final.shape[1]}")

    # 5. Model Training
    print("Training LightGBM model with best params...")
    
    # Parameters from Optuna study in notebook
    best_params = {
        'n_estimators': 2059,
        'learning_rate': 0.025343452969683825,
        'num_leaves': 68,
        'max_depth': 11,
        'colsample_bytree': 0.6625893951602744,
        'subsample': 0.6509179025430295,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1
    }

    # Split for validation reporting (optional but good practice)
    X_train, X_valid, y_train, y_valid = train_test_split(X_final, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=True)]
    )

    # Evaluation
    preds_log = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds_log))
    print(f"Validation RMSE: {rmse:.5f}")

    # Retrain on full data for submission (Optional, but often boost performance)
    print("Retraining on full dataset...")
    full_model = lgb.LGBMRegressor(**best_params)
    full_model.fit(X_final, y)

    # 6. Save Model & Artifacts
    output_dir = "models" if os.path.exists("models") else "../models"
    submissions_dir = "submissions" if os.path.exists("submissions") else "../submissions"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    print("Saving model and artifacts...")
    joblib.dump(full_model, os.path.join(output_dir, "lightgbm_final_model.pkl"))
    
    # 7. Prediction & Submission
    print("Generating submission...")
    final_preds_log = full_model.predict(X_test_final)
    final_preds = np.expm1(final_preds_log) # Inverse log transform
    final_preds = np.maximum(final_preds, 0) # Clip negative predictions

    submission = pd.DataFrame({
        "ID": test_raw["id"],
        "TARGET": final_preds
    })
    
    # Determine save path
    submission_path = os.path.join(submissions_dir, "submission_script_lgbm.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
