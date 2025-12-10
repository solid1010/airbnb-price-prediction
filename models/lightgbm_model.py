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

def feature_engineering_extra(df):
    """
    Extracts additional features from raw dataframe:
    - Host Metrics: Response Rate/Time, Acceptance Rate
    - Host Tenure: host_since
    - Review Scores
    - Flags: instant_bookable
    """
    df_feat = pd.DataFrame()
    df_feat['id'] = df['id'] # Keep ID for merging
    
    # 1. Host Metrics
    # Host Metrics Binning

    # Host Response Rate
    def clean_rate(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            return float(x.replace('%', '')) 
        return float(x) # Return raw percentage 0-100

    # Binning function
    def bin_rate(val):
        if pd.isna(val): return 0 # Unknown
        if val == 100: return 4   # Perfect
        if val >= 90:  return 3   # High
        if val >= 50:  return 2   # Medium
        return 1                  # Low

    # Standardize to 0-100 scale then bin
    df['host_response_rate_clean'] = df['host_response_rate'].apply(clean_rate)
    df['host_acceptance_rate_clean'] = df['host_acceptance_rate'].apply(clean_rate)
    
    df_feat['host_response_rate_bin'] = df['host_response_rate_clean'].apply(bin_rate)
    df_feat['host_acceptance_rate_bin'] = df['host_acceptance_rate_clean'].apply(bin_rate)

    # Host Response Time
    resp_map = {
        'within an hour': 1,
        'within a few hours': 2,
        'within a day': 3,
        'a few days or more': 4
    }
    df_feat['host_response_time_ord'] = df['host_response_time'].map(resp_map).fillna(2) # Default to 'within a few hours'

    # 2. Host Tenure
    ref_date = pd.to_datetime("2025-01-01")
    df_feat['host_tenure_days'] = (ref_date - pd.to_datetime(df['host_since'], errors='coerce')).dt.days
    df_feat['host_tenure_days'] = df_feat['host_tenure_days'].fillna(0)

    # 3. Flags
    df_feat['instant_bookable_flag'] = df['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)

    # 4. Review Scores
    review_cols = [
        'review_scores_rating', 
        'review_scores_accuracy', 
        'review_scores_cleanliness', 
        'review_scores_checkin', 
        'review_scores_communication', 
        'review_scores_location', 
        'review_scores_value'
    ]
    
    for col in review_cols:
        if col in df.columns:
            median_val = df[col].median()
            df_feat[col] = df[col].fillna(median_val)
    
    return df_feat

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

    # Helper to get IDs from processed files for merging
    try:
        df_train_proc = joblib.load(os.path.join(data_dir, "processed_train.joblib"))
        df_test_proc = joblib.load(os.path.join(data_dir, "processed_test.joblib"))
        train_ids = df_train_proc['id']
        test_ids = df_test_proc['id']
    except Exception as e:
        print(f"Error loading IDs: {e}")
        return

    X_team, y, X_test_team, train_raw, test_raw = load_data(data_dir)

    # 2. Align Data
    print("Aligning and filtering data...")
    # NOTE: We align everything to the processed joblib files using ID.
    
    # Subset train_raw and test_raw using IDs to ensure perfect match
    if 'id' in train_raw.columns:
        train_raw = train_raw.set_index('id')
        train_raw_subset = train_raw.loc[train_ids].reset_index()
    else:
        print("Warning: id not found in train.csv, using default order (risky)")
        train_raw_subset = train_raw

    if 'id' in test_raw.columns:
        test_raw = test_raw.set_index('id')
        test_raw_subset = test_raw.loc[test_ids].reset_index()
    else:
        test_raw_subset = test_raw

    # 3. Feature Engineering (Geospatial & Extra)
    print("Generating geospatial features...")
    X_geo_train = get_geo_features(train_raw_subset)
    X_geo_test = get_geo_features(test_raw_subset)

    print("Generating extra features (Host Metrics [Binned], Review Scores)...")
    X_extra_train = feature_engineering_extra(train_raw_subset)
    X_extra_test = feature_engineering_extra(test_raw_subset)

    # Drop ID from extra features before concat
    if 'id' in X_extra_train.columns:
        X_extra_train = X_extra_train.drop(columns=['id'])
    if 'id' in X_extra_test.columns:
        X_extra_test = X_extra_test.drop(columns=['id'])

    # 4. Merge & Clean
    print("Merging features...")
    # Reset indices to ensure smooth concat
    X_final = pd.concat([
        X_team.reset_index(drop=True), 
        X_geo_train.reset_index(drop=True),
        X_extra_train.reset_index(drop=True)
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test_team.reset_index(drop=True), 
        X_geo_test.reset_index(drop=True),
        X_extra_test.reset_index(drop=True)
    ], axis=1)

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

    # 5. K-Fold Cross Validation
    print("Starting 5-Fold Cross Validation...")
    from sklearn.model_selection import KFold
    
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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store OOF predictions and Test predictions
    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(X_test_final))
    
    fold_scores = []
    
    X_final_np = X_final.to_numpy()
    y_np = y.to_numpy()
    X_test_np = X_test_final.to_numpy()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_final_np, y_np)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_val = X_final_np[train_idx], X_final_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict Valid
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(score)
        print(f"    RMSE: {score:.5f}")
        
        # Predict Test
        test_preds_accum += model.predict(X_test_np)

    # Average CV Score
    avg_rmse = np.mean(fold_scores)
    print(f"\nAverage CV RMSE: {avg_rmse:.5f}")
    
    # Average Test Predictions
    test_preds_avg_log = test_preds_accum / 5
    
    # 6. Save Model (Saving the last fold model loosely, or we could retrain on all)
    # Ideally for production we retrain on all, but for ensemble we usually keep folds.
    # For now, let's retrain on FULL data one last time for the 'final_model.pkl' artifact, 
    # OR better, save the average prediction which is robust.
    # Let's stick to the strategy: 5-Fold Avg is for submission. 
    # To provide a single artifact, we can retrain on full data using the robust features.
    
    print("Retraining on full dataset for final artifact...")
    full_model = lgb.LGBMRegressor(**best_params)
    full_model.fit(X_final, y)
    
    output_dir = "models" if os.path.exists("models") else "../models"
    submissions_dir = "submissions" if os.path.exists("submissions") else "../submissions"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    print("Saving full model...")
    joblib.dump(full_model, os.path.join(output_dir, "lightgbm_final_model.pkl"))
    
    # 7. Prediction & Submission
    print("Generating submission from CV Average...")
    final_preds = np.expm1(test_preds_avg_log) # Inverse log transform of AVGERAGE predictions
    final_preds = np.maximum(final_preds, 0) # Clip negative predictions

    # test_raw might have 'id' as index now
    test_ids_lookup = test_raw.index if test_raw.index.name == 'id' else test_raw['id']

    submission = pd.DataFrame({
        "ID": test_ids_lookup,
        "TARGET": final_preds
    })
    
    # Determine save path
    submission_path = os.path.join(submissions_dir, "submission_lgbm_cv_binned.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
