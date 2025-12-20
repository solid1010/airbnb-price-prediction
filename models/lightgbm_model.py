import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import sys
import joblib

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features

def main():
    # 1. Load Raw Data
    print("Loading raw data...")
    data_dir = "data"
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
            data_dir = "../data"
    
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # 2. Feature Engineering Pipeline (from src/features.py)
    print("Applying unified feature pipeline...")
    
    # Process Train
    print("  Processing Train...")
    train = features.add_features(train)
    train = features.impute_missing_advanced(train)
    train, _ = features.add_room_type_dummies(train)
    train = features.clean_col_names(train)
    
    # Process Test
    print("  Processing Test...")
    test = features.add_features(test)
    test = features.impute_missing_advanced(test) # Note: Independent imputation (simplification)
    test, _ = features.add_room_type_dummies(test)
    test = features.clean_col_names(test)
    
    # Categorical handling
    cat_cols = ['neighbourhood_cleansed', 'property_type']
    for c in cat_cols:
        if c in train.columns:
            train[c] = train[c].astype('category')
        if c in test.columns:
            test[c] = test[c].astype('category')
    
    # 3. Feature Selection
    print("Selecting features...")
    # Get all numeric columns that are model ready
    feats = features.get_feature_columns(train)
    
    # Add categoricals
    feats += [c for c in cat_cols if c in train.columns]
    
    # Ensure all features exist in test
    feats = [f for f in feats if f in test.columns]
    
    print(f"Selected {len(feats)} features.")
    
    # Prepare X, y
    X = train[feats]
    
    # Target handling
    # src/features.py has clean_price but we need to ensure target is ready
    if 'price' in train.columns:
        # Drop missing targets if any
        train = train.dropna(subset=['price_num']).reset_index(drop=True)
        # Update X to match dropped rows
        X = train[feats]
        
        y = np.log1p(train['price_num'])
    else:
         raise ValueError("Price column missing or not processed")

    X_test = test[feats]
    
    # 4. Scaling (Robust Scaler) - Optional for Trees but good practice
    print("Scaling features...")
    X, scaler = features.scale_features_robust(X)
    X_test, _ = features.scale_features_robust(X_test, scaler)

    # 5. Model Training (5-Fold CV)
    print("Starting 5-Fold Cross Validation...")
    
    best_params = {
        'n_estimators': 3000, # Increased slightly
        'learning_rate': 0.02,
        'num_leaves': 64,
        'max_depth': 10,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(X_test))
    fold_scores = []
    
    # Use DataFrame directly to support categorical features
    # y is a Series, conversion to numpy is fine or keep as series
    y_np = y.to_numpy()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_np)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(score)
        print(f"    RMSE: {score:.5f}")
        
        test_preds_accum += model.predict(X_test)

    avg_rmse = np.mean(fold_scores)
    print(f"\nAverage CV RMSE: {avg_rmse:.5f}")
    
    # 6. Submission
    print("Generating submission...")
    final_preds = np.expm1(test_preds_accum / 5)
    final_preds = np.maximum(final_preds, 0)
    
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Handle IDs
    if 'id' in test.columns:
        test_ids = test['id']
    else:
        # Fallback if id got dropped (unlikely with this pipeline)
        # Re-read raw
        test_raw_for_id = pd.read_csv(os.path.join(data_dir, "test.csv"))
        test_ids = test_raw_for_id['id']

    submission = pd.DataFrame({
        "ID": test_ids,
        "TARGET": final_preds
    })
    
    sub_path = os.path.join(submissions_dir, "submission_refactored_pipeline.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved to {sub_path}")
    
    # Save Model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "lightgbm_refactored_last_fold.pkl"))

if __name__ == "__main__":
    main()
