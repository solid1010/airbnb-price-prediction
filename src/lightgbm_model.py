import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import sys
import joblib
import optuna  


# Get the absolute path of this script (src/lightgbm_model.py)
current_file_path = os.path.abspath(__file__) 
# Get the directory containing this script (src)
src_dir = os.path.dirname(current_file_path)
# Get the project root (one level up from src)
PROJECT_ROOT = os.path.dirname(src_dir)

# Add project root to system path to import modules
sys.path.append(PROJECT_ROOT)

from src import features

def main():
    # ---------------------------------------------------------
    # 1. Path Configuration and Load Raw Data
    # ---------------------------------------------------------
    
    # Get the directory where this script is located (src/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to find the Project Root (YZV311_2526_7/)
    PROJECT_ROOT = os.path.dirname(current_script_dir)
    
    print(f"Project Root Detected: {PROJECT_ROOT}")
    
    # Define data directories relative to Project Root
    data_dir = os.path.join(PROJECT_ROOT, "data")
    submissions_dir = os.path.join(PROJECT_ROOT, "submissions")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    # Create directories if they don't exist
    os.makedirs(submissions_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading raw data...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING PIPELINE
    # ---------------------------------------------------------
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
    
    # K-Means Location Clustering (Stateful Feature Engineering)
    print("  Training K-Means on Train locations...")
    # Train only on training data to avoid leakage
    # Reduce clusters to 10 to avoid overfitting to specific logical blocks
    kmeans_model = features.train_kmeans_geo(train, n_clusters=10)
    
    print("  Applying K-Means to Train and Test...")
    train = features.add_kmeans_geo_features(train, kmeans_model)
    test = features.add_kmeans_geo_features(test, kmeans_model)
    
    # Categorical handling
    cat_cols = ['neighbourhood_cleansed', 'property_type']
    for c in cat_cols:
        if c in train.columns:
            train[c] = train[c].astype('category')
        if c in test.columns:
            test[c] = test[c].astype('category')
    
    # ---------------------------------------------------------
    # 3. FEATURE SELECTION
    # ---------------------------------------------------------
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
    if 'price' in train.columns:
        # Only drop rows where the target itself is NaN (missing)
        train = train.dropna(subset=['price_num']).reset_index(drop=True)
        
        # --- Apply Winsorization instead of dropping rows ---
        # This caps extreme values at the 99th percentile instead of deleting them.
        # It allows the model to learn from luxury listings without being confused by errors.
        print("  Outlier Handling: Applying Winsorization (Soft Clipping 99%)...")
        train = features.handle_outliers_winsorization(train, columns=['price_num'], limits=(0.00, 0.99))
        
        X = train[feats]
        y = np.log1p(train['price_num'])
    else:
         raise ValueError("Price column missing")

    X_test = test[feats]
    
    # ---------------------------------------------------------
    # 4. SCALING (Robust Scaler)
    # ---------------------------------------------------------
    print("Scaling features...")
    X, scaler = features.scale_features_robust(X)
    X_test, _ = features.scale_features_robust(X_test, scaler)

    # Convert y to numpy for speed
    y_np = y.to_numpy()

    # ---------------------------------------------------------
    # 5. OPTUNA HYPERPARAMETER OPTIMIZATION
    # ---------------------------------------------------------
    
    print("Starting Optuna Hyperparameter Optimization...")

    def objective(trial):
        # Define search space with regularization to prevent overfitting
        params = {
            'n_estimators': 2000, # Fixed high number, used with early stopping
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 20, 45), # Restricted to prevent overfitting
            'max_depth': trial.suggest_int('max_depth', 5, 9),    # Restricted depth
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'subsample_freq': 1,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0), # L1 Regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0), # L2 Regularization
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 3-Fold CV for speed inside Optuna (5-Fold takes too long)
        kf_opt = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        
        y_np_opt = y.to_numpy()
        
        for train_idx, val_idx in kf_opt.split(X, y_np_opt):
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y_np_opt[train_idx], y_np_opt[val_idx]
            
            # --- Target Encoding (Must be done inside loop) ---
            X_tr["temp_target"] = y_tr
            X_tr, mapping = features.target_encode(X_tr, by="neighbourhood_cleansed", target="temp_target")
            X_val["dummy"] = 0
            X_val, _ = features.target_encode(X_val, by="neighbourhood_cleansed", target="dummy", mapping=mapping)
            
            # Clean up temp cols
            X_tr = X_tr.drop(columns=["temp_target"])
            X_val = X_val.drop(columns=["dummy"])
            # --------------------------------------------------
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_scores.append(rmse)
        
        return np.mean(fold_scores)

    # Run Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # 20 trials is a good start
    
    print(f"Best params: {study.best_params}")
    
    # Merge best params with fixed params for final training
    best_params = study.best_params
    best_params.update({
        'n_estimators': 3000, # Increase for final training
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1
    })

    # ---------------------------------------------------------
    # 6. FINAL MODEL TRAINING (5-Fold CV) WITH BEST PARAMS
    # ---------------------------------------------------------
    print("\nStarting 5-Fold Cross Validation with STANDARD parameters...")
    
    # Balanced parameters: Not too slow, not too aggressive.
    # Aiming to beat the baseline score of 0.503.
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 4000,          
        'learning_rate': 0.02,         # Standard speed (0.003 was too slow, caused underfitting)
        'num_leaves': 40,              # Increased capacity (15 was too simple)
        'max_depth': 8,                # Standard depth
        'min_child_samples': 30,       # Allow learning from smaller groups
        'colsample_bytree': 0.7,       # Use 70% of features per tree
        'subsample': 0.7,              
        'reg_alpha': 1.0,              # Light L1 regularization
        'reg_lambda': 1.0,             # Light L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    test_preds_accum = np.zeros(len(X_test))
    fold_scores = []
    
    # Convert target to numpy for speed
    y_np = y.to_numpy()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_np)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        
        # --- Internal Target Encoding (To prevent Data Leakage) ---
        X_tr["temp_target_log"] = y_tr
        
        # Apply encoding to Neighborhood (Stronger smoothing m=50)
        X_tr, mapping_nb = features.target_encode(X_tr, by="neighbourhood_cleansed", target="temp_target_log", m=50.0)
        X_val["dummy"] = 0
        X_val, _ = features.target_encode(X_val, by="neighbourhood_cleansed", target="dummy", mapping=mapping_nb)
        
        # Apply encoding to Property Type (Stronger smoothing m=50)
        X_tr, mapping_pt = features.target_encode(X_tr, by="property_type", target="temp_target_log", m=50.0)
        X_val, _ = features.target_encode(X_val, by="property_type", target="dummy", mapping=mapping_pt)
        
        # Cleanup temporary columns
        X_tr = X_tr.drop(columns=["temp_target_log"])
        X_val = X_val.drop(columns=["dummy"])
        
        # Apply encoding to Test Set (Using the mapping from the current fold)
        X_test_fold = X_test.copy()
        X_test_fold["dummy"] = 0
        X_test_fold, _ = features.target_encode(X_test_fold, by="neighbourhood_cleansed", target="dummy", mapping=mapping_nb)
        X_test_fold, _ = features.target_encode(X_test_fold, by="property_type", target="dummy", mapping=mapping_pt)
        X_test_fold = X_test_fold.drop(columns=["dummy"])
        # ----------------------------------------------------------
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(300, verbose=False)] # Higher patience for slow learning rate
        )
        
        val_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(score)
        print(f"    RMSE: {score:.5f}")
        
        test_preds_accum += model.predict(X_test_fold)

    avg_rmse = np.mean(fold_scores)
    print(f"\nAverage CV RMSE: {avg_rmse:.5f}")
    
    # ---------------------------------------------------------
    # 7. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("Generating submission...")
    final_preds = np.expm1(test_preds_accum / 5)
    final_preds = np.maximum(final_preds, 0)
    
    # Use PROJECT_ROOT to ensure we save in the main folder, not inside src
    submissions_dir = os.path.join(PROJECT_ROOT, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Handle IDs
    if 'id' in test.columns:
        test_ids = test['id']
    else:
        # Fallback if id got dropped (unlikely with this pipeline)
        test_raw_for_id = pd.read_csv(os.path.join(data_dir, "test.csv"))
        test_ids = test_raw_for_id['id']

    submission = pd.DataFrame({
        "ID": test_ids,
        "TARGET": final_preds
    })
    
    # Save submission with RMSE in filename
    sub_path = os.path.join(submissions_dir, f"submission_final_{avg_rmse:.5f}.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")
    
    # Save the final model to the main models directory
    model_path = os.path.join(models_dir, "lightgbm_final.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()