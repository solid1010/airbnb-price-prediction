import pandas as pd
import numpy as np
import os
import sys
import joblib
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# PATH SETUP: Robust Absolute Paths
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
PROJECT_ROOT = os.path.dirname(src_dir)
sys.path.append(PROJECT_ROOT)

from src import features

def main():
    # ---------------------------------------------------------
    # 1. LOAD RAW DATA
    # ---------------------------------------------------------
    print("Loading raw data...")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    if not os.path.exists(data_dir):
        if os.path.exists("../data"):
             data_dir = "../data"
        else:
            print(f"ERROR: Data directory not found: {data_dir}")
            return

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
    test = features.impute_missing_advanced(test)
    test, _ = features.add_room_type_dummies(test)
    test = features.clean_col_names(test)
    
    # K-Means Location Clustering
    print("  Training K-Means...")
    kmeans_model = features.train_kmeans_geo(train, n_clusters=10)
    train = features.add_kmeans_geo_features(train, kmeans_model)
    test = features.add_kmeans_geo_features(test, kmeans_model)
    
    # Categorical handling for CatBoost
    # We ensure they are category type
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
    feats = features.get_feature_columns(train)
    
    # Explicitly add categorical columns if not already in list
    for c in cat_cols:
        if c in train.columns and c not in feats:
            feats.append(c)
            
    # Ensure all features exist in test
    feats = [f for f in feats if f in test.columns]
    
    # Identify which of the final features are actually categorical
    # (This is needed for CatBoost's fit parameter)
    final_cat_features = [c for c in feats if c in cat_cols]
    
    print(f"Selected {len(feats)} features.")
    print(f"Categorical features: {final_cat_features}")
    
    X = train[feats]
    
    # Target handling (Winsorization)
    if 'price' in train.columns:
        train = train.dropna(subset=['price_num']).reset_index(drop=True)
        print("  Outlier Handling: Applying Winsorization (Soft Clipping 99%)...")
        train = features.handle_outliers_winsorization(train, columns=['price_num'], limits=(0.00, 0.99))
        
        X = train[feats]
        y = np.log1p(train['price_num'])
    else:
         raise ValueError("Price column missing")

    X_test = test[feats]
    
    # ---------------------------------------------------------
    # 4. SCALING
    # ---------------------------------------------------------
    print("Scaling features...")
    
    X, scaler = features.scale_features_robust(X)
    X_test, _ = features.scale_features_robust(X_test, scaler)
    
    y_np = y.to_numpy()

    # ---------------------------------------------------------
    # 5. OPTUNA OPTIMIZATION
    # ---------------------------------------------------------
    print("\nStarting Optuna Hyperparameter Optimization for CatBoost...")
    
    def objective(trial):
        
        params = {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'logging_level': 'Silent',
            'thread_count': -1,
            'n_estimators': 1500, 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08), 
            'depth': trial.suggest_int('depth', 4, 8),           
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2, 10), 
            
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
        }
        
        kf_opt = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf_opt.split(X, y_np):
            # Must split X using iloc for DataFrame
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y_np[train_idx], y_np[val_idx]
            
            # --- Manual Target Encoding ---
            X_tr["temp_target"] = y_tr
            X_tr, mapping = features.target_encode(X_tr, by="neighbourhood_cleansed", target="temp_target")
            X_val["dummy"] = 0
            X_val, _ = features.target_encode(X_val, by="neighbourhood_cleansed", target="dummy", mapping=mapping)
            
            X_tr = X_tr.drop(columns=["temp_target"])
            X_val = X_val.drop(columns=["dummy"])
            # ------------------------------
            
            model = CatBoostRegressor(**params)
            
            model.fit(
                X_tr, y_tr, 
                eval_set=(X_val, y_val), 
                early_stopping_rounds=50,
                cat_features=final_cat_features  
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=80) 
    
    print(f"Best RMSE from Optuna: {study.best_value:.5f}")
    print(f"Best Optuna Params: {study.best_params}")

    # ---------------------------------------------------------
    # 6. FINAL TRAINING
    # ---------------------------------------------------------
    print("\nStarting Final 5-Fold Cross Validation with optimized parameters...")
    
    best_params = study.best_params
    best_params.update({
        'n_estimators': 8000,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'logging_level': 'Silent',
        'early_stopping_rounds': 500,
        'thread_count': -1
    })
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_accum = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_np)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        
        # Target Encoding
        X_tr["temp_target"] = y_tr
        X_tr, mapping_nb = features.target_encode(X_tr, by="neighbourhood_cleansed", target="temp_target", m=50.0)
        X_val["dummy"] = 0
        X_val, _ = features.target_encode(X_val, by="neighbourhood_cleansed", target="dummy", mapping=mapping_nb)
        
        X_tr, mapping_pt = features.target_encode(X_tr, by="property_type", target="temp_target", m=50.0)
        X_val, _ = features.target_encode(X_val, by="property_type", target="dummy", mapping=mapping_pt)
        
        X_tr = X_tr.drop(columns=["temp_target"])
        X_val = X_val.drop(columns=["dummy"])
        
        # Apply to Test
        X_test_fold = X_test.copy()
        X_test_fold["dummy"] = 0
        X_test_fold, _ = features.target_encode(X_test_fold, by="neighbourhood_cleansed", target="dummy", mapping=mapping_nb)
        X_test_fold, _ = features.target_encode(X_test_fold, by="property_type", target="dummy", mapping=mapping_pt)
        X_test_fold = X_test_fold.drop(columns=["dummy"])
        
        model = CatBoostRegressor(**best_params)
   
        model.fit(
            X_tr, y_tr, 
            eval_set=(X_val, y_val), 
            use_best_model=True,
            cat_features=final_cat_features 
        )
        
        val_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(score)
        print(f"    RMSE: {score:.5f}")
        
        test_preds_accum += model.predict(X_test_fold)

    avg_rmse = np.mean(fold_scores)
    print(f"\nAverage CatBoost CV RMSE: {avg_rmse:.5f}")
    
    # ---------------------------------------------------------
    # 7. SUBMISSION
    # ---------------------------------------------------------
    print("Generating submission...")
    final_preds = np.expm1(test_preds_accum / 5)
    final_preds = np.maximum(final_preds, 0)
    
    submissions_dir = os.path.join(PROJECT_ROOT, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    
    if 'id' in test.columns:
        test_ids = test['id']
    else:
        test_raw_for_id = pd.read_csv(os.path.join(data_dir, "test.csv"))
        test_ids = test_raw_for_id['id']

    submission = pd.DataFrame({
        "ID": test_ids,
        "TARGET": final_preds
    })
    
    sub_path = os.path.join(submissions_dir, f"submission_catboost_optuna_{avg_rmse:.5f}.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved to {sub_path}")
    
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "catboost_optuna_final.pkl"))

if __name__ == "__main__":
    main()