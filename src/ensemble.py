import pandas as pd
import os
import sys

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
PROJECT_ROOT = os.path.dirname(src_dir)
submissions_dir = os.path.join(PROJECT_ROOT, "submissions")

def main():
    print("Starting Ensemble Process...")
    
    # ---------------------------------------------------------
    # 1. DEFINE FILENAMES
    # ---------------------------------------------------------
    # Using the specific files you generated
    file_lgbm = "submission_final_0.43268.csv"          # Best LightGBM
    file_cat  = "submission_catboost_optuna_0.42783.csv" # Best CatBoost
    
    path_lgbm = os.path.join(submissions_dir, file_lgbm)
    path_cat = os.path.join(submissions_dir, file_cat)
    
    # Validation
    if not os.path.exists(path_lgbm):
        print(f"ERROR: LightGBM file not found: {path_lgbm}")
        return
    if not os.path.exists(path_cat):
        print(f"ERROR: CatBoost file not found: {path_cat}")
        return

    # ---------------------------------------------------------
    # 2. LOAD SUBMISSIONS
    # ---------------------------------------------------------
    print(f"Loading LightGBM: {file_lgbm}")
    df_lgbm = pd.read_csv(path_lgbm)
    
    print(f"Loading CatBoost: {file_cat}")
    df_cat = pd.read_csv(path_cat)
    
    # Ensure ID alignment
    if not df_lgbm['ID'].equals(df_cat['ID']):
        print("WARNING: ID mismatch detected! Sorting by ID to fix alignment...")
        df_lgbm = df_lgbm.sort_values('ID').reset_index(drop=True)
        df_cat = df_cat.sort_values('ID').reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 3. BLENDING (AVERAGING)
    # ---------------------------------------------------------
    print("Calculating weighted average (50% LGBM + 50% CatBoost)...")
    
    # Simple Average (Robust Ensemble Strategy)
    final_preds = (df_lgbm['TARGET'] * 0.5) + (df_cat['TARGET'] * 0.5)
    
    # ---------------------------------------------------------
    # 4. SAVE FINAL SUBMISSION
    # ---------------------------------------------------------
    submission = pd.DataFrame({
        "ID": df_lgbm['ID'],
        "TARGET": final_preds
    })
    
    output_filename = "submission_ensemble_lgbm_cat.csv"
    output_path = os.path.join(submissions_dir, output_filename)
    
    submission.to_csv(output_path, index=False)
    print(f"\nEnsemble saved to: {output_path}")

if __name__ == "__main__":
    main()