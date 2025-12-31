# Airbnb Nightly Price Prediction - Team Overfitters

**YZV 311E - Data Mining (Fall 2025-2026)** **Istanbul Technical University**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Status](https://img.shields.io/badge/Status-Active-green)

## Project Overview
This project is developed for the **YZV 311E Data Mining** course competition. The main objective is to predict the nightly price of Airbnb listings based on a diverse set of features including host details, property descriptions, availability, and customer reviews.

The project focuses on the complete **Data Mining Pipeline**:
1.  **Data Understanding & EDA:** Analyzing distributions, correlations, and outliers.
2.  **Preprocessing:** Handling missing values, cleaning text data, and formatting prices.
3.  **Feature Engineering:** Creating interpretable features from text (NLP), dates, and geospatial data.
4.  **Modeling:** Implementing baseline regressions and advanced ensemble methods (XGBoost/LightGBM).
5.  **Evaluation:** Optimizing for **RMSLE** (Root Mean Squared Logarithmic Error).

## Team Members (Team Overfitters)
* **İbrahim Bancar** - Data Exploration & Preprocessing
* **Hasan Kan** - Feature Engineering & Modeling
* **Alperen Sağlam** - Evaluation & Reporting

## Repository Structure
```text
├── data/                  # Raw and processed data (Not included in git)
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   ├── reviews.csv        # Reviews data
│   └── calendar.csv       # Availability calendar data
├── notebooks/             # Jupyter Notebooks for experiments
│   ├── 01_EDA_Data_Understanding.ipynb
│   ├── 01_Baseline_Model.ipynb
│   ├── 02_features.ipynb
│   ├── 02_Advanced_Model.ipynb
│   ├── 03_Model_Improvement_Geospatial_Optuna.ipynb
│   └── 03_Model_Evaluation_and_Reporting.ipynb
├── src/                   # Production-ready source code
│   ├── features.py        # Feature engineering pipeline
│   ├── lightgbm_model.py  # LightGBM model training script
│   ├── catboost_model.py  # CatBoost model training script
│   └── ensemble.py        # Model ensemble and blending
├── analysis/              # Analysis scripts for feature exploration
│   ├── analyze_amenities_detailed.py
│   ├── analyze_titles.py
│   ├── analyze_descriptions.py
│   └── analyze_importance.py
├── models/                # Saved model files
├── submissions/           # Kaggle submission files
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Feature Engineering Pipeline (`src/features.py`)

Our feature engineering module provides a comprehensive toolkit for transforming raw Airbnb data into model-ready features:

### Data Preprocessing
- **Missing Value Imputation**: Median imputation with missing indicator flags
- **Outlier Handling**: Winsorization (soft clipping at 99th percentile)
- **Duplicate Removal**: Fingerprint-based duplicate detection
- **Column Name Cleaning**: Sanitization for model compatibility

### Feature Creation

#### 1. Text & NLP Features
- **Amenity Parsing**: Extracts and encodes 20+ key amenities (pool, gym, wifi, etc.)
- **Title Keywords**: Luxury indicators (bosphorus, penthouse, duplex, jacuzzi)
- **Description NLP**: High-value keywords (historic, renovated, security)
- **Smart Aggregation**: Synonym detection for amenity variants

#### 2. Geospatial Features
- **Distance Calculations**: Haversine distance to Istanbul landmarks
  - Taksim Square
  - Sultanahmet (Historic Center)
  - Besiktas
  - Kadikoy
  - Airport
- **K-Means Clustering**: Location-based grouping (n=10 clusters)
- **Minimum Distance to Center**: Proximity to nearest city center

#### 3. Host Features
- **Binned Metrics**: Categorical binning of host response/acceptance rates
  - Perfect (100%)
  - High (90-99%)
  - Medium (50-89%)
  - Low (<50%)
- **Response Time Ordinal**: Ordered ranking (1=fastest, 4=slowest)
- **Host Tenure**: Days since host registration

#### 4. Temporal Features
- **Cyclical Encoding**: Sine/cosine transformation for seasonality
  - Month
  - Day
  - Weekday
- **Days Since Last Review**: Recency indicator

#### 5. Engineered Ratios
- Price per person
- Bedrooms per person
- Beds per person
- Min/max nights ratio

#### 6. Target Encoding
- **Smoothed Target Encoding** for high-cardinality categoricals
  - `neighbourhood_cleansed`: Encoded by mean price
  - `property_type`: Encoded by mean price
  - Smoothing parameter: m=50.0 (prevents overfitting on rare categories)

#### 7. Scaling
- **RobustScaler**: Median-based scaling (robust to outliers)

## Model Training

### LightGBM Model (`src/lightgbm_model.py`)

**Hyperparameter Optimization:**
- Optuna-based search (80 trials)
- 3-fold CV for hyperparameter tuning
- 5-fold CV for final model training

**Key Features:**
- Early stopping (300 rounds)
- Target encoding inside CV loop (prevents data leakage)
- Log-price target transformation
- Regularization: L1/L2 penalties
- Best CV RMSE: ~0.432

**Training Pipeline:**
1. Load raw data
2. Apply feature engineering
3. Optuna optimization
4. 5-fold cross-validation
5. Test prediction (averaged across folds)

### CatBoost Model (`src/catboost_model.py`)

**Hyperparameter Optimization:**
- Optuna-based search (80 trials)
- Native categorical feature handling
- 3-fold CV for tuning, 5-fold CV for final model

**Key Features:**
- Categorical encoding: `neighbourhood_cleansed`, `property_type`
- Early stopping (500 rounds)
- Target encoding with smoothing
- Best CV RMSE: ~0.427

**Advantages:**
- Better handling of categorical features
- More robust to overfitting
- Faster training than LightGBM

### Ensemble Model (`src/ensemble.py`)

**Blending Strategy:**
- Simple average: 50% LightGBM + 50% CatBoost
- Robust to individual model weaknesses
- Reduces variance and improves generalization

**Usage:**
```bash
python src/ensemble.py
```

## Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train LightGBM Model
```bash
python src/lightgbm_model.py
```
**Output:**
- `submissions/submission_final_<rmse>.csv`
- `models/lightgbm_final.pkl`

### 3. Train CatBoost Model
```bash
python src/catboost_model.py
```
**Output:**
- `submissions/submission_catboost_optuna_<rmse>.csv`
- `models/catboost_optuna_final.pkl`

### 4. Create Ensemble
```bash
python src/ensemble.py
```
**Output:**
- `submissions/submission_ensemble_lgbm_cat.csv`

## Key Technical Decisions

### Why Median Imputation?
- **Robust to outliers**: Unlike mean, median is not affected by extreme values
- **Deterministic**: Reproducible results across runs
- **Safe from leakage**: No cross-contamination between train/test

### Why Winsorization over Dropping?
- **Data preservation**: Keeps all data points
- **Soft clipping**: Caps extreme values instead of removing them
- **Better generalization**: Prevents model from learning noise

### Why Target Encoding?
- **High-cardinality problem**: 50+ neighborhoods, 20+ property types
- **One-hot explosion**: Would create 70+ sparse columns
- **Smoothing**: Prevents overfitting on rare categories
- **Formula**: `(n × category_mean + m × global_mean) / (n + m)`

### Why RobustScaler?
- **Outlier resistance**: Uses IQR instead of standard deviation
- **Median centering**: More stable than mean centering
- **Better for Airbnb data**: Extreme luxury listings don't skew scaling

### Why Ensemble?
- **Diversity**: LightGBM and CatBoost have different strengths
- **Variance reduction**: Averaging reduces overfitting
- **Robust predictions**: Less sensitive to individual model errors

## Model Performance

| Model | CV RMSE | Description |
|-------|---------|-------------|
| Baseline Ridge | 0.607 | Simple linear regression |
| LightGBM (Optuna) | 0.432 | Gradient boosting with hyperparameter tuning |
| CatBoost (Optuna) | 0.427 | Better categorical handling |
| Ensemble (50-50) | ~0.425 | Combined predictions |

## Feature Importance Insights

**Top 10 Most Important Features:**
1. `latitude` / `longitude` - Location is paramount
2. `dist_Taksim` - Proximity to city center
3. `accommodates` - Capacity drives price
4. `property_type_encoded` - Property category matters
5. `neighbourhood_cleansed_encoded` - Neighborhood premium
6. `amenity_pool` - Luxury amenity signal
7. `host_tenure_days` - Experienced hosts charge more
8. `geo_cluster_X` - Location clusters
9. `title_has_bosphorus` - Bosphorus view premium
10. `bathrooms_num` - More bathrooms = higher price

## Competition Submission

**Final Submission:**
- File: `submission_ensemble_lgbm_cat.csv`
- Format: `ID, TARGET` (predicted nightly price)
- Metric: RMSLE (Root Mean Squared Logarithmic Error)

## Dependencies

**Core Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - ML utilities, scaling, metrics
- `lightgbm` - LightGBM model
- `catboost` - CatBoost model
- `optuna` - Hyperparameter optimization
- `joblib` - Model serialization

## Analysis Scripts

The `analysis/` directory contains exploratory scripts:
- `analyze_amenities_detailed.py` - Amenity frequency and pricing impact
- `analyze_titles.py` - Title keyword extraction
- `analyze_descriptions.py` - Description text mining
- `analyze_importance.py` - Feature importance visualization

## Notes

- All paths are relative to project root
- Models are saved in `models/` directory
- Submissions are saved in `submissions/` directory
- Feature engineering is fully modular and reusable
- Target encoding is applied inside CV loop to prevent data leakage
