import re
import ast
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler


# Preprocessing Functions

def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Unnamed columns."""
    df = df.copy()
    cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    return df.drop(columns=cols, errors="ignore")

def clean_price(x) -> float:
    """Convert price text to float."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d.,-]", "", s) # regex with GPT
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_percent(x) -> float:
    """Convert '85%' to 0.85."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").strip().lower()
    if s in ["", "nan", "none"]:
        return np.nan
    try:
        return float(s) / 100
    except Exception:
        return np.nan

def tf_to_binary(x) -> float:
    """Convert 't'/'f' to 1/0."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if s == "t":
        return 1
    if s == "f":
        return 0
    return np.nan

def safe_to_datetime(s):
    """Parse datetime safely."""
    return pd.to_datetime(s, errors="coerce")
def fill_categoricals_unknown(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """Fill missing categorical values with 'Unknown'."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Unknown")
    return df



def impute_missing_advanced(df: pd.DataFrame, target_cols=None) -> pd.DataFrame:
    """
    Imputation: Uses MEDIAN instead of MICE.
    Why? MICE can introduce noise/leakage. Median is robust, deterministic, and safe.
    """
    df = df.copy()

    # Select numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Exclude targets/IDs to be safe from leakage
    exclude = ["price", "price_num", "log_price", "id", "scrape_id", "host_id"]
    cols_to_impute = [c for c in numeric_cols if c not in exclude]

    # Fill with Median
    for c in cols_to_impute:
        # Check if column has missing values
        if df[c].isnull().any():
            # Add a flag column so the model knows it was originally missing
            df[f"{c}_is_missing"] = df[c].isnull().astype(int)

            # Fill with median value
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)

    print("  Imputation: Filled missing values with Median (Safe Mode).")
    return df

def handle_outliers_winsorization(df: pd.DataFrame, columns=None, limits=(0.01, 0.99)) -> pd.DataFrame:
    """
    Clips outliers in specified columns to the lower and upper percentile limits
    instead of dropping them. This prevents data loss while handling extreme values.
    """
    df = df.copy()

    # If no columns provided, automatically select numeric columns with enough unique values
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                   and df[c].nunique() > 10]

    lower_quantile, upper_quantile = limits

    for col in columns:
        if col in df.columns:
            # Calculate lower and upper bounds based on quantiles
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)

            # Clip values: values < lower_bound become lower_bound,
            # values > upper_bound become upper_bound.
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names to remove special characters, spaces, and non-ASCII chars.
    This is required to prevent errors in models like LightGBM and XGBoost.
    """
    df = df.copy()
    new_cols = []
    seen_cols = {}

    for col in df.columns:
        # Regex: Keep only alphanumeric characters and underscores. Remove everything else.
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', str(col))

        # Handle duplicates: If "Wifi" and "Wi-Fi" both become "Wifi", append a counter (Wifi_1)
        if new_col in seen_cols:
            seen_cols[new_col] += 1
            new_col = f"{new_col}_{seen_cols[new_col]}"
        else:
            seen_cols[new_col] = 1

        new_cols.append(new_col)

    df.columns = new_cols
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicates based on physical attributes (subset) rather than just ID,
    keeping the most recent record.
    """
    df = df.copy()

    # Ensure date column is datetime to sort by recency
    if "last_scraped" in df.columns:
        df["last_scraped"] = pd.to_datetime(df["last_scraped"], errors="coerce")
        # Sort by date descending to keep the newest record first
        df = df.sort_values(by="last_scraped", ascending=False)

    # Define the "fingerprint" of a listing.
    # Even if IDs differ, listings with same location, room type, and price are likely duplicates.
    subset_cols = [
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights"
    ]

    # Use only columns present in the dataframe
    valid_subset = [c for c in subset_cols if c in df.columns]

    if valid_subset:
        initial_len = len(df)
        # Drop duplicates based on subset, keeping the first (newest) one
        df = df.drop_duplicates(subset=valid_subset, keep="first")
        print(f"Duplicate Removal: {initial_len - len(df)} rows dropped using subset logic.")
    else:
        df = df.drop_duplicates()

    return df


# Feature Engineering Functions



def scale_features_robust(df: pd.DataFrame, scaler=None) -> tuple[pd.DataFrame, object]:
    """
    Scales numeric features using RobustScaler.
    RobustScaler removes the median and scales the data according to the quantile range (IQR).
    It is robust to outliers, unlike StandardScaler.
    """
    df = df.copy()

    # Select numeric columns to scale
    # Exclude target variables (price) and ID columns to prevent data leakage or errors
    exclude_cols = ['id', 'scrape_id', 'host_id', 'price', 'price_num', 'log_price']

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                    and c not in exclude_cols]

    # Safety check
    if not numeric_cols:
        return df, scaler

    # Fit a new scaler
    if scaler is None:
        scaler = RobustScaler()
        # Fit on train data
        scaler.fit(df[numeric_cols])

    # Apply scaling (works for both train and test)
    scaled_values = scaler.transform(df[numeric_cols])

    # Update dataframe with scaled values
    df[numeric_cols] = scaled_values

    return df, scaler
def select_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes only the features that have 0 variance (constant values).
    These features provide no information to the model.
    """
    df = df.copy()
    initial_shape = df.shape

    # Drop Constant Numeric Columns (Variance = 0)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        var_check = df[numeric_cols].var()
        # Find the columns which has a variance of 0 or NaN values.
        constant_numeric = var_check[var_check == 0].index.tolist()
    else:
        constant_numeric = []

    # Drop Constant Categorical Columns (Unique count <= 1)
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    constant_cat = [c for c in categorical_cols if df[c].nunique() <= 1]

    # Combine lists
    drop_cols = constant_numeric + constant_cat

    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Feature Selection: Dropped {len(drop_cols)} constant columns: {drop_cols}")

    print(f"Feature Selection Complete: Shape changed from {initial_shape} to {df.shape}")
    return df






def parse_bathrooms(text) -> float:
    """
    Parse bathrooms_text into a numeric bathrooms count.

    Examples:
    - "1 bath" -> 1.0
    - "1.5 baths" -> 1.5
    - "Shared half-bath" -> 0.5
    - "Shared bath" -> 0.5 (fallback assumption)
    """
    if pd.isna(text):
        return np.nan

    s = str(text).lower()

    # explicit "half"
    if "half" in s:
        return 0.5

    # shared but no number -> assume 0.5
    if "shared" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return 0.5

    # normal numeric parse
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1))

    return np.nan


def is_shared_bathroom(text):
    """1 if shared, 0 if private, NaN otherwise."""
    if pd.isna(text):
        return np.nan
    s = str(text).lower()
    if "shared" in s:
        return 1
    if "private" in s:
        return 0
    return np.nan


def parse_amenities_list(x):
    """Parse amenities into list."""
    if pd.isna(x):
        return []
    s = str(x)

    # try python literal (list-like)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, set, tuple)):
            return [str(v).strip().lower() for v in obj]
    except Exception:
        pass

    # fallback: split
    s = s.strip("{}[]")
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def add_room_type_dummies(df: pd.DataFrame, dummy_cols=None):
    """
    Add room_type one-hot columns.
    If dummy_cols is provided, ensures output contains exactly those columns.
    Returns (df_out, used_dummy_cols).
    """
    df = df.copy()

    if "room_type" not in df.columns:
        return df, (dummy_cols or [])

    dummies = pd.get_dummies(df["room_type"], prefix="room", drop_first=True)

    if dummy_cols is None:
        # fit mode: take whatever appears in df
        used_cols = list(dummies.columns)
        df = pd.concat([df, dummies], axis=1)
        return df, used_cols

    # transform mode: force same cols as train
    for c in dummy_cols:
        if c not in dummies.columns:
            dummies[c] = 0

    dummies = dummies[dummy_cols] if len(dummy_cols) else dummies
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummy_cols)





def target_encode(df: pd.DataFrame, by: str, target: str, m: float = 10.0, mapping: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Applies Target Encoding with Smoothing to handle high-cardinality categorical features.
    Replaces a category (e.g., 'neighbourhood') with the smoothed mean of the target (e.g., 'price').

    Returns:
        df (pd.DataFrame): Dataframe with the new encoded column.
        mapping (dict): The calculated means (used to apply the same mapping to the test set).
    """
    df = df.copy()
    col_name = f"{by}_encoded"
    
    # 1. Training: Calculate mapping if not provided
    if mapping is None:
        global_mean = df[target].mean()
        agg = df.groupby(by)[target].agg(['count', 'mean'])
        
        # Smoothing Formula: (n * mean + m * global_mean) / (n + m)
        smooth = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
        # Store as dictionary for reuse
        mapping = smooth.to_dict()
        # Save global mean to handle unknown categories in test set
        mapping['global_mean'] = global_mean

    # 2. Transform: Apply mapping
    # Ensure the column used for mapping is treated as a string to avoid categorical issues
    fill_value = mapping.get('global_mean', 0)
    df[col_name] = df[by].astype(str).map(mapping).fillna(fill_value)
    
    return df, mapping


# -----------------------
# Core feature builder
# -----------------------

def add_host_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binned versions of host metrics and ordinal response time.
    
    Why this helps:
    ----------------------------------
    1. Distribution Shift: Addresses the gap in host metrics between Train keys and Test keys.
    2. Robustness: Instead of overfitting to specific percentages (e.g., 98% vs 99%), 
       the model focuses on broader categories (High/Medium/Low), making it more robust.
    """
    df = df.copy()
    
    # 1. Host Response Time Ordinal
    # Map text values to ordinal integers (1=Best/Fastest, 4=Worst/Slowest)
    resp_map = {
        'within an hour': 1,
        'within a few hours': 2,
        'within a day': 3,
        'a few days or more': 4
    }
    if 'host_response_time' in df.columns:
        df['host_response_time_ord'] = df['host_response_time'].map(resp_map).fillna(2) # Default to 'within a few hours'

    # 2. Binning Helpers
    def bin_rate(val):
        """
        Bins percentage rates (0.0-1.0) into 4 main categories.
        Purpose: To reduce noise and increase generalization capabilities.
        """
        if pd.isna(val): return 0 # Unknown
        if val == 1.0: return 4   # Perfect (1.0 because src/features.py converts % to float 0-1)
        if val >= 0.90:  return 3   # High
        if val >= 0.50:  return 2   # Medium
        return 1                  # Low

    # Note: src/features.py already creates host_response_rate_num (0.0 - 1.0 scale)
    # We will use that if available, or clean it ourselves if not.
    
    if 'host_response_rate_num' in df.columns:
        df['host_response_rate_bin'] = df['host_response_rate_num'].apply(bin_rate)
        
    if 'host_acceptance_rate_num' in df.columns:
        df['host_acceptance_rate_bin'] = df['host_acceptance_rate_num'].apply(bin_rate)
        
    return df

def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and encode amenity features."""
    if "amenities" not in df.columns:
        return df
        
    df["amenities_list"] = df["amenities"].apply(parse_amenities_list)
    df["amenities_count"] = df["amenities_list"].apply(len)

    # EXPANDED AMENITIES LIST (Based on Pricing Analysis)
    key_amenities = [
        # Luxury
        "pool", "gym", "sauna", "hot tub", "view", "elevator",
        "waterfront", "patio or balcony", "private entrance",
        
        # Extras
        "air conditioning", "heating", "washer", "dryer", "dishwasher",
        
        # Must Haves
        "wifi", "kitchen", "tv", "dedicated workspace",
        
        # Security
        "security", "self check-in"
    ]
    
    # Create a dictionary first to avoid DataFrame fragmentation warning
    # (Adding columns one by one inside a loop causes performance issues)
    new_cols = {}
    for a in key_amenities:
        col_name = "amenity_" + a.replace(" ", "_")
        new_cols[col_name] = df["amenities_list"].apply(lambda x: 1 if a in x else 0)
        
    # Concatenate all new columns at once (Faster and Cleaner)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
    # Smart Keyword Aggregation (Catch variants like "Sea view", "Infinity pool")
    smart_keywords = ["view", "pool", "gym", "sauna", "jacuzzi", "sound", "baby", "children"]
    for kw in smart_keywords:
        df[f"has_{kw}"] = df["amenities_list"].apply(lambda lst: int(any(kw in item for item in lst)))
        
    return df

def add_title_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from listing Title/Name."""
    if "name" not in df.columns:
        return df
        
    # Clean name for searching
    s_name = df["name"].fillna("").astype(str).str.lower()
    
    # High Value Keywords (From Analysis)
    title_high_keywords = [
        "bosphorus", "penthouse", "duplex", "jacuzzi", "luxury", 
        "view", "residence", "terrace", "suite", "bomonti",
        "sea", "spa", "ultra", "galata", "taksim" 
    ]
    
    for k in title_high_keywords:
        df[f"title_has_{k}"] = s_name.apply(lambda x: 1 if k in x else 0)
        
    # Low Value / unique indicators
    title_low_keywords = [
        "economy", "shared", "room", "hostel", "budget", "cheap", 
        "metrob", "oda", "paylasimli"
    ]
    
    for k in title_low_keywords:
        df[f"title_is_{k}"] = s_name.apply(lambda x: 1 if k in x else 0)
        
    return df

def add_description_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from listing Description."""
    if "description" not in df.columns:
        return df
        
    # Clean description for searching
    s_desc = df["description"].fillna("").astype(str).str.lower()
    
    # High Value Keywords (From Analysis)
    desc_high_keywords = [
        "bosphorus", "pool", "historic", "luxury", "residence", 
        "security", "terrace", "modern", "families", "unique", 
        "view", "sea", "renovated"
    ]
    
    for k in desc_high_keywords:
        df[f"desc_has_{k}"] = s_desc.apply(lambda x: 1 if k in x else 0)
        
    # Low Value / unique indicators
    desc_low_keywords = [
        "metrobus", "marmaray", "shared", "room", "economy", 
        "budget", "studio", "simple", "basement"
    ]
    
    for k in desc_low_keywords:
        df[f"desc_is_{k}"] = s_desc.apply(lambda x: 1 if k in x else 0)
        
    return df

def add_cyclical_date_features(df: pd.DataFrame, col_name: str, period: str = 'month') -> pd.DataFrame:
    """
    Converts a date column into cyclical sine/cosine features to preserve seasonality.
    Example: Month 12 and Month 1 are numerically far but temporally close.
    """
    df = df.copy()
    
    # Ensure column is datetime
    if col_name not in df.columns:
        return df
        
    dt_col = pd.to_datetime(df[col_name], errors='coerce')
    
    if period == 'month':
        # Months range 1-12
        max_val = 12
        val = dt_col.dt.month
    elif period == 'day':
        # Days range 1-31 (approx)
        max_val = 31
        val = dt_col.dt.day
    elif period == 'weekday':
        # Monday=0, Sunday=6
        max_val = 7
        val = dt_col.dt.weekday
    else:
        return df

    # Apply Sine and Cosine transformation
    # Formula: sin(2 * pi * x / max_val)
    df[f"{col_name}_{period}_sin"] = np.sin(2 * np.pi * val / max_val)
    df[f"{col_name}_{period}_cos"] = np.cos(2 * np.pi * val / max_val)
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create base listing features (Orchestrator)."""
    df = df.copy()
    df = drop_unnamed(df)

    # Price
    if "price" in df.columns:
        df["price_num"] = df["price"].apply(clean_price)

    # Percent rates
    if "host_response_rate" in df.columns:
        df["host_response_rate_num"] = df["host_response_rate"].apply(clean_percent)
    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate_num"] = df["host_acceptance_rate"].apply(clean_percent)

    # Binary flags
    for c in [
        "host_is_superhost",
        "instant_bookable",
        "host_identity_verified",
        "host_has_profile_pic",
    ]:
        if c in df.columns:
            df[c + "_bin"] = df[c].apply(tf_to_binary)

    # Bathrooms
    if "bathrooms_text" in df.columns:
        df["bathrooms_num"] = df["bathrooms_text"].apply(parse_bathrooms)
        df["bathroom_shared_flag"] = df["bathrooms_text"].apply(is_shared_bathroom)

    # Modular Feature Extraction
    df = add_amenity_features(df)
    df = add_title_nlp_features(df)
    df = add_description_nlp_features(df)

    # Dates
    if "last_scraped" in df.columns:
        df["last_scraped_dt"] = safe_to_datetime(df["last_scraped"])
        # Convert "last_scraped" month into cyclical sine/cosine features to capture seasonality
        df = add_cyclical_date_features(df, "last_scraped", period="month")
    if "host_since" in df.columns:
        df["host_since_dt"] = safe_to_datetime(df["host_since"])

    # Host tenure
    if {"last_scraped_dt", "host_since_dt"}.issubset(df.columns):
        df["host_tenure_days"] = (df["last_scraped_dt"] - df["host_since_dt"]).dt.days

    df = add_geospatial_features(df)

    # Custom binning
    df = add_host_binned_features(df)

    # -----------------------
    # ratio / per-person features
    # -----------------------
    if "accommodates" in df.columns:
        df["accommodates_safe"] = df["accommodates"].replace(0, 1)
        if "price_num" in df.columns:
            df["price_per_person"] = df["price_num"] / df["accommodates_safe"]
        if "bedrooms" in df.columns:
            df["bedrooms_per_person"] = df["bedrooms"] / df["accommodates_safe"]
        if "beds" in df.columns:
            df["beds_per_person"] = df["beds"] / df["accommodates_safe"]

    if {"minimum_nights", "maximum_nights"}.issubset(df.columns):
        df["maximum_nights_safe"] = df["maximum_nights"].replace(0, np.nan)
        df["min_max_nights_ratio"] = df["minimum_nights"] / df["maximum_nights_safe"]

    return df

def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create log_price (train only)."""
    df = df.copy()
    if "price_num" not in df.columns:
        raise ValueError("price_num missing")
    df.loc[df["price_num"] < 0, "price_num"] = np.nan
    df["log_price"] = np.log1p(df["price_num"])
    return df

# -----------------------
# Geospatial Feature Engineering
# -----------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth surface.
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes distances to key Istanbul landmarks (Taksim, Sultanahmet, etc.)
    and adds them as new features. Also calculates distance to the nearest center.
    
    Ref: User's Notebook - Geospatial Feature Engineering
    """
    df = df.copy()
    
    # Coordinates must exist
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    # Critical locations in Istanbul impacting price
    locations = {
        "Taksim": (41.0370, 28.9851),
        "Sultanahmet": (41.0054, 28.9768),
        "Besiktas": (41.0422, 29.0060),
        "Kadikoy": (40.9901, 29.0254),
        "Airport": (41.2811, 28.7533)
    }
    
    # Calculate distance for each landmark
    for loc, (lat, lon) in locations.items():
        col_name = f"dist_{loc}"
        df[col_name] = haversine_distance(df["latitude"], df["longitude"], lat, lon)
    
    # Distance to the nearest city center (excluding Airport usually, as logic suggests)
    # We select columns starting with 'dist_' but exclude Airport for the 'center' logic
    center_cols = [f"dist_{loc}" for loc in locations.keys() if loc != "Airport"]
    
    if center_cols:
        df["min_dist_center"] = df[center_cols].min(axis=1)
        
    return df

# ------------------------------------------------------------------------------
# K-Means Clustering (Location Groups)
# ------------------------------------------------------------------------------
from sklearn.cluster import KMeans

def train_kmeans_geo(df: pd.DataFrame, n_clusters: int = 20) -> KMeans:
    """Trains a K-Means model on latitude and longitude."""
    # Fit only on valid coordinates
    coords = df[["latitude", "longitude"]].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(coords)
    return kmeans

def add_kmeans_geo_features(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Adds K-Means cluster IDs as One-Hot encoded features."""
    df = df.copy()
    if kmeans is None:
        return df
        
    # Predict clusters
    # Handle NaNs if any (though lat/lon used be clean by now)
    coords = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean()) 
    clusters = kmeans.predict(coords)
    
    # One-Hot Encoding
    n_clusters = kmeans.n_clusters
    for i in range(n_clusters):
        df[f"geo_cluster_{i}"] = (clusters == i).astype(int)
        
    return df



def build_reviews_features(reviews_df: pd.DataFrame, ref_date) -> pd.DataFrame:
    """Aggregate reviews per listing."""
    r = reviews_df.copy()
    r["date"] = safe_to_datetime(r["date"])

    agg = (
        r.groupby("listing_id", as_index=False)
         .agg(
             review_count=("date", "count"),
             last_review_date=("date", "max")
         )
    )

    ref = safe_to_datetime(ref_date)
    agg["days_since_last_review"] = (ref - agg["last_review_date"]).dt.days
    return agg


def merge_reviews_features(df: pd.DataFrame, reviews_feat: pd.DataFrame) -> pd.DataFrame:
    """Merge reviews by id."""
    df = df.copy()
    if "id" not in df.columns:
        return df
    out = df.merge(
        reviews_feat,
        left_on="id",
        right_on="listing_id",
        how="left"
    )
    return out.drop(columns=["listing_id"], errors="ignore")



def build_calendar_features(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate calendar availability."""
    c = calendar_df.copy()
    c["date"] = safe_to_datetime(c["date"])
    c["available_bin"] = c["available"].map({"t": 1, "f": 0})

    agg = (
        c.groupby("listing_id", as_index=False)
         .agg(
             cal_days=("date", "count"),
             cal_avail_rate=("available_bin", "mean"),
             cal_min_nights_mean=("minimum_nights", "mean"),
             cal_max_nights_mean=("maximum_nights", "mean"),
         )
    )
    return agg


def merge_calendar_features(df: pd.DataFrame, cal_feat: pd.DataFrame) -> pd.DataFrame:
    """Merge calendar by id."""
    df = df.copy()
    if "id" not in df.columns:
        return df
    out = df.merge(
        cal_feat,
        left_on="id",
        right_on="listing_id",
        how="left"
    )
    return out.drop(columns=["listing_id"], errors="ignore")


def get_feature_columns(df: pd.DataFrame):
    """Return model-ready numeric features."""
    base_cols = [
        "accommodates", "bedrooms", "beds",
        "bathrooms_num", "bathroom_shared_flag",
        "minimum_nights", "maximum_nights",
        "latitude", "longitude",
        "amenities_count",
        "host_response_rate_num", "host_acceptance_rate_num",
        "host_is_superhost_bin", "instant_bookable_bin",
        "host_identity_verified_bin",
        "host_tenure_days",
        "review_count", "days_since_last_review",
        "cal_avail_rate", "cal_min_nights_mean", "cal_max_nights_mean",

        # NEW engineered
        "price_per_person",
        "bedrooms_per_person",
        "beds_per_person",
        "min_max_nights_ratio",

        # Recovered Binned Features
        "host_response_time_ord",
        "host_response_rate_bin",
        "host_acceptance_rate_bin",
    ]

    # room_type dummies
    room_cols = [c for c in df.columns if c.startswith("room_") and c != "room_type"]
    
    # amenity dummies
    amenity_cols = [c for c in df.columns if c.startswith("amenity_") and c != "amenities_count"]

    # smart keyword cols
    has_cols = [c for c in df.columns if c.startswith("has_")]

    # title nlp cols
    title_cols = [c for c in df.columns if c.startswith("title_")]

    # description nlp cols
    desc_cols = [c for c in df.columns if c.startswith("desc_")]

    # distance cols
    dist_cols = [c for c in df.columns if c.startswith("dist_")]

    # K-Means Geo cols
    geo_cluster_cols = [c for c in df.columns if c.startswith("geo_cluster_")]

    # 4. EXPANDED FEATURES (Review, Availability, Host Counts)
    review_cols = [c for c in df.columns if c.startswith("review_scores_")]
    avail_cols = [c for c in df.columns if c.startswith("availability_")]
    host_list_cols = [c for c in df.columns if c.startswith("calculated_host_")]

    cols = base_cols + room_cols + amenity_cols + has_cols + title_cols + desc_cols + dist_cols + geo_cluster_cols + review_cols + avail_cols + host_list_cols
    # Remove duplicates if any
    cols = list(dict.fromkeys(cols))
    
    return [c for c in cols if c in df.columns]






