import re
import ast
import numpy as np
import pandas as pd


# -----------------------
# Basic helpers
# -----------------------
def fill_categoricals_unknown(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """Fill missing categorical values with 'Unknown'."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Unknown")
    return df

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
    s = re.sub(r"[^\d.,-]", "", s)
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


# -----------------------
# Text parsing
# -----------------------

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


# -----------------------
# Categorical (room_type) helper
# -----------------------

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


# -----------------------
# Core feature builder
# -----------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create base listing features."""
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

    # Amenities
    if "amenities" in df.columns:
        df["amenities_list"] = df["amenities"].apply(parse_amenities_list)
        df["amenities_count"] = df["amenities_list"].apply(len)

        key_amenities = [
            "wifi", "air conditioning", "kitchen", "heating",
            "washer", "dryer", "parking", "tv", "elevator", "pool"
        ]
        for a in key_amenities:
            col = "amenity_" + a.replace(" ", "_")
            df[col] = df["amenities_list"].apply(lambda lst: int(a in lst))

    # Dates
    if "last_scraped" in df.columns:
        df["last_scraped_dt"] = safe_to_datetime(df["last_scraped"])
    if "host_since" in df.columns:
        df["host_since_dt"] = safe_to_datetime(df["host_since"])

    # Host tenure
    if {"last_scraped_dt", "host_since_dt"}.issubset(df.columns):
        df["host_tenure_days"] = (df["last_scraped_dt"] - df["host_since_dt"]).dt.days

    # -----------------------
    # NEW: ratio / per-person features
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
# Reviews features
# -----------------------

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


# -----------------------
# Calendar features
# -----------------------

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


# -----------------------
# Feature selector
# -----------------------

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
    ]

    # room_type dummies
    room_cols = [c for c in df.columns if c.startswith("room_")]

    cols = base_cols + room_cols
    return [c for c in cols if c in df.columns]
