import re
import ast
import numpy as np
import pandas as pd

# Basic helpers

def drop_unnamed(df):
    """Drop Unnamed columns."""
    df = df.copy()
    cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    return df.drop(columns=cols, errors="ignore")


def clean_price(x):
    """Convert price text to float."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d.,-]", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan


def clean_percent(x):
    """Convert '85%' to 0.85."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").strip().lower()
    if s in ["", "nan", "none"]:
        return np.nan
    try:
        return float(s) / 100
    except:
        return np.nan


def tf_to_binary(x):
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



# Text parsing


def extract_bathrooms(text):
    """Extract number from bathrooms_text."""
    if pd.isna(text):
        return np.nan
    s = str(text).lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1))
    if "half" in s:
        return 0.5
    return np.nan


def is_shared_bathroom(text):
    """1 if shared, 0 if private."""
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
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, set, tuple)):
            return [str(v).strip().lower() for v in obj]
    except:
        pass
    s = s.strip("{}[]")
    return [p.strip().lower() for p in s.split(",") if p.strip()]



# Core feature builder


def add_features(df):
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
        df["bathrooms_num"] = df["bathrooms_text"].apply(extract_bathrooms)
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
        df["host_tenure_days"] = (
            df["last_scraped_dt"] - df["host_since_dt"]
        ).dt.days

    return df


def add_log_target(df):
    """Create log_price (train only)."""
    df = df.copy()
    if "price_num" not in df.columns:
        raise ValueError("price_num missing")
    df.loc[df["price_num"] < 0, "price_num"] = np.nan
    df["log_price"] = np.log1p(df["price_num"])
    return df



# Reviews features

def build_reviews_features(reviews_df, ref_date):
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


def merge_reviews_features(df, reviews_feat):
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



# Calendar features


def build_calendar_features(calendar_df):
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


def merge_calendar_features(df, cal_feat):
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


# Feature selector

def get_feature_columns(df):
    """Return model-ready numeric features."""
    cols = [
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
    ]
    return [c for c in cols if c in df.columns]
