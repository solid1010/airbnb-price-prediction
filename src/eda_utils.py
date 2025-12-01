import pandas as pd
import numpy as np
import ast
import re

class EDA:
    def __init__(self):
        self.median_values = {}
        self.cat_cols = []
        self.num_cols = []
        self.bool_cols = []
        self.key_amenities = [
            'wifi', 'air conditioning', 'kitchen', 'heating', 'washer',
            'dryer', 'parking', 'tv', 'elevator', 'pool'
        ]

    def clean_price(self, price):
        if isinstance(price, str):
            return float(price.replace('$', '').replace(',', ''))
        return price

    def parse_amenities(self, val):
        """Convert amenities text into a cleaned list."""
        if pd.isna(val):
            return []
        s = str(val)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, set, tuple)):
                return [str(x).strip().lower() for x in obj]
        except (SyntaxError, ValueError):
            pass
        s = s.strip('{}[]')
        return [x.strip().lower() for x in s.split(',') if x.strip()]

    def add_amenity_features(self, df):
        """Add amenity count and key amenity flags."""
        if 'amenities' not in df.columns:
            return df
        
        # Avoid SettingWithCopyWarning
        df = df.copy()
        
        df['amenities_list'] = df['amenities'].apply(self.parse_amenities)
        df['amenity_count'] = df['amenities_list'].apply(len)
        
        for amen in self.key_amenities:
            col_name = 'amenity_' + amen.replace(' ', '_')
            df[col_name] = df['amenities_list'].apply(lambda lst: int(amen in lst))
            
        # Drop intermediate list column if desired, or keep it. 
        # The notebook kept it, so we keep it or drop it? 
        # Usually better to drop object columns before ML, but for EDA it's fine.
        # Let's drop it to be clean.
        df = df.drop(columns=['amenities_list'])
        
        return df

    def fit(self, df):
        """Learn median values and column types from training data."""
        # Identify columns
        self.num_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.bool_cols = df.select_dtypes(include=['boolean']).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Calculate medians for numeric
        for col in self.num_cols:
            self.median_values[col] = df[col].median()
            
    def transform(self, df):
        """Apply cleaning and imputation."""
        df = df.copy()
        
        # Clean price if exists
        if 'price' in df.columns:
            df['price'] = df['price'].apply(self.clean_price)
            
        # Amenities
        df = self.add_amenity_features(df)
        
        # Impute Missing Values
        # 1. Categorical -> "Unknown"
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                
        # 2. Numeric -> Median
        for col in self.num_cols:
            if col in df.columns:
                median_val = self.median_values.get(col, 0)
                df[col] = df[col].fillna(median_val)
        
        # 3. Boolean -> False
        for col in self.bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False)
                
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
