"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the
Melbourne Housing Price Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Features used for training
FEATURES = [
    'Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
    'Landsize', 'BuildingArea', 'YearBuilt', 'Propertycount'
]
TARGET = 'Price'


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("\n--- Shape ---")
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n--- Price Statistics ---")
    print(df[TARGET].describe().apply(lambda x: f"${x:,.0f}"))

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\n--- Missing Values ---")
        for col, n in missing.items():
            print(f"  {col:<20} {n:>5} ({n/len(df)*100:.1f}%)")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select relevant features, drop rows with missing target.
    Returns X (features) and y (target).
    """
    df_clean = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    print(f"✅ After cleaning: {len(df_clean):,} rows retained")

    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split into train / test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test
