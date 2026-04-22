"""
predict.py
──────────
Trains the best model (Random Forest) on the full dataset,
then predicts the price of a custom house defined below.

Usage:
    python src/predict.py

To predict your own house, edit the NEW_HOUSE dictionary at the bottom.
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "melb_data.csv")

FEATURES = [
    "Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
    "Landsize", "BuildingArea", "YearBuilt", "Propertycount"
]
TARGET = "Price"
SEED   = 42

# ── Train model on full dataset ────────────────────────────────────────────────

def train_model() -> Pipeline:
    """Train a Random Forest pipeline on the complete dataset."""
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(n_estimators=100, random_state=SEED)),
    ])
    model.fit(X, y)
    print(f"✅ Model trained on {len(df):,} houses.")
    return model

# ── Predict ────────────────────────────────────────────────────────────────────

def predict_price(model: Pipeline, house: dict) -> float:
    """Return the predicted price for a house defined as a dictionary."""
    df_house = pd.DataFrame([house])
    price    = model.predict(df_house)[0]
    return price

# ── Main ───────────────────────────────────────────────────────────────────────

# ✏️  EDIT THIS DICTIONARY to predict the price of your own house:
NEW_HOUSE = {
    "Rooms":         3,       # Total number of rooms
    "Distance":      10,      # Distance from Melbourne CBD (km)
    "Bedroom2":      3,       # Number of bedrooms
    "Bathroom":      1,       # Number of bathrooms
    "Car":           1,       # Number of car spots
    "Landsize":      500,     # Land size in m²
    "BuildingArea":  120,     # Building area in m²
    "YearBuilt":     1990,    # Year the house was built
    "Propertycount": 4000,    # Number of properties in the suburb
}

if __name__ == "__main__":
    print("=" * 50)
    print("  HOUSE PRICE PREDICTOR — Melbourne")
    print("=" * 50)

    model = train_model()
    price = predict_price(model, NEW_HOUSE)

    print("\n🏠 House features:")
    for key, val in NEW_HOUSE.items():
        print(f"   {key:<16}: {val}")

    print(f"\n💰 Predicted price (Random Forest): ${price:,.0f}")
    print(f"   (≈ ${price/1e6:.2f} million AUD)\n")
