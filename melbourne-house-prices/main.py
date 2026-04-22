"""
main.py
-------
Entry point for the Melbourne House Price Prediction project.
Run this file to reproduce all results and save charts to outputs/.

Usage:
    python main.py
    python main.py --data data/melb_data.csv
"""

import argparse
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_data, summarize, clean_data, split_data, FEATURES
from src.models import get_models, train_all, best_model, predict
from src.visualization import (
    plot_price_distribution,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
)

OUTPUTS = "outputs"


def main(data_path: str = "data/melb_data.csv"):
    os.makedirs(OUTPUTS, exist_ok=True)

    print("\n" + "="*60)
    print("  MELBOURNE HOUSE PRICE PREDICTION")
    print("="*60)

    # ── 1. Load & explore ──────────────────────────────────────
    print("\n📦 STEP 1 — Load & Explore")
    df = load_data(data_path)
    summarize(df)

    # ── 2. Clean & split ──────────────────────────────────────
    print("\n🧹 STEP 2 — Clean & Split")
    X, y = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Price distribution chart
    plot_price_distribution(y, save_path=f"{OUTPUTS}/01_price_distribution.png")

    # ── 3. Train & compare models ─────────────────────────────
    print("\n🤖 STEP 3 — Train & Evaluate All Models")
    models  = get_models()
    summary, results = train_all(models, X_train, X_test, y_train, y_test)

    print("\n📊 Leaderboard (sorted by R²):")
    print(summary.to_string(index=False,
        formatters={
            "MAE":  lambda x: f"${x:>12,.0f}",
            "RMSE": lambda x: f"${x:>12,.0f}",
            "R2":   lambda x: f"{x:.3f}",
            "CV_R2":lambda x: f"{x:.3f}",
        }
    ))

    plot_model_comparison(summary, save_path=f"{OUTPUTS}/02_model_comparison.png")

    # ── 4. Best model ─────────────────────────────────────────
    print("\n🏆 STEP 4 — Best Model")
    name, best = best_model(results)
    print(f"  Winner : {name}")
    print(f"  R²     : {best['r2']:.3f}")
    print(f"  MAE    : ${best['mae']:,.0f}")
    print(f"  RMSE   : ${best['rmse']:,.0f}")

    plot_actual_vs_predicted(
        y_test, best["y_pred"], name,
        save_path=f"{OUTPUTS}/03_actual_vs_predicted.png"
    )
    plot_feature_importance(
        best["pipeline"], FEATURES,
        save_path=f"{OUTPUTS}/04_feature_importance.png"
    )

    # ── 5. Example prediction ─────────────────────────────────
    print("\n🏠 STEP 5 — Predict a New House")
    example = {
        "Rooms": 3, "Distance": 10, "Bedroom2": 3, "Bathroom": 1,
        "Car": 1, "Landsize": 500, "BuildingArea": 120,
        "YearBuilt": 1990, "Propertycount": 4000,
    }
    print("  Input features:")
    for k, v in example.items():
        print(f"    {k:<20}: {v}")

    price = predict(best["pipeline"], example)
    print(f"\n  💰 Estimated price ({name}): ${price:,.0f} AUD")

    print(f"\n✅ All charts saved to {OUTPUTS}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melbourne House Price Prediction")
    parser.add_argument("--data", default="data/melb_data.csv",
                        help="Path to melb_data.csv")
    args = parser.parse_args()
    main(data_path=args.data)
