"""
train.py
────────
Trains and evaluates 5 regression models on the Melbourne housing dataset.
Saves the comparison chart to outputs/model_comparison.png.

Usage:
    python src/train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "melb_data.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_comparison.png")

FEATURES = [
    "Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
    "Landsize", "BuildingArea", "YearBuilt", "Propertycount"
]
TARGET   = "Price"
SEED     = 42

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple:
    """Load the dataset and return feature matrix X and target vector y."""
    df = pd.read_csv(path)
    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    X  = df[FEATURES]
    y  = df[TARGET]
    print(f"✅ Data loaded: {len(df):,} rows × {len(FEATURES)} features")
    return X, y

# ── Model Definitions ──────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a dict of named sklearn pipelines."""
    def pipeline(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   model),
        ])

    return {
        "Linear Regression":  pipeline(LinearRegression()),
        "Ridge":              pipeline(Ridge(alpha=10)),
        "Decision Tree":      pipeline(DecisionTreeRegressor(max_depth=6, random_state=SEED)),
        "Random Forest":      pipeline(RandomForestRegressor(n_estimators=100, random_state=SEED)),
        "Gradient Boosting":  pipeline(GradientBoostingRegressor(n_estimators=100, random_state=SEED)),
    }

# ── Training & Evaluation ──────────────────────────────────────────────────────

def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train each model and compute MAE, RMSE, R², CV R²."""
    results = {}

    print(f"\n{'Model':<22} {'MAE ($)':>12} {'RMSE ($)':>12} {'R²':>8} {'CV R²':>8}")
    print("─" * 68)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

        results[name] = dict(pipe=pipe, mae=mae, rmse=rmse, r2=r2, cv_r2=cv_r2, y_pred=y_pred)
        print(f"{name:<22} {mae:>12,.0f} {rmse:>12,.0f} {r2:>8.3f} {cv_r2:>8.3f}")

    return results

# ── Visualization ──────────────────────────────────────────────────────────────

def plot_results(results: dict, y_test, features: list, output_path: str):
    """Generate and save the 4-panel comparison chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    COLORS = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#f87171"]
    DARK, LIGHT = "#1e293b", "#e2e8f0"

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f172a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    names    = list(results.keys())
    r2_vals  = [v["r2"]      for v in results.values()]
    mae_vals = [v["mae"]/1e3 for v in results.values()]

    # Panel 1 — R² comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(DARK)
    bars = ax1.barh(names, r2_vals, color=COLORS, height=0.55, edgecolor="none")
    ax1.axvline(0.8, color="#f59e0b", linestyle="--", linewidth=1.2, label="Good threshold (0.8)")
    for bar, val in zip(bars, r2_vals):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=LIGHT, fontsize=9, fontweight="bold")
    ax1.set_xlabel("R² Score", color=LIGHT)
    ax1.set_title("Model Comparison (R²)", color=LIGHT, fontsize=12, pad=10)
    ax1.tick_params(colors=LIGHT)
    ax1.spines[["top","right","bottom","left"]].set_visible(False)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=LIGHT)
    ax1.set_xlim(0, 1.05)

    # Panel 2 — MAE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(DARK)
    bars2 = ax2.bar(names, mae_vals, color=COLORS, width=0.55, edgecolor="none")
    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1,
                 f"{val:.0f}k", ha="center", color=LIGHT, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error (k$)", color=LIGHT)
    ax2.set_title("MAE — Lower is Better", color=LIGHT, fontsize=12, pad=10)
    ax2.tick_params(colors=LIGHT, axis="y")
    ax2.tick_params(colors=LIGHT, axis="x", rotation=15)
    ax2.spines[["top","right","bottom","left"]].set_visible(False)

    # Panel 3 — Predicted vs Actual (best model)
    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(DARK)
    idx = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    y_test_arr = np.array(y_test)
    ax3.scatter(y_test_arr[idx]/1e6, best["y_pred"][idx]/1e6,
                alpha=0.4, s=12, color="#38bdf8", label="Houses")
    max_val = max(y_test_arr.max(), best["y_pred"].max()) / 1e6
    ax3.plot([0, max_val], [0, max_val], color="#f59e0b", linewidth=1.5, label="Perfect prediction")
    ax3.set_xlabel("Actual price (M$)", color=LIGHT)
    ax3.set_ylabel("Predicted price (M$)", color=LIGHT)
    ax3.set_title(f"Actual vs Predicted — {best_name}\n→ Points on the line = perfect",
                  color=LIGHT, fontsize=12, pad=10)
    ax3.tick_params(colors=LIGHT)
    ax3.spines[["top","right","bottom","left"]].set_visible(False)
    ax3.legend(fontsize=8, facecolor=DARK, labelcolor=LIGHT)

    # Panel 4 — Feature importance (Random Forest)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(DARK)
    rf_pipe = results["Random Forest"]["pipe"]
    importances = rf_pipe.named_steps["model"].feature_importances_
    sorted_idx  = np.argsort(importances)
    ax4.barh([features[i] for i in sorted_idx], importances[sorted_idx],
             color="#818cf8", edgecolor="none")
    ax4.set_xlabel("Importance", color=LIGHT)
    ax4.set_title("Feature Importance (Random Forest)", color=LIGHT, fontsize=12, pad=10)
    ax4.tick_params(colors=LIGHT)
    ax4.spines[["top","right","bottom","left"]].set_visible(False)

    fig.suptitle("🏠 Melbourne House Price Prediction — Model Comparison",
                 fontsize=15, color="white", fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Chart saved → {output_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  MELBOURNE HOUSE PRICE PREDICTION — Training Pipeline")
    print("=" * 68)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"   Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    models  = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]
    print(f"\n🏆 Best model : {best_name}")
    print(f"   R²   = {best['r2']:.3f}")
    print(f"   MAE  = ${best['mae']:,.0f}")
    print(f"   RMSE = ${best['rmse']:,.0f}")

    plot_results(results, y_test, FEATURES, OUTPUT_PATH)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
