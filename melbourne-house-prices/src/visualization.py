"""
visualization.py
----------------
All plotting functions for the Melbourne Housing Price Prediction project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLORS = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#f87171"]
BG     = "#0f172a"
PANEL  = "#1e293b"
TEXT   = "#e2e8f0"
ACCENT = "#f59e0b"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, fontsize=9)


def plot_model_comparison(summary_df, save_path=None):
    """Bar charts comparing R² and MAE across all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    models = summary_df["Model"].tolist()
    r2_vals  = summary_df["R2"].tolist()
    mae_vals = (summary_df["MAE"] / 1000).tolist()

    # R² chart
    bars = ax1.barh(models, r2_vals, color=COLORS[:len(models)], height=0.5, edgecolor="none")
    ax1.axvline(0.8, color=ACCENT, linestyle="--", linewidth=1.2, label="Good threshold (0.8)")
    for bar, val in zip(bars, r2_vals):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 1.05)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    _style_ax(ax1, title="Model Comparison — R² Score\n(higher = better)", xlabel="R² Score")

    # MAE chart
    bars2 = ax2.barh(models, mae_vals, color=COLORS[:len(models)], height=0.5, edgecolor="none")
    for bar, val in zip(bars2, mae_vals):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"${val:.0f}k", va="center", color=TEXT, fontsize=9, fontweight="bold")
    _style_ax(ax2, title="Model Comparison — MAE\n(lower = better)", xlabel="Mean Absolute Error (k$)")

    plt.tight_layout(pad=2)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, model_name, save_path=None):
    """Scatter plot of actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)

    sample = min(600, len(y_test))
    idx = np.random.choice(len(y_test), sample, replace=False)
    yt = np.array(y_test)[idx] / 1e6
    yp = y_pred[idx] / 1e6

    ax.scatter(yt, yp, alpha=0.35, s=14, color="#38bdf8", label="Houses")
    max_val = max(yt.max(), yp.max())
    ax.plot([0, max_val], [0, max_val], color=ACCENT, linewidth=1.5, label="Perfect prediction")

    _style_ax(ax,
        title=f"Actual vs Predicted Prices — {model_name}\n(points on the line = perfect)",
        xlabel="Actual Price (M$)",
        ylabel="Predicted Price (M$)"
    )
    ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.close()


def plot_feature_importance(pipeline, feature_names, save_path=None):
    """Horizontal bar chart of feature importances (tree-based models)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print("  ⚠️  This model does not expose feature importances.")
        return

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    ax.barh([feature_names[i] for i in sorted_idx],
            importances[sorted_idx], color="#818cf8", edgecolor="none")
    _style_ax(ax,
        title="Feature Importance\n(which variables drive price the most?)",
        xlabel="Importance"
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.close()


def plot_price_distribution(y, save_path=None):
    """Histogram of house prices."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG)
    ax.hist(y / 1e6, bins=60, color="#38bdf8", edgecolor="none", alpha=0.85)
    ax.axvline(y.median() / 1e6, color=ACCENT, linewidth=1.5,
               label=f"Median: ${y.median()/1e6:.2f}M")
    _style_ax(ax,
        title="Distribution of House Prices — Melbourne",
        xlabel="Price (M$)",
        ylabel="Count"
    )
    ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.close()
