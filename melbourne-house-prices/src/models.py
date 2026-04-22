"""
models.py
---------
Model definitions, training, cross-validation, and evaluation
for the Melbourne Housing Price Prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_pipeline(model) -> Pipeline:
    """
    Wrap any sklearn estimator in a preprocessing + model pipeline.
    Steps:
        1. SimpleImputer  — fill missing values with the column median
        2. StandardScaler — normalize features to zero mean / unit variance
        3. model          — the regressor
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def get_models() -> dict:
    """Return a dictionary of named model pipelines to compare."""
    return {
        "Linear Regression":  build_pipeline(LinearRegression()),
        "Ridge":              build_pipeline(Ridge(alpha=10)),
        "Decision Tree":      build_pipeline(DecisionTreeRegressor(max_depth=6, random_state=42)),
        "Random Forest":      build_pipeline(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        "Gradient Boosting":  build_pipeline(GradientBoostingRegressor(n_estimators=100, random_state=42)),
    }


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, cv: int = 5) -> dict:
    """
    Train a pipeline and return a dictionary of evaluation metrics.

    Metrics
    -------
    mae   : Mean Absolute Error       — average prediction error in $
    rmse  : Root Mean Squared Error   — penalises large errors more
    r2    : R² score on test set      — proportion of variance explained
    cv_r2 : Mean R² over k-fold CV    — how well the model generalises
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    cv_r2 = cv_scores.mean()

    return {
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "mae":      mae,
        "rmse":     rmse,
        "r2":       r2,
        "cv_r2":    cv_r2,
    }


def train_all(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train and evaluate every model. Returns a summary DataFrame
    and a results dict (with pipelines + predictions).
    """
    results = {}
    rows    = []

    print(f"\n{'Model':<22} {'MAE ($)':>12} {'RMSE ($)':>12} {'R²':>8} {'CV R²':>8}")
    print("-" * 68)

    for name, pipeline in models.items():
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        results[name] = metrics
        rows.append({
            "Model":  name,
            "MAE":    metrics["mae"],
            "RMSE":   metrics["rmse"],
            "R2":     metrics["r2"],
            "CV_R2":  metrics["cv_r2"],
        })
        print(f"{name:<22} {metrics['mae']:>12,.0f} {metrics['rmse']:>12,.0f} "
              f"{metrics['r2']:>8.3f} {metrics['cv_r2']:>8.3f}")

    summary = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return summary, results


def best_model(results: dict) -> tuple[str, dict]:
    """Return the name and metrics dict of the highest R² model."""
    name = max(results, key=lambda k: results[k]["r2"])
    return name, results[name]


def predict(pipeline, features: dict) -> float:
    """
    Predict the price of a single house.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    features : dict of feature_name -> value

    Returns
    -------
    Predicted price in AUD
    """
    import pandas as pd
    df = pd.DataFrame([features])
    return pipeline.predict(df)[0]
