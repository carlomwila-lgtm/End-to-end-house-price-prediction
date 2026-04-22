[README.md](https://github.com/user-attachments/files/26966500/README.md)
# 🏠 Melbourne House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=flat&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=flat)

> A complete, end-to-end machine learning project predicting house sale prices in Melbourne, Australia.  
> Covers the full ML workflow: **data exploration → cleaning → feature engineering → model comparison → evaluation → prediction**.

---

## 📊 Results at a Glance

| Model | R² Score | MAE (AUD) | RMSE (AUD) |
|---|---|---|---|
| Linear Regression | 0.436 | $328,120 | $473,440 |
| Ridge Regression | 0.436 | $328,098 | $473,408 |
| Decision Tree | 0.511 | $296,693 | $440,836 |
| **Random Forest** ✅ | **0.731** | **$208,637** | **$326,911** |
| Gradient Boosting | 0.669 | $246,950 | $362,695 |

**Winner: Random Forest** — explains 73% of price variance with an average error of ~$209k.

---

## 📸 Visualizations

<table>
<tr>
<td><img src="outputs/01_price_distribution.png" width="400"/><br><sub>Price Distribution</sub></td>
<td><img src="outputs/02_model_comparison.png" width="400"/><br><sub>Model Comparison</sub></td>
</tr>
<tr>
<td><img src="outputs/03_actual_vs_predicted.png" width="400"/><br><sub>Actual vs Predicted</sub></td>
<td><img src="outputs/04_feature_importance.png" width="400"/><br><sub>Feature Importance</sub></td>
</tr>
</table>

---

## 📂 Project Structure

```
melbourne-house-prices/
│
├── 📓 notebooks/
│   └── house_price_prediction.ipynb   # Full walkthrough with explanations
│
├── 🐍 src/
│   ├── preprocessing.py               # Data loading, cleaning, splitting
│   ├── models.py                      # Model definitions and evaluation
│   └── visualization.py              # All plotting functions
│
├── 📁 data/
│   └── README.md                      # How to download the dataset
│
├── 📁 outputs/                        # Generated charts (auto-created)
│
├── main.py                            # Run everything from CLI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/carlomwila/melbourne-house-prices.git
cd melbourne-house-prices
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Get `melb_data.csv` from [Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) and place it in the `data/` folder.

### 4a. Run the full pipeline

```bash
python main.py
# or with a custom data path:
python main.py --data path/to/melb_data.csv
```

### 4b. Explore the Jupyter Notebook

```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

---

## 🔍 Key Findings

1. **Distance from CBD** is the single most important predictor of Melbourne house prices
2. **Ensemble models** (Random Forest, Gradient Boosting) dramatically outperform linear models
3. **Missing data** is the biggest data quality challenge — `BuildingArea` is missing in 47% of rows
4. Prices are **right-skewed** — a small number of luxury properties pull the average well above the median

---

## 🧠 ML Concepts Covered

| Concept | Where used |
|---------|-----------|
| Train / Test split | `src/preprocessing.py` |
| Sklearn Pipeline | `src/models.py` |
| Imputation (median fill) | `SimpleImputer` |
| Feature Scaling | `StandardScaler` |
| Cross-Validation (k=5) | `cross_val_score` |
| R², MAE, RMSE | `src/models.py` → `evaluate_model()` |
| Feature Importance | `src/visualization.py` → `plot_feature_importance()` |

---

## 💡 Possible Next Steps

- [ ] One-Hot Encode categorical features (`Suburb`, `Type`, `CouncilArea`)
- [ ] Log-transform `Price` to handle right skew
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Try **XGBoost** or **LightGBM** for a performance boost
- [ ] Add geospatial features using lat/long coordinates
- [ ] Deploy as a web app with **Streamlit** or **FastAPI**

---

## 📦 Dataset

- **Source**: [Melbourne Housing Snapshot — Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)
- **Size**: 13,580 properties × 21 features
- **Target variable**: `Price` (sale price in AUD)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation
- **scikit-learn** — ML models, pipelines, evaluation
- **matplotlib** — visualizations
- **Jupyter** — interactive notebook

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built as a portfolio project demonstrating end-to-end machine learning skills.*
