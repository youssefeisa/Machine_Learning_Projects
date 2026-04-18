# 🏠 House Price Prediction

A machine learning project that predicts residential house sale prices using the Ames Housing dataset. The model is built using **CatBoost**, a gradient boosting algorithm that handles categorical features natively without manual encoding.

---

## 📋 Project Overview

The goal of this project is to predict the final sale price of homes in Ames, Iowa based on 79 explanatory features describing almost every aspect of residential properties — from square footage and neighborhood to garage finish and kitchen quality.

This is based on the well-known [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

---

## 📁 Project Structure

```
├── House_Price_Prediction.ipynb   # Main notebook
├── train.csv                      # Training dataset (1,460 samples, 81 features)
├── test.csv                       # Test dataset for final predictions
├── data_description.txt           # Full feature descriptions
└── submission.csv                 # Output predictions file
```

---

## 📊 Dataset

- **Training set:** 1,460 houses × 81 columns (80 features + `SalePrice` target)
- **Target variable:** `SalePrice` — the sale price of the house in USD
- **Feature types:**
  - 35 numerical features (e.g., `GrLivArea`, `LotArea`, `YearBuilt`)
  - 43 categorical features (e.g., `Neighborhood`, `MSZoning`, `KitchenQual`)
  - 3 float features (e.g., `LotFrontage`, `GarageYrBlt`, `MasVnrArea`)

---

## 🔧 Data Preprocessing

### 1. Dropping High-Null Columns
Columns with more than **70% missing values** were dropped:
- `Alley`, `PoolQC`, `Fence`, `MiscFeature`

### 2. Missing Value Imputation
Each group of related features was handled with domain-aware logic:

| Feature Group | Strategy |
|---|---|
| `FireplaceQu` | Filled with `"No Fireplace"` |
| `LotFrontage` | Filled with **median per Neighborhood** (group-wise imputation) |
| `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | Filled with `"No Garage"` |
| `GarageYrBlt` | Filled with overall median |
| `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | Filled with `"No Basement"` |
| `MasVnrType` | Filled with `"No Masonry Veneer"` |
| `MasVnrArea` | Filled with `0` |
| `Electrical` | Filled with mode (most frequent value) |

### 3. Feature / Target Split
- **Target:** `SalePrice`
- **Features:** All remaining columns except `Id` and `SalePrice`

---

## 🤖 Model

**Algorithm:** `CatBoostRegressor`

CatBoost was chosen because it:
- Handles categorical features **natively** (no need for Label Encoding or One-Hot Encoding)
- Is robust to overfitting due to ordered boosting
- Delivers strong performance out of the box

### Hyperparameters

```python
CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    loss_function='RMSE',
    verbose=False
)
```

The model was trained on the **full training set** (not a train/val split) to maximize the data available for learning before generating test predictions.

---

## 📈 Results

The model was evaluated on the training data:

| Metric | Value |
|---|---|
| **RMSE** | $7,838 |
| **R²** | 0.9903 |

> ⚠️ Note: These metrics are computed on the training set (in-sample). For a real-world estimate of generalization, cross-validation should be used.

---

## 🔍 Top 20 Most Important Features

| Rank | Feature | Importance (%) |
|---|---|---|
| 1 | `OverallQual` | 21.71 |
| 2 | `GrLivArea` | 13.59 |
| 3 | `BsmtFinSF1` | 4.43 |
| 4 | `1stFlrSF` | 4.22 |
| 5 | `GarageCars` | 3.90 |
| 6 | `FireplaceQu` | 3.88 |
| 7 | `BsmtQual` | 3.71 |
| 8 | `TotalBsmtSF` | 3.66 |
| 9 | `LotArea` | 3.20 |
| 10 | `KitchenQual` | 2.62 |
| 11 | `GarageArea` | 1.97 |
| 12 | `YearBuilt` | 1.92 |
| 13 | `GarageType` | 1.70 |
| 14 | `Neighborhood` | 1.60 |
| 15 | `GarageYrBlt` | 1.56 |
| 16 | `LandContour` | 1.55 |
| 17 | `OverallCond` | 1.51 |
| 18 | `FullBath` | 1.51 |
| 19 | `2ndFlrSF` | 1.43 |
| 20 | `TotRmsAbvGrd` | 1.39 |

**Overall quality** and **living area** are by far the strongest predictors of sale price.

---

## 🚀 How to Run

1. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn
```

2. **Place the data files in the same directory as the notebook:**
   - `train.csv`
   - `test.csv`

3. **Run all cells** in `House_Price_Prediction.ipynb`

4. **Output:** A `submission.csv` file will be generated with predicted sale prices for the test set.

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Train/test split, metrics |
| `catboost` | Gradient boosting model |
| `matplotlib` / `seaborn` | Data visualization |

---

## 💡 Potential Improvements

- Add **cross-validation** (e.g., 5-fold CV) for a more honest evaluation
- Apply **log transformation** on `SalePrice` to reduce skewness
- Engineer new features (e.g., total bathrooms, house age, remodel age)
- Tune hyperparameters using **Optuna** or **GridSearchCV**
- Blend with other models (XGBoost, LightGBM) for an ensemble

---

## 📄 License

This project uses the Ames Housing dataset, originally compiled by Dean De Cock and made available through Kaggle for educational purposes.
