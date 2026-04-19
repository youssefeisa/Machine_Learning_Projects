# 🚕 Taxi Fare Prediction

A machine learning project to predict NYC taxi fares using ride pickup/dropoff coordinates, timestamps, and passenger count.

---

## 📦 Data Loading

The training dataset is large, so instead of loading it all at once, it's read in **chunks of 1M rows** and a **1% random sample** is taken from each chunk — resulting in a manageable but representative dataset. Only the relevant columns are loaded with optimized dtypes to save memory.

---

## 🔍 Exploration

Basic EDA is done on both train and test sets using `.info()`, `.describe()`, and checking the date range of `pickup_datetime`.

---

## ⚙️ Feature Engineering

Raw coordinates and timestamps aren't enough on their own, so several features are added:

- **Date parts**: Year, month, day, weekday, and hour are extracted from `pickup_datetime`.
- **Trip distance**: Calculated using the **Haversine formula** (great-circle distance between pickup and dropoff).
- **Landmark distances**: Distance from the dropoff point to 5 NYC landmarks — JFK, LaGuardia (LGA), Newark (EWR), the Met Museum, and the World Trade Center.

---

## 🧹 Cleaning & Outlier Removal

Rows with invalid or extreme values are filtered out — unrealistic fares, coordinates outside the NYC area, and passenger counts outside the 1–6 range.

A **correlation heatmap** is then plotted to understand feature relationships. The cleaned data is saved as `.parquet` for faster loading later.

---

## 🤖 Model Training

A **baseline Linear Regression** was tried first — it gave a terrible RMSE close to the max fare value, confirming the relationship is non-linear.

Three models are then trained and evaluated using RMSE on both train and validation sets:

| Model | Notes |
|---|---|
| **Ridge Regression** | Linear baseline with regularization |
| **Random Forest** | `max_depth=10`, 50 estimators |
| **XGBoost** | Gradient boosting with squared error objective |

---

## 📤 Submission

Predictions from the best models are saved to CSV files formatted for Kaggle submission.
