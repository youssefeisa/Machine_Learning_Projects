# 🌧️ Rain Prediction in Australia — Machine Learning Project

## 📌 Project Overview

This project aims to answer one simple question: **Will it rain tomorrow?**

Using real weather data from Australia (145,460 rows and 23 columns), three machine learning models were built, trained, and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

---

## 📊 Dataset

**Dataset:** `weatherAUS.csv`  
**Source:** Kaggle — Weather in Australia

| Detail | Value |
|---|---|
| Total Rows | 145,460 |
| Total Columns | 23 |
| Target Column | `RainTomorrow` (Yes / No) |

### Key Features:
- Temperature: `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`
- Wind: `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`
- Humidity & Pressure: `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`
- Other: `Rainfall`, `Evaporation`, `Sunshine`, `Cloud9am`, `Cloud3pm`
- Categorical: `Location`, `WindGustDir`, `RainToday`

---

## 🛠️ Preprocessing Pipeline

### 1. Data Splitting
Data was split based on year (time-based split to avoid data leakage):
- **Training set:** Before 2015 → 98,988 rows
- **Validation set:** Year 2015 → 17,231 rows
- **Test set:** After 2015 → 25,974 rows

### 2. Handling Missing Values
- `SimpleImputer(strategy='mean')` applied to all numeric columns
- Categorical missing values filled with `'Unknown'`

### 3. Feature Scaling
- `MinMaxScaler` applied to numeric columns — all values scaled to range [0, 1]

### 4. Categorical Encoding
- `OneHotEncoder` applied to categorical columns: `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`

---

## 🧪 Models & Results

### 1. Logistic Regression
A simple and fast baseline model using `liblinear` solver. All pipeline components (model, imputer, scaler, encoder) were saved to `aussie_rain.joblib` for future use.

**Top influential features:** Humidity3pm, Pressure3pm, Rainfall, WindGustSpeed

---

### 2. Decision Tree Classifier

#### Problem — Overfitting:
Training without depth constraints resulted in:
- **Training Accuracy: ~100%** ← Clear overfitting!
- The tree memorized the training data instead of learning general patterns.

#### Solution — Limiting `max_depth`:
Values from 1 to 20 were tested for `max_depth`, comparing training vs. validation error at each step.

**Best result at `max_depth=8`:**

| Dataset | Accuracy |
|---|---|
| Training | 85.2% |
| Validation | 84.2% |

---

### 3. Random Forest Classifier

Random Forest builds 100 decision trees and combines their predictions — significantly better generalization!

| Dataset | Accuracy |
|---|---|
| Training | ~99.99% |
| Validation | **85.7%** |

#### Hyperparameter Tuning Experiments:

| Configuration | Training Acc | Validation Acc |
|---|---|---|
| Default (100 trees) | 99.99% | 85.7% |
| n_estimators=10 | 98.7% | 84.9% |
| n_estimators=500 | 99.99% | 85.8% |
| max_depth=5 | 82.0% | 82.4% |
| max_depth=25 | 97.8% | 85.6% |
| max_leaf_nodes=2^20 | 99.99% | 85.7% |

---

## 🏆 Feature Importance

### Decision Tree:
| Feature | Importance |
|---|---|
| Humidity3pm | 26.1% |
| Pressure3pm | 6.2% |
| Rainfall | 5.9% |
| WindGustSpeed | 5.6% |
| Sunshine | 4.9% |

### Random Forest:
| Feature | Importance |
|---|---|
| Humidity3pm | 14.0% |
| Sunshine | 5.4% |
| Pressure3pm | 5.3% |
| Humidity9am | 5.0% |
| Rainfall | 4.8% |

> **Key Insight:** Afternoon humidity (`Humidity3pm`) is the single most important predictor of rain across all three models.

---

## 📈 Model Comparison

| Model | Validation Accuracy |
|---|---|
| Logistic Regression | ~84–85% |
| Decision Tree (max_depth=8) | 84.2% |
| **Random Forest (100 trees)** | **85.7%** ✅ |

**Winner: Random Forest** — achieves the highest accuracy and handles overfitting far better than a single decision tree.

---

## 📁 Project Structure

```
├── weatherAUS.csv                            # Raw dataset
├── Raining_Prediction_LogReg.ipynb           # Logistic Regression notebook
├── Raining_Prediction_DTree_RanForest.ipynb  # Decision Tree & Random Forest notebook
├── aussie_rain.joblib                        # Saved model pipeline
└── README.md
```

---

## 🚀 Loading the Saved Model

```python
import joblib

# Load the saved pipeline
aussie_rain = joblib.load('aussie_rain.joblib')

# Run predictions
model = aussie_rain['model']
predictions = model.predict(X_test)
```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
  ├── SimpleImputer
  ├── MinMaxScaler
  ├── OneHotEncoder
  ├── LogisticRegression
  ├── DecisionTreeClassifier
  ├── RandomForestClassifier
  └── accuracy_score, confusion_matrix
joblib
```
