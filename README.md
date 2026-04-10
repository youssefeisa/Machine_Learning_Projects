# Depression Prediction — ML Pipeline

A machine learning project to predict depression among working professionals and students using survey data. The pipeline covers the full lifecycle from raw data exploration through feature engineering, model selection, hyperparameter tuning, threshold optimization, and semi-supervised learning with pseudo-labeling.

---

## Dataset

| Split | Rows | Columns |
|-------|------|---------|
| Train | 140,700 | 20 |
| Test | 93,800 | 19 |

**Target variable:** `Depression` (binary: 0 = No, 1 = Yes)  
**Class distribution:** ~81.8% negative (0), ~18.2% positive (1) — imbalanced dataset.

**Key features:** Gender, Age, City, Working Professional or Student, Profession, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Sleep Duration, Dietary Habits, Degree, Suicidal Thoughts, Work/Study Hours, Financial Stress, Family History of Mental Illness.

---

## Pipeline Steps

### Step 1 — Exploratory Data Analysis

Loaded the training CSV and performed an initial inspection:

- Printed shape, column names, and dtypes.
- Identified missing values per column. Key findings:
  - `Profession`: 36,630 nulls
  - `Academic Pressure`, `CGPA`, `Study Satisfaction`: ~112,800 nulls each (mostly workers who have no academic fields)
  - `Work Pressure`, `Job Satisfaction`: ~27,900 nulls each (mostly students)
  - `Dietary Habits`, `Financial Stress`, `Degree`: small number of nulls (≤ 4)
- Confirmed the dataset contains both **Working Professionals** (112,799) and **Students** (27,901).

---

### Step 2 — Missing Value Imputation

Recognized that the high null counts in pressure/satisfaction columns are structurally caused by the two population types (workers vs. students), not random missingness.

**Strategy:**

- Grouped by `Working Professional or Student` and filled numeric columns with the **group median**:
  - `Academic Pressure`, `Study Satisfaction`, `CGPA` → imputed for workers (these fields don't apply to them).
  - `Work Pressure`, `Job Satisfaction` → imputed for students.
- `Dietary Habits` → filled with global mode.
- `Financial Stress` → filled with global median.
- `Degree` → filled with global mode.

After imputation: **zero nulls** across all columns except `Profession`.

---

### Step 3 — Profession Cleaning & Grouping

`Profession` had 36,630 nulls and 64 unique values with noisy/incorrect entries.

**Actions taken:**

1. Students with a null profession were assigned the label `"Student"`.
2. Remaining nulls (working professionals with no profession listed) were labeled `"Unknown"`.
3. All 64+ profession values were mapped into 10 semantic groups:

| Group | Examples |
|-------|---------|
| Education | Teacher, Researcher, PhD, Academic |
| Creative | Content Writer, Architect, Chef, Pilot, UX/UI Designer |
| Business | Consultant, HR Manager, Entrepreneur, Financial Analyst |
| Medical | Doctor, Pharmacist, MBBS |
| Legal | Lawyer, Judge |
| Tech | Data Scientist, Software Engineer, Civil Engineer |
| Trade | Chemist, Electrician, Plumber, Customer Support |
| Travel | Travel Consultant |
| Student | Student |
| Unknown / Other | Unknown, unrecognized entries |

A second-pass fix mapped edge cases (`Medical Doctor`, `MBBS`, `PhD`, `Academic`, `Analyst`, etc.) to the correct groups, reducing the `Other` bucket to 31 records out of 140,700.

---

### Step 4 — Categorical Data Cleaning

Inspected `Sleep Duration` and `Dietary Habits` for dirty values. Both columns contained corrupted entries (city names, header strings, numeric values) mixed in with valid categories.

**Valid categories defined:**

- `Sleep Duration`: `Less than 5 hours`, `5-6 hours`, `7-8 hours`, `More than 8 hours`
- `Dietary Habits`: `Healthy`, `Moderate`, `Unhealthy`

All out-of-vocabulary values were set to `None`, then filled with the column mode. The same cleaning was applied to the **test set** using mode values learned from the training set.

---

### Step 5 — Preprocessing Pipeline (Reusable Function)

Refactored all preprocessing steps (profession grouping, numeric imputation, categorical encoding, sleep/diet cleaning) into a single `preprocess(train, test)` function.

The function also applied `OrdinalEncoder` to encode all categorical columns, producing a fully numeric feature matrix. Processed shapes:

- Train: `(140,700, 27)`
- Test: `(93,800, 26)`

---

### Step 6 — Baseline Modeling

Established performance baselines using 5-fold cross-validation (F1 score):

| Model | CV F1 | Std |
|-------|-------|-----|
| Dummy Classifier (most frequent) | 0.0000 | ±0.0000 |
| Logistic Regression | 0.8254 | ±0.0038 |
| Random Forest (100 trees) | 0.8189 | ±0.0026 |
| **LightGBM** | **0.8279** | **±0.0014** |
| XGBoost | 0.8248 | ±0.0022 |

**LightGBM** showed the best F1 with lowest variance — selected as the primary model.

---

### Step 7 — Feature Importance Analysis

Trained LightGBM on the full training set and plotted the top 15 features by importance. This informed the feature engineering step by revealing which combinations of stress and satisfaction signals drive predictions.

---

### Step 8 — Feature Engineering

Created 5 new derived features:

| Feature | Formula |
|---------|---------|
| `Total_Stress` | Mean of Work Pressure, Financial Stress, Academic Pressure |
| `Total_Satisfaction` | Mean of Job Satisfaction, Study Satisfaction |
| `Stress_Satisfaction_Ratio` | `(Total_Stress + 1) / (Total_Satisfaction + 1)` |
| `WorkPressure_Sleep` | Work Pressure × Sleep Duration (encoded) |
| `WorkHours_Sleep` | Work/Study Hours × Sleep Duration (encoded) |

LightGBM with new features: **F1 = 0.8287** (↑ from 0.8279).

---

### Step 9 — Hyperparameter Tuning with Optuna (Round 1)

Used Optuna with 50 trials to optimize LightGBM hyperparameters on 5-fold CV F1.

**Search space:** `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`.

**Best result:** F1 = **0.8303**

```
n_estimators=269, learning_rate=0.0676, num_leaves=117,
max_depth=5, min_child_samples=15, subsample=0.824, colsample_bytree=0.519
```

Generated **submission.csv** (first submission).

---

### Step 10 — Stratified K-Fold Validation

Switched from standard KFold to `StratifiedKFold` (5 splits) to ensure balanced class distribution per fold, given the ~18% minority class.

Stratified KFold F1: **0.8299 ± 0.0041**

---

### Step 11 — Decision Threshold Optimization

Instead of using the default 0.5 threshold, swept thresholds from 0.1 to 0.9 using out-of-fold probabilities from `cross_val_predict`.

| Threshold | F1 |
|-----------|----|
| 0.50 (default) | 0.8299 |
| **0.38 (optimal)** | **0.8338** |

The model is better calibrated by lowering the threshold due to class imbalance.

---

### Step 12 — Joint Hyperparameter + Threshold Tuning (Round 2)

Combined model hyperparameter search and threshold search into a single Optuna study with 100 trials.

**Best result:** F1 = **0.8342**

```
n_estimators=650, learning_rate=0.0182, num_leaves=113, max_depth=7,
min_child_samples=49, subsample=0.990, colsample_bytree=0.504, threshold=0.400
```

Generated **submission_v2.csv**.

---

### Step 13 — Pseudo-Labeling (Semi-Supervised Learning)

Leveraged high-confidence test predictions to augment the training data.

**Approach:**

1. Predicted probabilities on the test set using the best model.
2. Kept only samples where the model was highly confident:
   - Positive (depressed): probability ≥ 0.85
   - Negative: probability ≤ 0.15
3. **80,610 out of 93,800** test samples were confident enough to use.
4. Appended these pseudo-labeled samples to the training set.

| Dataset | Size |
|---------|------|
| Original training | 140,700 |
| After pseudo-labeling | 221,310 |

Retrained LightGBM on the combined dataset and generated **submission_v3.csv**.

---

## Submission Summary

| Version | Description | CV F1 |
|---------|-------------|-------|
| `submission.csv` | LightGBM + feature engineering + Optuna round 1 | 0.8303 |
| `submission_v2.csv` | LightGBM + joint hyperparameter & threshold tuning | 0.8342 |
| `submission_v3.csv` | v2 model + pseudo-labeling on confident test samples | — |

---

## Tech Stack

- **Python** (pandas, numpy, matplotlib, seaborn)
- **scikit-learn** — preprocessing, cross-validation, metrics
- **LightGBM** — primary classifier
- **XGBoost**, **Random Forest** — baseline comparison
- **Optuna** — hyperparameter optimization
- **OrdinalEncoder** — categorical encoding

---

## Project Structure

```
├── train.csv                # Raw training data (140,700 rows)
├── test.csv                 # Raw test data (93,800 rows)
├── sample_submission.csv    # Submission format
├── Playground.ipynb         # Main notebook (all steps)
├── submission.csv           # Submission v1
├── submission_v2.csv        # Submission v2 (best tuning)
├── submission_v3.csv        # Submission v3 (pseudo-labels)
└── README.md                # This file
```
