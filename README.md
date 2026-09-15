# Student Test Score Prediction

A regression pipeline predicting exam scores from student lifestyle and academic features (study hours, sleep, class attendance, study method, etc.), built for a Kaggle-style competition dataset.

## Overview

This notebook covers a full, straightforward ML workflow:

1. **Exploratory Data Analysis (EDA)** — missing values check, duplicate check, distribution plots (age, exam score), categorical breakdowns (gender, study method), and correlation analysis between numerical features and the target.
2. **Preprocessing**
   - Ordinal mapping for ordered categorical features: `internet_access`, `sleep_quality`, `facility_rating`, `exam_difficulty`.
   - One-Hot Encoding (via `ColumnTransformer`) for the remaining categorical features.
3. **Modeling** — an `XGBRegressor` wrapped in a scikit-learn `Pipeline`, trained on a single train/validation split (`train_test_split`, `random_state=0`).
4. **Evaluation** — RMSE on the held-out validation split.
5. **Inference** — predictions on the test set, exported as a Kaggle-style submission file.

## Result

| Metric | Value |
|---|---|
| Validation RMSE (single split) | **8.7638** |

## Model configuration

```python
XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    eval_metric='rmse'
)
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Notes / possible next steps

- Current evaluation uses a single train/validation split, which can be noisy. Switching to k-fold cross-validation would give a more robust estimate of generalization performance.
- No target encoding or interaction features are used here — the model relies on the raw features plus one-hot/ordinal encoding, which keeps the pipeline simple and easy to maintain.
