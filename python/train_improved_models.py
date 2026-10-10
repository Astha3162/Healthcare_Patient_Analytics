import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# HEALTHCARE PATIENT ANALYTICS
# IMPROVED MODEL TRAINING

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "improved_ml_dataset.csv"

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - MODEL TRAINING")
print("=" * 70)


# 1. LOAD DATA

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)


# 2. SEPARATE FEATURES AND TARGET

target_column = "readmitted_30_days"

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


print("\nFeatures:", X.shape[1])
print("Target:", target_column)

print("\nTarget distribution:")
print(y.value_counts())


# 3. TRAIN / VALIDATION / TEST SPLIT

# First split:
# 80% temporary training data
# 20% final test data

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Second split:
# 75% training
# 25% validation
#
# Final proportions:
# Training   = 60%
# Validation = 20%
# Test       = 20%

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("\nTraining samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples      :", len(X_test))


# 4. FEATURE SCALING FOR LOGISTIC REGRESSION

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# 5. HANDLE CLASS IMBALANCE

positive_count = y_train.sum()
negative_count = len(y_train) - positive_count

scale_pos_weight = (
    negative_count / positive_count
)

print("\nPositive training records:", positive_count)
print("Negative training records:", negative_count)
print(
    "XGBoost scale_pos_weight:",
    round(scale_pos_weight, 2)
)


# 6. CREATE MODELS

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgboost_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# 7. TRAIN LOGISTIC REGRESSION

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression training completed.")


# 8. TRAIN RANDOM FOREST

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# 9. TRAIN XGBOOST

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgboost_model.fit(
    X_train,
    y_train
)

print("XGBoost training completed.")


# 10. MODEL EVALUATION FUNCTION

def evaluate_model(
    model_name,
    model,
    X_eval,
    y_eval
):

    predictions = model.predict(X_eval)

    probabilities = model.predict_proba(
        X_eval
    )[:, 1]

    accuracy = accuracy_score(
        y_eval,
        predictions
    )

    precision = precision_score(
        y_eval,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_eval,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_eval,
        probabilities
    )

    cm = confusion_matrix(
        y_eval,
        predictions
    )

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Confusion_Matrix": str(cm.tolist())
    }


# 11. VALIDATION RESULTS

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

validation_results = []

validation_results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_val_scaled,
        y_val
    )
)

validation_results.append(
    evaluate_model(
        "Random Forest",
        random_forest_model,
        X_val,
        y_val
    )
)

validation_results.append(
    evaluate_model(
        "XGBoost",
        xgboost_model,
        X_val,
        y_val
    )
)

validation_df = pd.DataFrame(
    validation_results
)

print(
    validation_df.to_string(index=False)
)


# 12. TEST RESULTS

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

test_results = []

test_results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test_scaled,
        y_test
    )
)

test_results.append(
    evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test
    )
)

test_results.append(
    evaluate_model(
        "XGBoost",
        xgboost_model,
        X_test,
        y_test
    )
)

test_df = pd.DataFrame(
    test_results
)

print(
    test_df.to_string(index=False)
)


# 13. SAVE RESULTS

validation_file = (
    OUTPUT_DIR / "improved_validation_results.csv"
)

test_file = (
    OUTPUT_DIR / "improved_test_results.csv"
)

validation_df.to_csv(
    validation_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# 14. SAVE MODELS

joblib.dump(
    logistic_model,
    OUTPUT_DIR / "improved_logistic_regression.pkl"
)

joblib.dump(
    random_forest_model,
    OUTPUT_DIR / "improved_random_forest.pkl"
)

joblib.dump(
    xgboost_model,
    OUTPUT_DIR / "improved_xgboost.pkl"
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "improved_scaler.pkl"
)


# 15. FEATURE IMPORTANCE

feature_names = X.columns.tolist()

# Random Forest importance

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": random_forest_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)

rf_importance.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False
)


# XGBoost importance

xgb_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgboost_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    "Importance",
    ascending=False
)

xgb_importance.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance.csv",
    index=False
)


# 16. FINAL MESSAGE

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nSaved files:")

print("- improved_validation_results.csv")
print("- improved_test_results.csv")
print("- improved_logistic_regression.pkl")
print("- improved_random_forest.pkl")
print("- improved_xgboost.pkl")
print("- improved_scaler.pkl")
print("- random_forest_feature_importance.csv")
print("- xgboost_feature_importance.csv")

print("\nThe three required models have been trained and evaluated.")

print("=" * 70)