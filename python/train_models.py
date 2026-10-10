import pandas as pd
import numpy as np
import os
import joblib

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
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# HEALTHCARE PATIENT ANALYTICS
# IMPROVED READMISSION PREDICTION

INPUT_FILE = "output/ml_dataset.csv"
MODEL_PATH = "output/models"

os.makedirs(MODEL_PATH, exist_ok=True)

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - IMPROVED MODEL TRAINING")
print("=" * 70)


# 1. LOAD DATA

print("\nLoading ML dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 2. SEPARATE FEATURES AND TARGET

print("\nSeparating features and target...")

X = df.drop(columns=["readmitted_30_days"])
y = df["readmitted_30_days"]

print("Features:", X.shape[1])
print("Target: readmitted_30_days")


# 3. TRAIN / VALIDATION / TEST SPLIT

print("\nCreating train, validation and test datasets...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))


# 4. FEATURE SCALING

print("\nScaling features for Logistic Regression...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODEL_PATH, "scaler.pkl")
)

print("Feature scaling completed.")


# 5. CALCULATE CLASS WEIGHT FOR XGBOOST

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass distribution in training data:")
print("Non-readmitted:", negative_count)
print("Readmitted:", positive_count)

print(
    "XGBoost scale_pos_weight:",
    round(scale_pos_weight, 2)
)


# 6. CREATE MODELS

print("\nCreating improved models...")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("Models created successfully.")


# 7. TRAIN MODELS

print("\n" + "-" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("-" * 70)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression training completed.")


print("\n" + "-" * 70)
print("TRAINING RANDOM FOREST")
print("-" * 70)

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


print("\n" + "-" * 70)
print("TRAINING XGBOOST")
print("-" * 70)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost training completed.")


# 8. GET VALIDATION PROBABILITIES

print("\nGenerating validation probabilities...")

logistic_validation_prob = logistic_model.predict_proba(
    X_validation_scaled
)[:, 1]

random_forest_validation_prob = random_forest_model.predict_proba(
    X_validation
)[:, 1]

xgb_validation_prob = xgb_model.predict_proba(
    X_validation
)[:, 1]

print("Validation probabilities generated.")


# 9. FIND BEST THRESHOLD

def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_f1 = 0

    for threshold in np.arange(0.10, 0.91, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


print("\nFinding optimal probability thresholds...")

logistic_threshold, logistic_validation_f1 = (
    find_best_threshold(
        y_validation,
        logistic_validation_prob
    )
)

random_forest_threshold, random_forest_validation_f1 = (
    find_best_threshold(
        y_validation,
        random_forest_validation_prob
    )
)

xgb_threshold, xgb_validation_f1 = (
    find_best_threshold(
        y_validation,
        xgb_validation_prob
    )
)

print(
    f"Logistic Regression threshold: "
    f"{logistic_threshold:.2f}"
)

print(
    f"Random Forest threshold: "
    f"{random_forest_threshold:.2f}"
)

print(
    f"XGBoost threshold: "
    f"{xgb_threshold:.2f}"
)


# 10. GET TEST PROBABILITIES

print("\nGenerating final test predictions...")

logistic_test_prob = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

random_forest_test_prob = random_forest_model.predict_proba(
    X_test
)[:, 1]

xgb_test_prob = xgb_model.predict_proba(
    X_test
)[:, 1]


# 11. APPLY VALIDATION THRESHOLDS TO TEST DATA

logistic_test_predictions = (
    logistic_test_prob >= logistic_threshold
).astype(int)

random_forest_test_predictions = (
    random_forest_test_prob >= random_forest_threshold
).astype(int)

xgb_test_predictions = (
    xgb_test_prob >= xgb_threshold
).astype(int)


# 12. MODEL EVALUATION FUNCTION

def evaluate_model(
    name,
    y_true,
    predictions,
    probabilities,
    threshold
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    }


# 13. EVALUATE MODELS

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_test_predictions,
        logistic_test_prob,
        logistic_threshold
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        random_forest_test_predictions,
        random_forest_test_prob,
        random_forest_threshold
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_test_predictions,
        xgb_test_prob,
        xgb_threshold
    )
)


# 14. MODEL COMPARISON

print("\n" + "=" * 70)
print("IMPROVED MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "output/model_comparison.csv",
    index=False
)


# 15. SELECT BEST MODEL

best_index = results_df["ROC_AUC"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_threshold = results_df.loc[
    best_index,
    "Threshold"
]

print("\nBest model based on ROC-AUC:")
print(best_model_name)

print(
    f"Selected threshold: {best_threshold:.2f}"
)


# 16. SAVE MODELS

print("\nSaving trained models...")

joblib.dump(
    logistic_model,
    os.path.join(
        MODEL_PATH,
        "logistic_regression.pkl"
    )
)

joblib.dump(
    random_forest_model,
    os.path.join(
        MODEL_PATH,
        "random_forest.pkl"
    )
)

joblib.dump(
    xgb_model,
    os.path.join(
        MODEL_PATH,
        "xgboost.pkl"
    )
)

print("Models saved successfully.")


# 17. SAVE SELECTED THRESHOLDS

thresholds = {
    "Logistic Regression": logistic_threshold,
    "Random Forest": random_forest_threshold,
    "XGBoost": xgb_threshold
}

joblib.dump(
    thresholds,
    os.path.join(
        MODEL_PATH,
        "thresholds.pkl"
    )
)


# 18. SAVE TEST PREDICTIONS

print("\nSaving test prediction results...")

prediction_output = X_test.copy()

prediction_output["actual_readmission"] = (
    y_test.values
)

prediction_output["logistic_probability"] = (
    logistic_test_prob
)

prediction_output["logistic_prediction"] = (
    logistic_test_predictions
)

prediction_output["random_forest_probability"] = (
    random_forest_test_prob
)

prediction_output["random_forest_prediction"] = (
    random_forest_test_predictions
)

prediction_output["xgboost_probability"] = (
    xgb_test_prob
)

prediction_output["xgboost_prediction"] = (
    xgb_test_predictions
)

prediction_output.to_csv(
    "output/model_predictions.csv",
    index=False
)

print("Prediction results saved.")


# 19. FINAL SUMMARY

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(" - output/model_comparison.csv")
print(" - output/model_predictions.csv")
print(" - output/models/scaler.pkl")
print(" - output/models/logistic_regression.pkl")
print(" - output/models/random_forest.pkl")
print(" - output/models/xgboost.pkl")
print(" - output/models/thresholds.pkl")

print("\nBest model:", best_model_name)
print(f"Selected threshold: {best_threshold:.2f}")

print("\nModel training finished.")