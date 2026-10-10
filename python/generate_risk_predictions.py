import pandas as pd
import joblib
import os


# HEALTHCARE PATIENT ANALYTICS
# READMISSION RISK PREDICTION

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - READMISSION RISK PREDICTION")
print("=" * 70)

# 1. File paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ML_DATA_FILE = os.path.join(
    BASE_DIR,
    "output",
    "improved_ml_dataset.csv"
)

ID_DATA_FILE = os.path.join(
    BASE_DIR,
    "output",
    "improved_prediction_dataset.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "output",
    "improved_logistic_regression.pkl"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "output",
    "improved_scaler.pkl"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "readmission_risk_predictions.csv"
)

# 2. Load ML dataset

print("\nLoading ML dataset...")

df = pd.read_csv(ML_DATA_FILE)

print("ML dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 3. Load original prediction dataset
# This file contains admission_id and patient_id.
# The ML dataset does not contain these identifiers because
# they were removed before model training.

print("\nLoading prediction dataset for admission and patient IDs...")

id_df = pd.read_csv(ID_DATA_FILE)

print("Prediction dataset loaded successfully.")
print("Rows:", len(id_df))
print("Columns:", len(id_df.columns))

# 4. Check row counts

if len(df) != len(id_df):
    raise ValueError(
        "Row count mismatch between ML dataset and prediction dataset."
    )


# 5. Separate target

target_column = "readmitted_30_days"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' was not found."
    )

actual_target = df[target_column]

X = df.drop(columns=[target_column])


# 6. Remove identifier columns if present

identifier_columns = [
    "admission_id",
    "patient_id"
]

for column in identifier_columns:
    if column in X.columns:
        X = X.drop(columns=[column])


# 7. Load trained model and scaler

print("\nLoading trained Logistic Regression model...")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

print("Model loaded successfully.")
print("Scaler loaded successfully.")


# 8. Handle missing values

numeric_columns = X.select_dtypes(
    include=["number"]
).columns

for column in numeric_columns:
    X[column] = X[column].fillna(
        X[column].median()
    )


# 9. Scale features

X_scaled = scaler.transform(X)


# 10. Generate probability scores

print("\nGenerating readmission risk scores...")

risk_scores = model.predict_proba(X_scaled)[:, 1]

risk_scores_percentage = risk_scores * 100


# 11. Define risk levels
# Project-defined analytical thresholds:
# Low    : below 30%
# Medium : 30% to below 60%
# High   : 60% or above
# These are NOT clinical diagnostic thresholds.

def classify_risk(score):

    if score < 30:
        return "Low"

    elif score < 60:
        return "Medium"

    else:
        return "High"


risk_levels = [
    classify_risk(score)
    for score in risk_scores_percentage
]


# 12. Create prediction output

prediction_df = pd.DataFrame()

# Get identifiers from the original prediction dataset
prediction_df["admission_id"] = id_df["admission_id"]

prediction_df["patient_id"] = id_df["patient_id"]

# Prediction date
prediction_df["prediction_date"] = pd.Timestamp.today().date()

# Risk score
prediction_df["risk_score"] = (
    risk_scores_percentage.round(2)
)

# Risk level
prediction_df["risk_level"] = risk_levels

# Actual outcome for validation
prediction_df["actual_readmission_30_days"] = (
    actual_target.values
)


# 13. Sort by highest risk

prediction_df = prediction_df.sort_values(
    by="risk_score",
    ascending=False
)


# 14. Save predictions

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nRisk prediction file created successfully.")

print("\nOutput file:")
print(OUTPUT_FILE)


# 15. Risk summary

print("\n" + "=" * 70)
print("RISK SUMMARY")
print("=" * 70)

print("\nRisk level distribution:")

print(
    prediction_df["risk_level"]
    .value_counts()
)

print("\nAverage risk score:")

print(
    round(
        prediction_df["risk_score"].mean(),
        2
    )
)

print("\nHighest risk score:")

print(
    round(
        prediction_df["risk_score"].max(),
        2
    )
)

print("\nLowest risk score:")

print(
    round(
        prediction_df["risk_score"].min(),
        2
    )
)


# 16. Top 10 highest-risk admissions

print("\n" + "=" * 70)
print("TOP 10 HIGHEST-RISK ADMISSIONS")
print("=" * 70)

print(
    prediction_df[
        [
            "admission_id",
            "patient_id",
            "risk_score",
            "risk_level"
        ]
    ].head(10).to_string(index=False)
)


# 17. Completion message

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)