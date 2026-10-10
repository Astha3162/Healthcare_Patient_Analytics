import pandas as pd
import os


# HEALTHCARE PATIENT ANALYTICS
# ML DATA PREPARATION

INPUT_FILE = "output/admission_analytics.csv"
OUTPUT_FILE = "output/ml_dataset.csv"

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - ML DATA PREPARATION")
print("=" * 70)


# 1. LOAD ANALYTICAL DATASET

print("\nLoading analytical dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 2. SELECT PREDICTION FEATURES

print("\nSelecting machine learning features...")

features = [
    "patient_id",
    "doctor_id",
    "length_of_stay",
    "previous_admissions",
    "age",
    "gender",
    "has_chronic_condition",
    "is_repeat_patient",
    "avg_heart_rate",
    "max_heart_rate",
    "min_heart_rate",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_oxygen_level",
    "min_oxygen_level",
    "avg_temperature",
    "max_temperature",
    "vital_record_count",
    "high_heart_rate_count",
    "low_oxygen_count",
    "high_temperature_count",
    "treatment_count",
    "unique_medication_count",
    "unique_procedure_count",
    "abnormal_vital_flag",
    "readmitted_30_days"
]

ml_df = df[features].copy()

print("Selected features:", len(features))


# 3. HANDLE MISSING VALUES

print("\nChecking missing values...")

numeric_columns = ml_df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    if column != "readmitted_30_days":
        ml_df[column] = ml_df[column].fillna(
            ml_df[column].median()
        )

ml_df["gender"] = ml_df["gender"].fillna("Unknown")

print("Missing values handled.")


# 4. CONVERT CATEGORICAL VARIABLES

print("\nEncoding categorical variables...")

ml_df = pd.get_dummies(
    ml_df,
    columns=["gender"],
    drop_first=True
)

print("Categorical variables encoded.")


# 5. REMOVE IDENTIFIER COLUMNS

print("\nRemoving identifier columns...")

ml_df = ml_df.drop(
    columns=["patient_id", "doctor_id"],
    errors="ignore"
)

print("Identifier columns removed.")


# 6. VALIDATE TARGET

print("\nChecking target variable...")

print(
    "Readmission target distribution:"
)

print(
    ml_df["readmitted_30_days"].value_counts()
)

print(
    "\nReadmission target percentages:"
)

print(
    ml_df["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# 7. SAVE ML DATASET

print("\nSaving machine learning dataset...")

os.makedirs("output", exist_ok=True)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("ML dataset saved successfully.")


# 8. FINAL VALIDATION

print("\n" + "=" * 70)
print("ML DATASET VALIDATION")
print("=" * 70)

print("\nFinal shape:", ml_df.shape)

print("\nRemaining missing values:")

print(
    ml_df.isnull().sum().sum()
)

print(
    "\nTarget column:",
    "readmitted_30_days"
)

print(
    "\nML dataset columns:"
)

for column in ml_df.columns:
    print(" -", column)

print("\n" + "=" * 70)
print("ML DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)