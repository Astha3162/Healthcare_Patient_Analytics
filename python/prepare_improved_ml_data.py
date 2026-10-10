import pandas as pd
import numpy as np
from pathlib import Path


# HEALTHCARE PATIENT ANALYTICS
# PREPARE IMPROVED ML DATASET

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "improved_prediction_dataset.csv"
OUTPUT_FILE = OUTPUT_DIR / "improved_ml_dataset.csv"

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - ML DATA PREPARATION")
print("=" * 70)


# 1. LOAD DATA

df = pd.read_csv(INPUT_FILE)

print("\nInput dataset shape:", df.shape)


# 2. SEPARATE TARGET

target_column = "readmitted_30_days"

y = df[target_column].copy()

X = df.drop(
    columns=[target_column]
).copy()


# 3. REMOVE IDENTIFIER COLUMNS

identifier_columns = [
    "admission_id",
    "patient_id"
]

X = X.drop(
    columns=identifier_columns,
    errors="ignore"
)


# 4. CONVERT GENDER TO NUMERIC

X["gender"] = (
    X["gender"]
    .fillna("Unknown")
)

X = pd.get_dummies(
    X,
    columns=["gender"],
    drop_first=True,
    dtype=int
)


# 5. HANDLE NUMERIC MISSING VALUES

numeric_columns = X.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    median_value = X[column].median()

    X[column] = (
        X[column]
        .fillna(median_value)
    )


# 6. COMBINE FEATURES AND TARGET

ml_dataset = X.copy()

ml_dataset[target_column] = y


# 7. FINAL VALIDATION

print("\n" + "=" * 70)
print("ML DATASET SUMMARY")
print("=" * 70)

print(
    "\nRows:",
    ml_dataset.shape[0]
)

print(
    "Columns:",
    ml_dataset.shape[1]
)

print(
    "\nFeature columns:"
)

for column in X.columns:
    print("-", column)


print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
    .sort_index()
)


print(
    "\nTarget percentage:"
)

print(
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print(
    "\nMissing values:",
    ml_dataset.isnull().sum().sum()
)


print(
    "Duplicate rows:",
    ml_dataset.duplicated().sum()
)


# 8. SAVE

ml_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("ML DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nSaved file:\n{OUTPUT_FILE}"
)

print("=" * 70)