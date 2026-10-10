import pandas as pd
import numpy as np
from pathlib import Path


# HEALTHCARE PATIENT ANALYTICS
# IMPROVED PREDICTION DATASET

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - PREDICTION DATASET")
print("=" * 70)


# 1. LOAD DATA

print("\nLoading source data...")

patients = pd.read_csv(DATA_DIR / "Patients.csv")
admissions = pd.read_csv(DATA_DIR / "Admissions.csv")
vitals = pd.read_csv(DATA_DIR / "Vitals.csv")

print("Patients   :", patients.shape)
print("Admissions :", admissions.shape)
print("Vitals     :", vitals.shape)


# 2. STANDARDIZE COLUMN NAMES

patients.columns = patients.columns.str.strip().str.lower()
admissions.columns = admissions.columns.str.strip().str.lower()
vitals.columns = vitals.columns.str.strip().str.lower()


# 3. CONVERT DATE COLUMNS

patients["dob"] = pd.to_datetime(
    patients["dob"],
    errors="coerce"
)

admissions["admission_date"] = pd.to_datetime(
    admissions["admission_date"],
    errors="coerce"
)

admissions["discharge_date"] = pd.to_datetime(
    admissions["discharge_date"],
    errors="coerce"
)

vitals["recorded_time"] = pd.to_datetime(
    vitals["recorded_time"],
    errors="coerce"
)


# 4. SORT ADMISSIONS

admissions = admissions.sort_values(
    ["patient_id", "admission_date", "admission_id"]
).reset_index(drop=True)


# 5. CALCULATE AGE AT ADMISSION

admissions = admissions.merge(
    patients[
        [
            "patient_id",
            "dob",
            "gender",
            "chronic_conditions"
        ]
    ],
    on="patient_id",
    how="left"
)

admissions["age"] = (
    admissions["admission_date"] - admissions["dob"]
).dt.days / 365.25

admissions["age"] = admissions["age"].round(1)


# 6. CHRONIC CONDITION FLAG

admissions["chronic_conditions"] = (
    admissions["chronic_conditions"]
    .fillna("None Reported")
    .astype(str)
    .str.strip()
)

admissions["has_chronic_condition"] = (
    ~admissions["chronic_conditions"]
    .str.lower()
    .isin([
        "",
        "none",
        "none reported",
        "nan"
    ])
).astype(int)


# 7. PREVIOUS ADMISSION FEATURES

admissions["previous_admissions"] = (
    admissions
    .groupby("patient_id")
    .cumcount()
)

# Previous admission date
admissions["previous_admission_date"] = (
    admissions
    .groupby("patient_id")["admission_date"]
    .shift(1)
)

admissions["days_since_previous_admission"] = (
    admissions["admission_date"]
    - admissions["previous_admission_date"]
).dt.days

admissions["days_since_previous_admission"] = (
    admissions["days_since_previous_admission"]
    .fillna(-1)
)

admissions["is_repeat_patient"] = (
    admissions["previous_admissions"] > 0
).astype(int)


# 8. VALID READMISSION TARGET

# A valid next admission must happen AFTER the current
# admission's discharge date.

# This prevents overlapping admissions from automatically
# being treated as readmissions.

admissions["next_valid_admission_date"] = pd.NaT

for patient_id, group in admissions.groupby("patient_id"):

    indices = group.index.tolist()

    for i in range(len(indices)):

        current_index = indices[i]

        discharge_date = admissions.loc[
            current_index,
            "discharge_date"
        ]

        if pd.isna(discharge_date):
            continue

        future_indices = indices[i + 1:]

        for future_index in future_indices:

            future_admission = admissions.loc[
                future_index,
                "admission_date"
            ]

            if pd.isna(future_admission):
                continue

            if future_admission > discharge_date:

                admissions.loc[
                    current_index,
                    "next_valid_admission_date"
                ] = future_admission

                break


# Calculate days from discharge to next valid admission

admissions["days_to_next_valid_admission"] = (
    admissions["next_valid_admission_date"]
    - admissions["discharge_date"]
).dt.days


# 30-day readmission target

admissions["readmitted_30_days"] = (
    admissions["days_to_next_valid_admission"]
    .between(1, 30)
).astype(int)


# 9. VITAL FEATURES

print("\nPreparing vital features...")

vitals["blood_pressure"] = (
    vitals["blood_pressure"]
    .astype(str)
    .str.strip()
)

bp_split = vitals["blood_pressure"].str.split(
    "/",
    expand=True
)

vitals["systolic_bp"] = pd.to_numeric(
    bp_split[0],
    errors="coerce"
)

vitals["diastolic_bp"] = pd.to_numeric(
    bp_split[1],
    errors="coerce"
)


# Analytical project flags
# These are project-level rules and are NOT medical diagnoses.

vitals["high_heart_rate"] = (
    vitals["heart_rate"] > 100
).astype(int)

vitals["low_oxygen"] = (
    vitals["oxygen_level"] < 95
).astype(int)

vitals["high_temperature"] = (
    vitals["temperature"] > 100.4
).astype(int)


# Aggregate vitals by admission

vital_summary = (
    vitals
    .groupby("admission_id")
    .agg(
        avg_heart_rate=("heart_rate", "mean"),
        max_heart_rate=("heart_rate", "max"),
        min_heart_rate=("heart_rate", "min"),

        avg_systolic_bp=("systolic_bp", "mean"),
        avg_diastolic_bp=("diastolic_bp", "mean"),

        avg_oxygen_level=("oxygen_level", "mean"),
        min_oxygen_level=("oxygen_level", "min"),

        avg_temperature=("temperature", "mean"),
        max_temperature=("temperature", "max"),

        vital_record_count=("vital_id", "count"),

        high_heart_rate_count=("high_heart_rate", "sum"),
        low_oxygen_count=("low_oxygen", "sum"),
        high_temperature_count=("high_temperature", "sum")
    )
    .reset_index()
)


# 10. CALCULATE VITAL ABNORMALITY RATES

vital_summary["high_heart_rate_rate"] = (
    vital_summary["high_heart_rate_count"]
    / vital_summary["vital_record_count"]
)

vital_summary["low_oxygen_rate"] = (
    vital_summary["low_oxygen_count"]
    / vital_summary["vital_record_count"]
)

vital_summary["high_temperature_rate"] = (
    vital_summary["high_temperature_count"]
    / vital_summary["vital_record_count"]
)


vital_summary["abnormal_vital_count"] = (
    vital_summary["high_heart_rate_count"]
    + vital_summary["low_oxygen_count"]
    + vital_summary["high_temperature_count"]
)


vital_summary["abnormal_vital_rate"] = (
    vital_summary["abnormal_vital_count"]
    / vital_summary["vital_record_count"]
)


# 11. MERGE VITAL FEATURES

prediction_data = admissions.merge(
    vital_summary,
    on="admission_id",
    how="left"
)


# 12. HANDLE MISSING VITAL DATA

vital_numeric_columns = [
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
    "high_heart_rate_rate",
    "low_oxygen_rate",
    "high_temperature_rate",
    "abnormal_vital_count",
    "abnormal_vital_rate"
]

for column in vital_numeric_columns:
    prediction_data[column] = (
        prediction_data[column].fillna(0)
    )


# 13. SELECT FINAL PREDICTION FEATURES

final_columns = [
    "admission_id",
    "patient_id",

    # Demographic / history
    "age",
    "gender",
    "has_chronic_condition",
    "previous_admissions",
    "days_since_previous_admission",
    "is_repeat_patient",

    # Vital information
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

    "high_heart_rate_rate",
    "low_oxygen_rate",
    "high_temperature_rate",

    "abnormal_vital_count",
    "abnormal_vital_rate",

    # Target
    "readmitted_30_days"
]

prediction_data = prediction_data[
    final_columns
]


# 14. REMOVE INVALID AGE VALUES

prediction_data = prediction_data[
    (prediction_data["age"] >= 0) &
    (prediction_data["age"] <= 120)
].copy()


# 15. FILL MISSING NUMERIC VALUES

numeric_columns = prediction_data.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if column == "readmitted_30_days":
        continue

    median_value = prediction_data[column].median()

    prediction_data[column] = (
        prediction_data[column]
        .fillna(median_value)
    )


# 16. FILL MISSING GENDER

prediction_data["gender"] = (
    prediction_data["gender"]
    .fillna("Unknown")
)


# 17. SAVE DATASET

output_file = (
    OUTPUT_DIR / "improved_prediction_dataset.csv"
)

prediction_data.to_csv(
    output_file,
    index=False
)


# 18. DATASET SUMMARY

print("\n" + "=" * 70)
print("IMPROVED DATASET SUMMARY")
print("=" * 70)

print(
    "\nDataset shape:",
    prediction_data.shape
)

print(
    "\nReadmission target distribution:"
)

print(
    prediction_data["readmitted_30_days"]
    .value_counts()
    .sort_index()
)

print(
    "\nReadmission target percentage:"
)

print(
    prediction_data["readmitted_30_days"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# 19. VALIDATION

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "\nDuplicate admission IDs:",
    prediction_data["admission_id"].duplicated().sum()
)

print(
    "Missing values:",
    prediction_data.isnull().sum().sum()
)

print(
    "Readmission records:",
    prediction_data["readmitted_30_days"].sum()
)

print(
    "Non-readmission records:",
    (
        prediction_data["readmitted_30_days"] == 0
    ).sum()
)


# 20. FINAL MESSAGE

print("\n" + "=" * 70)
print("PREDICTION DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nSaved file:\n{output_file}"
)

print(
    "\nThe new dataset uses admission history and vital-based "
    "features while avoiding the previous admission-overlap "
    "problem in the readmission target."
)

print("=" * 70)