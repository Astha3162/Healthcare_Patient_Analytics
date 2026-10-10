import pandas as pd
import numpy as np
import os


# HEALTHCARE PATIENT ANALYTICS
# ETL PIPELINE

DATA_PATH = "data"
OUTPUT_PATH = "output"

# Create output folder if it does not exist
os.makedirs(OUTPUT_PATH, exist_ok=True)


# 1. LOAD RAW DATA

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - ETL PIPELINE")
print("=" * 70)

print("\nLoading raw datasets...")

patients = pd.read_csv(
    os.path.join(DATA_PATH, "patients.csv")
)

admissions = pd.read_csv(
    os.path.join(DATA_PATH, "admissions.csv")
)

doctors = pd.read_csv(
    os.path.join(DATA_PATH, "doctors.csv")
)

vitals = pd.read_csv(
    os.path.join(DATA_PATH, "vitals.csv")
)

treatments = pd.read_csv(
    os.path.join(DATA_PATH, "treatments.csv")
)

print("All datasets loaded successfully.")


# 2. STANDARDIZE COLUMN NAMES

print("\nStandardizing column names...")

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


patients = clean_column_names(patients)
admissions = clean_column_names(admissions)
doctors = clean_column_names(doctors)
vitals = clean_column_names(vitals)
treatments = clean_column_names(treatments)

print("Column names standardized.")


# 3. REMOVE DUPLICATES

print("\nRemoving duplicate records...")

patients = patients.drop_duplicates()
admissions = admissions.drop_duplicates()
doctors = doctors.drop_duplicates()
vitals = vitals.drop_duplicates()
treatments = treatments.drop_duplicates()

print("Duplicate removal completed.")


# 4. HANDLE MISSING VALUES

print("\nHandling missing values...")

# chronic_conditions is nullable according to the case study.
# We replace missing values with "None Reported".

patients["chronic_conditions"] = (
    patients["chronic_conditions"]
    .fillna("None Reported")
)

print("Missing chronic condition values handled.")


# 5. CONVERT DATE AND TIME COLUMNS

print("\nConverting date and time columns...")

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

treatments["treatment_date"] = pd.to_datetime(
    treatments["treatment_date"],
    errors="coerce"
)

print("Date conversion completed.")


# 6. CREATE PATIENT AGE

print("\nCalculating patient age...")

REFERENCE_DATE = pd.Timestamp("2026-12-31")

patients["age"] = (
    REFERENCE_DATE.year
    - patients["dob"].dt.year
)

# If the patient's birthday has not occurred yet
# in the reference year, subtract one year.
birthday_not_reached = (
    (patients["dob"].dt.month > REFERENCE_DATE.month)
    |
    (
        (patients["dob"].dt.month == REFERENCE_DATE.month)
        &
        (patients["dob"].dt.day > REFERENCE_DATE.day)
    )
)

patients.loc[birthday_not_reached, "age"] -= 1

patients["age"] = patients["age"].astype(int)

print("Patient age calculated.")


# 7. CALCULATE LENGTH OF STAY

print("\nCalculating hospital length of stay...")

admissions["length_of_stay"] = np.where(
    admissions["discharge_date"].notna(),
    (
        admissions["discharge_date"]
        - admissions["admission_date"]
    ).dt.days,
    np.nan
)

# Prevent negative stays if any invalid record exists
admissions.loc[
    admissions["length_of_stay"] < 0,
    "length_of_stay"
] = np.nan

print("Length of stay calculated.")


# 8. PROCESS BLOOD PRESSURE

print("\nProcessing blood pressure values...")

blood_pressure = (
    vitals["blood_pressure"]
    .astype(str)
    .str.split("/", expand=True)
)

vitals["systolic_bp"] = pd.to_numeric(
    blood_pressure[0],
    errors="coerce"
)

vitals["diastolic_bp"] = pd.to_numeric(
    blood_pressure[1],
    errors="coerce"
)

print("Blood pressure processing completed.")


# 9. CREATE ABNORMAL VITAL FLAGS

print("\nCreating abnormal vital indicators...")

# These thresholds are analytical rules for this project,
# not medical diagnosis rules.

vitals["high_heart_rate"] = np.where(
    vitals["heart_rate"] > 100,
    1,
    0
)

vitals["low_oxygen"] = np.where(
    vitals["oxygen_level"] < 95,
    1,
    0
)

vitals["high_temperature"] = np.where(
    vitals["temperature"] > 100.4,
    1,
    0
)

print("Abnormal vital indicators created.")


# 10. AGGREGATE VITALS BY ADMISSION

print("\nAggregating vital information by admission...")

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

print(
    f"Vital summary created for "
    f"{len(vital_summary)} admissions."
)


# 11. AGGREGATE TREATMENTS BY ADMISSION

print("\nAggregating treatment information...")

treatment_summary = (
    treatments
    .groupby("admission_id")
    .agg(
        treatment_count=("treatment_id", "count"),
        unique_medication_count=("medication", "nunique"),
        unique_procedure_count=("procedure", "nunique")
    )
    .reset_index()
)

print(
    f"Treatment summary created for "
    f"{len(treatment_summary)} admissions."
)


# 12. CALCULATE PREVIOUS ADMISSIONS

print("\nCalculating previous admission counts...")

admissions = admissions.sort_values(
    ["patient_id", "admission_date"]
).reset_index(drop=True)

admissions["previous_admissions"] = (
    admissions
    .groupby("patient_id")
    .cumcount()
)

print("Previous admission counts calculated.")


# 13. CALCULATE 30-DAY READMISSION TARGET

print("\nCreating 30-day readmission target...")

# For every admission, find the patient's next admission date.

admissions["next_admission_date"] = (
    admissions
    .groupby("patient_id")["admission_date"]
    .shift(-1)
)

admissions["days_to_next_admission"] = (
    admissions["next_admission_date"]
    - admissions["admission_date"]
).dt.days

# Readmission is defined as another admission
# within 30 days after the current admission.

admissions["readmitted_30_days"] = np.where(
    admissions["days_to_next_admission"].between(
        1, 30, inclusive="both"
    ),
    1,
    0
)

print(
    "30-day readmission target created."
)

print(
    "Total 30-day readmissions:",
    admissions["readmitted_30_days"].sum()
)


# 14. CREATE ANALYTICAL ADMISSION DATASET

print("\nCreating analytical admission dataset...")

admission_analytics = admissions.merge(
    patients[
        [
            "patient_id",
            "dob",
            "gender",
            "chronic_conditions",
            "age"
        ]
    ],
    on="patient_id",
    how="left"
)

admission_analytics = admission_analytics.merge(
    doctors[
        [
            "doctor_id",
            "specialization"
        ]
    ],
    on="doctor_id",
    how="left"
)

admission_analytics = admission_analytics.merge(
    vital_summary,
    on="admission_id",
    how="left"
)

admission_analytics = admission_analytics.merge(
    treatment_summary,
    on="admission_id",
    how="left"
)


# 15. FILL MISSING AGGREGATED VALUES

print("\nFilling missing aggregated values...")

numeric_summary_columns = [
    "treatment_count",
    "unique_medication_count",
    "unique_procedure_count",
    "vital_record_count",
    "high_heart_rate_count",
    "low_oxygen_count",
    "high_temperature_count"
]

for column in numeric_summary_columns:
    admission_analytics[column] = (
        admission_analytics[column]
        .fillna(0)
    )


# 16. CREATE HIGH-RISK VITAL FLAG

admission_analytics["abnormal_vital_flag"] = np.where(
    (
        (admission_analytics["high_heart_rate_count"] > 0)
        |
        (admission_analytics["low_oxygen_count"] > 0)
        |
        (admission_analytics["high_temperature_count"] > 0)
    ),
    1,
    0
)


# 17. CREATE READABLE RISK FEATURES

admission_analytics["has_chronic_condition"] = np.where(
    admission_analytics["chronic_conditions"]
    .str.lower()
    .eq("none reported"),
    0,
    1
)

admission_analytics["is_repeat_patient"] = np.where(
    admission_analytics["previous_admissions"] > 0,
    1,
    0
)


# 18. SAVE CLEANED DATASETS

print("\nSaving cleaned datasets...")

patients.to_csv(
    os.path.join(OUTPUT_PATH, "patients_clean.csv"),
    index=False
)

doctors.to_csv(
    os.path.join(OUTPUT_PATH, "doctors_clean.csv"),
    index=False
)

admissions.to_csv(
    os.path.join(OUTPUT_PATH, "admissions_clean.csv"),
    index=False
)

vitals.to_csv(
    os.path.join(OUTPUT_PATH, "vitals_clean.csv"),
    index=False
)

treatments.to_csv(
    os.path.join(OUTPUT_PATH, "treatments_clean.csv"),
    index=False
)

vital_summary.to_csv(
    os.path.join(OUTPUT_PATH, "vital_summary.csv"),
    index=False
)

treatment_summary.to_csv(
    os.path.join(OUTPUT_PATH, "treatment_summary.csv"),
    index=False
)

admission_analytics.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "admission_analytics.csv"
    ),
    index=False
)


# 19. DATA QUALITY VALIDATION

print("\n" + "=" * 70)
print("DATA QUALITY VALIDATION")
print("=" * 70)

print(
    "\nPatients rows:",
    len(patients)
)

print(
    "Doctors rows:",
    len(doctors)
)

print(
    "Admissions rows:",
    len(admissions)
)

print(
    "Vitals rows:",
    len(vitals)
)

print(
    "Treatments rows:",
    len(treatments)
)

print(
    "Analytical admissions rows:",
    len(admission_analytics)
)

print(
    "\nMissing patient IDs in analytical dataset:",
    admission_analytics["patient_id"].isna().sum()
)

print(
    "Missing doctor IDs in analytical dataset:",
    admission_analytics["doctor_id"].isna().sum()
)

print(
    "Missing vital summaries:",
    admission_analytics["vital_record_count"].isna().sum()
)

print(
    "Missing treatment counts:",
    admission_analytics["treatment_count"].isna().sum()
)


# 20. FINAL SUMMARY

print("\n" + "=" * 70)
print("ETL PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput files created in:")
print(OUTPUT_PATH)

print("\nFiles:")

for filename in sorted(os.listdir(OUTPUT_PATH)):
    print(" -", filename)

print("\n30-day readmission records:")
print(
    admissions["readmitted_30_days"].sum()
)

print("\nETL pipeline finished.")