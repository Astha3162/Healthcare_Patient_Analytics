import pandas as pd
from pathlib import Path


# HEALTHCARE PATIENT ANALYTICS
# CHECK FEATURE TIMING FOR PREDICTION

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - PREDICTION TIMING CHECK")
print("=" * 70)


# 1. LOAD DATA

print("\nLoading data...")

admissions = pd.read_csv(DATA_DIR / "Admissions.csv")
vitals = pd.read_csv(DATA_DIR / "Vitals.csv")
treatments = pd.read_csv(DATA_DIR / "Treatments.csv")

print("Admissions :", admissions.shape)
print("Vitals     :", vitals.shape)
print("Treatments :", treatments.shape)


# 2. STANDARDIZE COLUMN NAMES

admissions.columns = admissions.columns.str.strip().str.lower()
vitals.columns = vitals.columns.str.strip().str.lower()
treatments.columns = treatments.columns.str.strip().str.lower()


# 3. CONVERT DATE/TIME COLUMNS

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


# 4. MERGE VITALS WITH ADMISSION DATES

vitals_check = vitals.merge(
    admissions[
        [
            "admission_id",
            "admission_date",
            "discharge_date"
        ]
    ],
    on="admission_id",
    how="left"
)

vitals_check["hours_from_admission"] = (
    vitals_check["recorded_time"]
    - vitals_check["admission_date"]
).dt.total_seconds() / 3600

vitals_check["days_from_admission"] = (
    vitals_check["hours_from_admission"] / 24
)

vitals_check["days_before_discharge"] = (
    vitals_check["discharge_date"]
    - vitals_check["recorded_time"]
).dt.total_seconds() / 86400


# 5. VITAL TIMING SUMMARY

print("\n" + "=" * 70)
print("VITAL TIMING")
print("=" * 70)

print(
    "\nHours from admission:"
)

print(
    vitals_check["hours_from_admission"].describe()
)

print(
    "\nDays before discharge:"
)

print(
    vitals_check["days_before_discharge"].describe()
)

vitals_before_admission = (
    vitals_check["hours_from_admission"] < 0
).sum()

vitals_first_24h = (
    (vitals_check["hours_from_admission"] >= 0) &
    (vitals_check["hours_from_admission"] <= 24)
).sum()

vitals_after_24h = (
    vitals_check["hours_from_admission"] > 24
).sum()

print("\nVital records before admission :", vitals_before_admission)
print("Vital records within first 24h :", vitals_first_24h)
print("Vital records after first 24h  :", vitals_after_24h)


# 6. FIRST VITAL PER ADMISSION

vitals_sorted = vitals_check.sort_values(
    ["admission_id", "recorded_time"]
)

first_vitals = (
    vitals_sorted
    .groupby("admission_id")
    .first()
    .reset_index()
)

print("\nNumber of admissions with vital records:")
print(first_vitals["admission_id"].nunique())

print(
    "First vital timing summary:"
)

print(
    first_vitals["hours_from_admission"].describe()
)


# 7. MERGE TREATMENTS WITH ADMISSION DATES

treatments_check = treatments.merge(
    admissions[
        [
            "admission_id",
            "admission_date",
            "discharge_date"
        ]
    ],
    on="admission_id",
    how="left"
)

treatments_check["days_from_admission"] = (
    treatments_check["treatment_date"]
    - treatments_check["admission_date"]
).dt.total_seconds() / 86400

treatments_check["days_before_discharge"] = (
    treatments_check["discharge_date"]
    - treatments_check["treatment_date"]
).dt.total_seconds() / 86400


# 8. TREATMENT TIMING SUMMARY

print("\n" + "=" * 70)
print("TREATMENT TIMING")
print("=" * 70)

print(
    "\nDays from admission:"
)

print(
    treatments_check["days_from_admission"].describe()
)

treatments_before_admission = (
    treatments_check["days_from_admission"] < 0
).sum()

treatments_first_24h = (
    (treatments_check["days_from_admission"] >= 0) &
    (treatments_check["days_from_admission"] <= 1)
).sum()

treatments_after_24h = (
    treatments_check["days_from_admission"] > 1
).sum()

print("\nTreatment records before admission :", treatments_before_admission)
print("Treatment records within first 24h :", treatments_first_24h)
print("Treatment records after first 24h  :", treatments_after_24h)


# 9. ADMISSION-LEVEL COVERAGE

vital_admissions = vitals["admission_id"].nunique()
treatment_admissions = treatments["admission_id"].nunique()
total_admissions = admissions["admission_id"].nunique()

print("\n" + "=" * 70)
print("ADMISSION COVERAGE")
print("=" * 70)

print("\nTotal admissions              :", total_admissions)
print("Admissions with vitals       :", vital_admissions)
print("Admissions with treatments   :", treatment_admissions)


# 10. CONCLUSION

print("\n" + "=" * 70)
print("TIMING CHECK COMPLETED")
print("=" * 70)

print("""
This analysis checks whether vitals and treatment information
is available early enough to be used for readmission-risk
prediction.

We will use these results to design the next prediction dataset
without unnecessarily using information that becomes available
only after the prediction point.
""")

print("=" * 70)