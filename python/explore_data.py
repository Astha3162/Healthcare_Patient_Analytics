import pandas as pd
import os

# HEALTHCARE PATIENT ANALYTICS
# DATA EXPLORATION

DATA_PATH = "data"

files = {
    "Patients": "patients.csv",
    "Admissions": "admissions.csv",
    "Doctors": "doctors.csv",
    "Vitals": "vitals.csv",
    "Treatments": "treatments.csv"
}


def inspect_dataset(name, filename):
    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    filepath = os.path.join(DATA_PATH, filename)

    df = pd.read_csv(filepath)

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    return df


# LOAD AND INSPECT ALL DATASETS

patients = inspect_dataset(
    "Patients",
    files["Patients"]
)

admissions = inspect_dataset(
    "Admissions",
    files["Admissions"]
)

doctors = inspect_dataset(
    "Doctors",
    files["Doctors"]
)

vitals = inspect_dataset(
    "Vitals",
    files["Vitals"]
)

treatments = inspect_dataset(
    "Treatments",
    files["Treatments"]
)


# RELATIONSHIP CHECKS

print("\n" + "=" * 70)
print("RELATIONSHIP VALIDATION")
print("=" * 70)

patient_ids = set(patients["patient_id"])
doctor_ids = set(doctors["doctor_id"])
admission_ids = set(admissions["admission_id"])

orphan_patients = set(admissions["patient_id"]) - patient_ids
orphan_doctors = set(admissions["doctor_id"]) - doctor_ids
orphan_vitals = set(vitals["admission_id"]) - admission_ids
orphan_treatments = set(treatments["admission_id"]) - admission_ids

print("\nAdmissions with missing patients:")
print(len(orphan_patients))

print("\nAdmissions with missing doctors:")
print(len(orphan_doctors))

print("\nVitals with missing admissions:")
print(len(orphan_vitals))

print("\nTreatments with missing admissions:")
print(len(orphan_treatments))


# SUMMARY

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Patients     : {len(patients)}")
print(f"Admissions   : {len(admissions)}")
print(f"Doctors      : {len(doctors)}")
print(f"Vitals       : {len(vitals)}")
print(f"Treatments   : {len(treatments)}")

print("\nData exploration completed successfully.")