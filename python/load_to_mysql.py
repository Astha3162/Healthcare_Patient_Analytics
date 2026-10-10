import pandas as pd
import mysql.connector
import os
from getpass import getpass


# HEALTHCARE PATIENT ANALYTICS
# LOAD DATA INTO MYSQL

print("=" * 70)
print("HEALTHCARE PATIENT ANALYTICS - MYSQL DATA LOADING")
print("=" * 70)


# 1. Project paths

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


# 2. MySQL connection details

print("\nEnter your MySQL credentials.")

MYSQL_HOST = "localhost"
MYSQL_USER = input("MySQL username [root]: ").strip()

if MYSQL_USER == "":
    MYSQL_USER = "root"

MYSQL_PASSWORD = getpass(
    "MySQL password: "
)


# 3. Connect to MySQL

print("\nConnecting to MySQL...")

connection = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database="healthcare_dw"
)

cursor = connection.cursor()

print("Connected to MySQL successfully.")


# 4. Helper function

def clean_value(value):

    if pd.isna(value):
        return None

    return value


def insert_dataframe(
    dataframe,
    sql,
    columns,
    table_name
):

    records = []

    for _, row in dataframe.iterrows():

        values = tuple(
            clean_value(row[column])
            for column in columns
        )

        records.append(values)

    cursor.executemany(
        sql,
        records
    )

    connection.commit()

    print(
        f"{table_name}: {len(records)} rows loaded successfully."
    )



# 5. LOAD DOCTORS

print("\n" + "=" * 70)
print("1. LOADING DOCTORS")
print("=" * 70)

doctors_file = os.path.join(
    OUTPUT_DIR,
    "doctors_clean.csv"
)

doctors = pd.read_csv(
    doctors_file
)

doctor_columns = [
    "doctor_id",
    "first_name",
    "last_name",
    "specialization",
    "contact_no"
]

doctor_sql = """
INSERT INTO doctors
(
    doctor_id,
    first_name,
    last_name,
    specialization,
    contact_no
)
VALUES (%s, %s, %s, %s, %s)
"""

insert_dataframe(
    doctors,
    doctor_sql,
    doctor_columns,
    "Doctors"
)


# 6. LOAD PATIENTS

print("\n" + "=" * 70)
print("2. LOADING PATIENTS")
print("=" * 70)

patients_file = os.path.join(
    OUTPUT_DIR,
    "patients_clean.csv"
)

patients = pd.read_csv(
    patients_file
)

patient_columns = [
    "patient_id",
    "first_name",
    "last_name",
    "dob",
    "gender",
    "contact_no",
    "address",
    "chronic_conditions"
]

patient_sql = """
INSERT INTO patients
(
    patient_id,
    first_name,
    last_name,
    dob,
    gender,
    contact_no,
    address,
    chronic_conditions
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

insert_dataframe(
    patients,
    patient_sql,
    patient_columns,
    "Patients"
)


# 7. LOAD ADMISSIONS

print("\n" + "=" * 70)
print("3. LOADING ADMISSIONS")
print("=" * 70)

admissions_file = os.path.join(
    OUTPUT_DIR,
    "admission_analytics.csv"
)

admissions = pd.read_csv(
    admissions_file
)

admission_columns = [
    "admission_id",
    "patient_id",
    "admission_date",
    "discharge_date",
    "diagnosis",
    "doctor_id",
    "room_no",
    "length_of_stay",
    "previous_admissions",
    "abnormal_vital_flag",
    "has_chronic_condition",
    "is_repeat_patient"
]

admission_sql = """
INSERT INTO admissions
(
    admission_id,
    patient_id,
    admission_date,
    discharge_date,
    diagnosis,
    doctor_id,
    room_no,
    length_of_stay,
    previous_admissions,
    abnormal_vital_flag,
    has_chronic_condition,
    is_repeat_patient
)
VALUES
(
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s
)
"""

insert_dataframe(
    admissions,
    admission_sql,
    admission_columns,
    "Admissions"
)


# 8. LOAD VITALS

print("\n" + "=" * 70)
print("4. LOADING VITALS")
print("=" * 70)

vitals_file = os.path.join(
    OUTPUT_DIR,
    "vitals_clean.csv"
)

vitals = pd.read_csv(
    vitals_file
)

vital_columns = [
    "vital_id",
    "admission_id",
    "recorded_time",
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "temperature"
]

vital_sql = """
INSERT INTO vitals
(
    vital_id,
    admission_id,
    recorded_time,
    heart_rate,
    blood_pressure,
    oxygen_level,
    temperature
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

insert_dataframe(
    vitals,
    vital_sql,
    vital_columns,
    "Vitals"
)


# 9. LOAD TREATMENTS

print("\n" + "=" * 70)
print("5. LOADING TREATMENTS")
print("=" * 70)

treatments_file = os.path.join(
    OUTPUT_DIR,
    "treatments_clean.csv"
)

treatments = pd.read_csv(
    treatments_file
)

treatment_columns = [
    "treatment_id",
    "admission_id",
    "treatment_date",
    "procedure",
    "medication",
    "dosage"
]

treatment_sql = """
INSERT INTO treatments
(
    treatment_id,
    admission_id,
    treatment_date,
    procedure_name,
    medication,
    dosage
)
VALUES (%s, %s, %s, %s, %s, %s)
"""

# Rename procedure temporarily because the MySQL column
# is called procedure_name.

treatments["procedure_name"] = treatments["procedure"]

treatment_columns_for_insert = [
    "treatment_id",
    "admission_id",
    "treatment_date",
    "procedure_name",
    "medication",
    "dosage"
]

insert_dataframe(
    treatments,
    treatment_sql,
    treatment_columns_for_insert,
    "Treatments"
)


# 10. LOAD READMISSION RISK

print("\n" + "=" * 70)
print("6. LOADING READMISSION RISK")
print("=" * 70)

risk_file = os.path.join(
    OUTPUT_DIR,
    "readmission_risk_predictions.csv"
)

risk = pd.read_csv(
    risk_file
)

risk_columns = [
    "admission_id",
    "patient_id",
    "prediction_date",
    "risk_score",
    "risk_level",
    "actual_readmission_30_days"
]

risk_sql = """
INSERT INTO readmission_risk
(
    admission_id,
    patient_id,
    prediction_date,
    risk_score,
    risk_level,
    actual_readmission_30_days
)
VALUES (%s, %s, %s, %s, %s, %s)
"""

insert_dataframe(
    risk,
    risk_sql,
    risk_columns,
    "Readmission_Risk"
)


# 11. CLOSE CONNECTION

cursor.close()
connection.close()

print("\n" + "=" * 70)
print("ALL DATA LOADED SUCCESSFULLY")
print("=" * 70)

print("\nMySQL database:")
print("healthcare_dw")

print("\nTables loaded:")
print("1. Doctors")
print("2. Patients")
print("3. Admissions")
print("4. Vitals")
print("5. Treatments")
print("6. Readmission_Risk")

print("\n" + "=" * 70)
print("MYSQL DATA LOADING COMPLETED")
print("=" * 70)