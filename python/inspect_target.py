import pandas as pd

# HEALTHCARE PATIENT ANALYTICS
# READMISSION TARGET INVESTIGATION

INPUT_FILE = "output/admission_analytics.csv"

print("=" * 70)
print("READMISSION TARGET INVESTIGATION")
print("=" * 70)


# 1. LOAD DATA

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# 2. TARGET DISTRIBUTION

print("\n" + "=" * 70)
print("READMISSION TARGET")
print("=" * 70)

print(
    df["readmitted_30_days"].value_counts()
)

print("\nPercentages:")

print(
    df["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# 3. COMPARE NUMERIC FEATURES

print("\n" + "=" * 70)
print("FEATURE COMPARISON")
print("=" * 70)

numeric_features = [
    "length_of_stay",
    "previous_admissions",
    "age",
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
    "unique_procedure_count"
]

comparison = df.groupby(
    "readmitted_30_days"
)[numeric_features].mean().T

comparison.columns = [
    "Not Readmitted",
    "Readmitted"
]

comparison["Difference"] = (
    comparison["Readmitted"]
    - comparison["Not Readmitted"]
)

print(
    comparison.round(2).to_string()
)


# 4. CATEGORICAL / FLAG FEATURES

print("\n" + "=" * 70)
print("READMISSION RATE BY KEY FEATURES")
print("=" * 70)

categorical_features = [
    "gender",
    "has_chronic_condition",
    "is_repeat_patient",
    "abnormal_vital_flag"
]

for column in categorical_features:

    print("\n" + "-" * 60)
    print(column)

    rate = (
        df.groupby(column)["readmitted_30_days"]
        .agg(["count", "sum", "mean"])
    )

    rate["readmission_rate_percent"] = (
        rate["mean"] * 100
    )

    print(
        rate[
            [
                "count",
                "sum",
                "readmission_rate_percent"
            ]
        ].round(2).to_string()
    )


# 5. READMISSION TIMING

print("\n" + "=" * 70)
print("READMISSION TIMING")
print("=" * 70)

readmitted = df[
    df["readmitted_30_days"] == 1
].copy()

print(
    "\nNumber of readmitted records:",
    len(readmitted)
)

print(
    "\nDays to next admission statistics:"
)

print(
    readmitted["days_to_next_admission"]
    .describe()
    .round(2)
)


# 6. CHECK WHETHER READMISSIONS OCCUR AFTER DISCHARGE

print("\n" + "=" * 70)
print("DISCHARGE-TO-NEXT-ADMISSION CHECK")
print("=" * 70)

df["admission_date"] = pd.to_datetime(
    df["admission_date"]
)

df["discharge_date"] = pd.to_datetime(
    df["discharge_date"]
)

df["next_admission_date"] = pd.to_datetime(
    df["next_admission_date"]
)

df["days_from_discharge_to_next_admission"] = (
    df["next_admission_date"]
    - df["discharge_date"]
).dt.days

valid_next = df[
    df["next_admission_date"].notna()
].copy()

print(
    "\nRecords with a subsequent admission:",
    len(valid_next)
)

print(
    "\nDays from discharge to next admission:"
)

print(
    valid_next[
        "days_from_discharge_to_next_admission"
    ]
    .describe()
    .round(2)
)

print(
    "\nRecords where next admission occurred before discharge:",
    (
        valid_next[
            "days_from_discharge_to_next_admission"
        ] < 0
    ).sum()
)

print(
    "\nRecords where next admission occurred within 30 days after discharge:",
    (
        valid_next[
            "days_from_discharge_to_next_admission"
        ].between(1, 30)
    ).sum()
)


# 7. CURRENT TARGET VS DISCHARGE-BASED TARGET

df["readmitted_30_days_after_discharge"] = (
    df[
        "days_from_discharge_to_next_admission"
    ].between(1, 30)
    .astype(int)
)

print("\n" + "=" * 70)
print("TARGET COMPARISON")
print("=" * 70)

print(
    "\nCurrent target:",
    df["readmitted_30_days"].sum()
)

print(
    "Discharge-based target:",
    df["readmitted_30_days_after_discharge"].sum()
)

print(
    "\nCurrent target percentage:",
    round(
        df["readmitted_30_days"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Discharge-based target percentage:",
    round(
        df[
            "readmitted_30_days_after_discharge"
        ].mean() * 100,
        2
    ),
    "%"
)


# 8. FINAL

print("\n" + "=" * 70)
print("TARGET INVESTIGATION COMPLETED")
print("=" * 70)