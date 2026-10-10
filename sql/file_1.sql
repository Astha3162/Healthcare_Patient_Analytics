
-- HEALTHCARE PATIENT ANALYTICS
-- MYSQL DATA WAREHOUSE

-- Create database
CREATE DATABASE IF NOT EXISTS healthcare_dw;

USE healthcare_dw;

-- DROP EXISTING TABLES

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS Readmission_Risk;
DROP TABLE IF EXISTS Treatments;
DROP TABLE IF EXISTS Vitals;
DROP TABLE IF EXISTS Admissions;
DROP TABLE IF EXISTS Patients;
DROP TABLE IF EXISTS Doctors;

SET FOREIGN_KEY_CHECKS = 1;


-- 1. DOCTORS TABLE

CREATE TABLE Doctors (
    doctor_id INT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    specialization VARCHAR(150),
    contact_no VARCHAR(30)
);


-- 2. PATIENTS TABLE

CREATE TABLE Patients (
    patient_id INT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    dob DATE,
    gender VARCHAR(20),
    contact_no VARCHAR(30),
    address VARCHAR(255),
    chronic_conditions VARCHAR(255)
);


-- 3. ADMISSIONS TABLE

CREATE TABLE Admissions (
    admission_id INT PRIMARY KEY,
    patient_id INT NOT NULL,
    admission_date DATETIME,
    discharge_date DATETIME,
    diagnosis VARCHAR(255),
    doctor_id INT NOT NULL,
    room_no VARCHAR(50),
    length_of_stay DECIMAL(10,2),
    previous_admissions INT,
    abnormal_vital_flag INT,
    has_chronic_condition INT,
    is_repeat_patient INT,

    FOREIGN KEY (patient_id)
        REFERENCES Patients(patient_id),

    FOREIGN KEY (doctor_id)
        REFERENCES Doctors(doctor_id)
);


-- 4. VITALS TABLE

CREATE TABLE Vitals (
    vital_id INT PRIMARY KEY,
    admission_id INT NOT NULL,
    recorded_time DATETIME,
    heart_rate DECIMAL(10,2),
    blood_pressure VARCHAR(30),
    oxygen_level DECIMAL(10,2),
    temperature DECIMAL(10,2),

    FOREIGN KEY (admission_id)
        REFERENCES Admissions(admission_id)
);


-- 5. TREATMENTS TABLE

CREATE TABLE Treatments (
    treatment_id INT PRIMARY KEY,
    admission_id INT NOT NULL,
    treatment_date DATE,
    procedure_name VARCHAR(255),
    medication VARCHAR(255),
    dosage VARCHAR(100),

    FOREIGN KEY (admission_id)
        REFERENCES Admissions(admission_id)
);


-- 6. READMISSION RISK TABLE

CREATE TABLE Readmission_Risk (
    risk_id INT AUTO_INCREMENT PRIMARY KEY,
    admission_id INT NOT NULL,
    patient_id INT NOT NULL,
    prediction_date DATE,
    risk_score DECIMAL(5,2),
    risk_level VARCHAR(20),
    actual_readmission_30_days INT,

    FOREIGN KEY (admission_id)
        REFERENCES Admissions(admission_id),

    FOREIGN KEY (patient_id)
        REFERENCES Patients(patient_id)
);


-- DISPLAY TABLES

SHOW TABLES;




USE healthcare_dw;

SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE readmission_risk;
TRUNCATE TABLE treatments;
TRUNCATE TABLE vitals;
TRUNCATE TABLE admissions;
TRUNCATE TABLE patients;
TRUNCATE TABLE doctors;

SET FOREIGN_KEY_CHECKS = 1;






USE healthcare_dw;

SELECT
    'Doctors' AS table_name,
    COUNT(*) AS row_count
FROM doctors

UNION ALL

SELECT
    'Patients',
    COUNT(*)
FROM patients

UNION ALL

SELECT
    'Admissions',
    COUNT(*)
FROM admissions

UNION ALL

SELECT
    'Vitals',
    COUNT(*)
FROM vitals

UNION ALL

SELECT
    'Treatments',
    COUNT(*)
FROM treatments

UNION ALL

SELECT
    'Readmission_Risk',
    COUNT(*)
FROM readmission_risk;






USE healthcare_dw;

SELECT COUNT(*) AS missing_patient_admissions
FROM admissions a
LEFT JOIN patients p
    ON a.patient_id = p.patient_id
WHERE p.patient_id IS NULL;







SELECT COUNT(*) AS missing_doctor_admissions
FROM admissions a
LEFT JOIN doctors d
    ON a.doctor_id = d.doctor_id
WHERE d.doctor_id IS NULL;







SELECT COUNT(*) AS missing_vital_admissions
FROM vitals v
LEFT JOIN admissions a
    ON v.admission_id = a.admission_id
WHERE a.admission_id IS NULL;






SELECT COUNT(*) AS missing_treatment_admissions
FROM treatments t
LEFT JOIN admissions a
    ON t.admission_id = a.admission_id
WHERE a.admission_id IS NULL;







SELECT COUNT(*) AS missing_risk_admissions
FROM readmission_risk r
LEFT JOIN admissions a
    ON r.admission_id = a.admission_id
WHERE a.admission_id IS NULL;