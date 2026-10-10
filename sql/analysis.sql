USE healthcare_dw;


-- HEALTHCARE PATIENT ANALYTICS
-- SQL ANALYSIS


-- QUERY 1: OVERALL ADMISSION OVERVIEW

SELECT
    COUNT(*) AS total_admissions,
    COUNT(DISTINCT patient_id) AS unique_patients,
    COUNT(DISTINCT doctor_id) AS unique_doctors,
    ROUND(AVG(length_of_stay), 2) AS average_length_of_stay,
    ROUND(AVG(previous_admissions), 2) AS average_previous_admissions
FROM admissions;


-- QUERY 2: ADMISSIONS BY GENDER

SELECT
    p.gender,
    COUNT(*) AS total_admissions
FROM admissions a
JOIN patients p
    ON a.patient_id = p.patient_id
GROUP BY p.gender
ORDER BY total_admissions DESC;


-- QUERY 3: ADMISSIONS BY DIAGNOSIS

SELECT
    diagnosis,
    COUNT(*) AS admission_count
FROM admissions
GROUP BY diagnosis
ORDER BY admission_count DESC;


-- QUERY 4: ADMISSIONS BY DOCTOR SPECIALIZATION

SELECT
    d.specialization,
    COUNT(*) AS total_admissions
FROM admissions a
JOIN doctors d
    ON a.doctor_id = d.doctor_id
GROUP BY d.specialization
ORDER BY total_admissions DESC;


-- QUERY 5: AVERAGE LENGTH OF STAY BY DIAGNOSIS

SELECT
    diagnosis,
    COUNT(*) AS admissions,
    ROUND(AVG(length_of_stay), 2) AS average_length_of_stay
FROM admissions
GROUP BY diagnosis
ORDER BY average_length_of_stay DESC;


-- QUERY 6: PATIENTS WITH CHRONIC CONDITIONS

SELECT
    COUNT(*) AS patients_with_chronic_conditions
FROM patients
WHERE chronic_conditions <> 'None Reported';


-- QUERY 7: REPEAT PATIENT ANALYSIS

SELECT
    COUNT(*) AS repeat_patient_admissions
FROM admissions
WHERE is_repeat_patient = 1;


-- QUERY 8: ADMISSIONS WITH ABNORMAL VITALS

SELECT
    COUNT(*) AS admissions_with_abnormal_vitals
FROM admissions
WHERE abnormal_vital_flag = 1;


-- QUERY 9: MONTHLY ADMISSION TREND

SELECT
    DATE_FORMAT(admission_date, '%Y-%m') AS admission_month,
    COUNT(*) AS total_admissions
FROM admissions
GROUP BY DATE_FORMAT(admission_date, '%Y-%m')
ORDER BY admission_month;


-- QUERY 10: AVERAGE VITAL MEASUREMENTS

SELECT
    ROUND(AVG(heart_rate), 2) AS average_heart_rate,
    ROUND(AVG(oxygen_level), 2) AS average_oxygen_level,
    ROUND(AVG(temperature), 2) AS average_temperature
FROM vitals;


-- QUERY 11: VITALS BY ADMISSION

SELECT
    admission_id,
    COUNT(*) AS vital_records,
    ROUND(AVG(heart_rate), 2) AS avg_heart_rate,
    ROUND(AVG(oxygen_level), 2) AS avg_oxygen_level,
    ROUND(AVG(temperature), 2) AS avg_temperature
FROM vitals
GROUP BY admission_id
ORDER BY vital_records DESC
LIMIT 10;


-- QUERY 12: LOW OXYGEN LEVEL RECORDS

SELECT
    COUNT(*) AS low_oxygen_records
FROM vitals
WHERE oxygen_level < 95;


-- QUERY 13: HIGH HEART RATE RECORDS

SELECT
    COUNT(*) AS high_heart_rate_records
FROM vitals
WHERE heart_rate > 100;


-- QUERY 14: HIGH TEMPERATURE RECORDS

SELECT
    COUNT(*) AS high_temperature_records
FROM vitals
WHERE temperature > 100.4;


-- QUERY 15: TREATMENT SUMMARY

SELECT
    COUNT(*) AS total_treatments,
    COUNT(DISTINCT admission_id) AS admissions_with_treatments,
    COUNT(DISTINCT medication) AS unique_medications,
    COUNT(DISTINCT procedure_name) AS unique_procedures
FROM treatments;


-- QUERY 16: MOST COMMON MEDICATIONS

SELECT
    medication,
    COUNT(*) AS treatment_count
FROM treatments
GROUP BY medication
ORDER BY treatment_count DESC
LIMIT 10;


-- QUERY 17: MOST COMMON PROCEDURES

SELECT
    procedure_name,
    COUNT(*) AS procedure_count
FROM treatments
GROUP BY procedure_name
ORDER BY procedure_count DESC
LIMIT 10;


-- QUERY 18: READMISSION RISK DISTRIBUTION

SELECT
    risk_level,
    COUNT(*) AS admission_count,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM readmission_risk),
        2
    ) AS percentage
FROM readmission_risk
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END;


-- QUERY 19: AVERAGE RISK SCORE BY RISK LEVEL

SELECT
    risk_level,
    ROUND(AVG(risk_score), 2) AS average_risk_score,
    ROUND(MIN(risk_score), 2) AS minimum_risk_score,
    ROUND(MAX(risk_score), 2) AS maximum_risk_score
FROM readmission_risk
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END;


-- QUERY 20: TOP 10 HIGHEST-RISK ADMISSIONS

SELECT
    r.admission_id,
    r.patient_id,
    r.risk_score,
    r.risk_level,
    a.diagnosis,
    a.length_of_stay
FROM readmission_risk r
JOIN admissions a
    ON r.admission_id = a.admission_id
ORDER BY r.risk_score DESC
LIMIT 10;


-- QUERY 21: HIGH-RISK ADMISSIONS BY DIAGNOSIS

SELECT
    a.diagnosis,
    COUNT(*) AS high_risk_admissions,
    ROUND(AVG(r.risk_score), 2) AS average_risk_score
FROM readmission_risk r
JOIN admissions a
    ON r.admission_id = a.admission_id
WHERE r.risk_level = 'High'
GROUP BY a.diagnosis
ORDER BY high_risk_admissions DESC;


-- QUERY 22: HIGH-RISK ADMISSIONS WITH CHRONIC CONDITIONS

SELECT
    a.has_chronic_condition,
    COUNT(*) AS high_risk_admissions,
    ROUND(AVG(r.risk_score), 2) AS average_risk_score
FROM readmission_risk r
JOIN admissions a
    ON r.admission_id = a.admission_id
WHERE r.risk_level = 'High'
GROUP BY a.has_chronic_condition
ORDER BY high_risk_admissions DESC;


-- QUERY 23: HIGH-RISK REPEAT PATIENTS

SELECT
    a.is_repeat_patient,
    COUNT(*) AS high_risk_admissions,
    ROUND(AVG(r.risk_score), 2) AS average_risk_score
FROM readmission_risk r
JOIN admissions a
    ON r.admission_id = a.admission_id
WHERE r.risk_level = 'High'
GROUP BY a.is_repeat_patient
ORDER BY high_risk_admissions DESC;


-- QUERY 24: ACTUAL 30-DAY READMISSION RATE

SELECT
    COUNT(*) AS total_admissions,
    SUM(actual_readmission_30_days) AS readmitted_within_30_days,
    ROUND(
        SUM(actual_readmission_30_days) * 100.0 /
        COUNT(*),
        2
    ) AS readmission_rate_percentage
FROM readmission_risk;


-- QUERY 25: PREDICTED VS ACTUAL READMISSION

SELECT
    risk_level,
    COUNT(*) AS total_admissions,
    SUM(actual_readmission_30_days) AS actual_readmissions,
    ROUND(
        SUM(actual_readmission_30_days) * 100.0 /
        COUNT(*),
        2
    ) AS actual_readmission_rate
FROM readmission_risk
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END;


-- QUERY 26: HIGH-RISK PATIENT DETAILS

SELECT
    r.admission_id,
    r.patient_id,
    CONCAT(p.first_name, ' ', p.last_name) AS patient_name,
    p.gender,
    p.chronic_conditions,
    a.diagnosis,
    r.risk_score,
    r.risk_level
FROM readmission_risk r
JOIN patients p
    ON r.patient_id = p.patient_id
JOIN admissions a
    ON r.admission_id = a.admission_id
WHERE r.risk_level = 'High'
ORDER BY r.risk_score DESC
LIMIT 20;


-- QUERY 27: DOCTOR ADMISSION AND HIGH-RISK ANALYSIS

SELECT
    d.doctor_id,
    CONCAT(d.first_name, ' ', d.last_name) AS doctor_name,
    d.specialization,
    COUNT(a.admission_id) AS total_admissions,
    SUM(
        CASE
            WHEN r.risk_level = 'High' THEN 1
            ELSE 0
        END
    ) AS high_risk_admissions
FROM doctors d
LEFT JOIN admissions a
    ON d.doctor_id = a.doctor_id
LEFT JOIN readmission_risk r
    ON a.admission_id = r.admission_id
GROUP BY
    d.doctor_id,
    d.first_name,
    d.last_name,
    d.specialization
ORDER BY high_risk_admissions DESC;


-- QUERY 28: ADMISSIONS BY CHRONIC CONDITION AND RISK

SELECT
    a.has_chronic_condition,
    r.risk_level,
    COUNT(*) AS admission_count
FROM admissions a
JOIN readmission_risk r
    ON a.admission_id = r.admission_id
GROUP BY
    a.has_chronic_condition,
    r.risk_level
ORDER BY
    a.has_chronic_condition DESC,
    CASE r.risk_level
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END;


-- QUERY 29: AVERAGE LENGTH OF STAY BY RISK LEVEL

SELECT
    r.risk_level,
    COUNT(*) AS admission_count,
    ROUND(AVG(a.length_of_stay), 2) AS average_length_of_stay
FROM readmission_risk r
JOIN admissions a
    ON r.admission_id = a.admission_id
GROUP BY r.risk_level
ORDER BY
    CASE r.risk_level
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END;


-- QUERY 30: FINAL HEALTHCARE ANALYTICS SUMMARY

SELECT
    (SELECT COUNT(*) FROM patients) AS total_patients,
    (SELECT COUNT(*) FROM admissions) AS total_admissions,
    (SELECT COUNT(*) FROM doctors) AS total_doctors,
    (SELECT COUNT(*) FROM vitals) AS total_vital_records,
    (SELECT COUNT(*) FROM treatments) AS total_treatments,
    (SELECT COUNT(*) FROM readmission_risk) AS risk_predictions,
    (
        SELECT COUNT(*)
        FROM readmission_risk
        WHERE risk_level = 'High'
    ) AS high_risk_admissions,
    (
        SELECT ROUND(
            SUM(actual_readmission_30_days) * 100.0 /
            COUNT(*),
            2
        )
        FROM readmission_risk
    ) AS actual_30_day_readmission_rate;