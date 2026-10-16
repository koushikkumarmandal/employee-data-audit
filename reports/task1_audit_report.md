# Task 1: Data Audit & Exploration Report
**Project:** CortexHR Employee Attrition & Attendance Monitoring  
**Audience:** Non-Technical Stakeholders, HR Leadership & Engineering Team  
**Author:** AI & Data Science Team  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Before any machine learning model is trained, the underlying data must be rigorously audited. In real-world enterprise environments, silent data defects—such as data leakage, recording typos, duplicate entries, and class imbalance—will quietly invalidate predictive models.

This audit profiled the master dataset (`02_CortexHR_AI_ML_Dataset.xlsx`, sheet `attrition_dataset`), which contains **2,389 records and 42 columns**. Our audit successfully uncovered:
- **6 exact duplicate rows** artificially inflated the dataset.
- **6 critical data leakage columns** that accidentally reveal future employee exits.
- **Data entry errors**, including negative tenure values, impossible attendance rates (> 100%), and 10x salary typos.
- **Categorical inconsistencies** in office locations (e.g., "Madras" vs. "Chennai", "Bangalore" vs. "Bengaluru").
- **Severe class imbalance**, where **85.4% of employees stayed** and only **14.6% left**, proving that raw "accuracy" is a deceptive metric.

Every identified issue has been recorded with an explicit operational decision that directly governs our Task 2 Reusable Feature Pipeline.

---

## 2. Dataset Overview & Data Dictionary

The master table reflects employee profiles recorded across various operating divisions.

| Dimension | Count / Type | Business Meaning |
| :--- | :--- | :--- |
| **Total Rows** | 2,389 raw (2,383 unique) | Employee observation records |
| **Total Columns** | 42 | Demographic, compensation, performance, attendance, and exit data |
| **Target Variable** | `left_within_12m` | Binary: `1` = Left the company within 12 months, `0` = Stayed |
| **Identifier Columns** | 4 columns | `row_id`, `employee_id`, `observation_date`, `source` |

### Key Column Groups:
1. **Identifiers (Drop):** Tracking keys like `employee_id` and `row_id` carry no generalizable signal and cause overfitting.
2. **Demographics & Work Setup:** `work_location`, `work_mode` (Remote/Hybrid/Onsite), `employment_type`, `highest_education`.
3. **Tenure & Experience:** `tenure_months`, `total_experience_years`, `previous_employers_count`.
4. **Compensation:** `annual_fixed_pay`, `comp_ratio`, `raises_received`, `last_raise_pct`, `months_since_last_raise`.
5. **Performance & Mobility:** `promotions_received`, `internal_transfers`, `last_performance_rating`, `last_engagement_score`, `manager_changes_12m`.
6. **Attendance & Workload:** `attendance_rate_3m`, `late_arrivals_3m`, `avg_overtime_hours_per_month_3m`, `avg_hours_worked_3m`, `wfh_share_3m`, `unpaid_leave_days_6m`, `paid_leave_days_6m`.

---

## 3. Data Leakage Assessment (The "Cheating" Columns)

### What is Data Leakage?
Data leakage happens when information from the future (or information that would only be available *after* the event occurs) is included in the model's training data. If a model learns from leaked features, it will score 99% accuracy in tests, but fail completely when deployed on current employees.

### Leaking Columns Identified & Drop Decisions:

| Column Name | Missing % | Why It Leaks the Answer | Decision |
| :--- | :---: | :--- | :--- |
| `exit_date` | 85.4% | Populated only after the employee officially separates. | **DROP IMMEDIATELY** |
| `resignation_date` | 86.6% | The formal resignation submission date reveals the decision. | **DROP IMMEDIATELY** |
| `exit_reason` | 85.4% | Describes why the employee left (e.g., "Better Offer", "Relocation"). | **DROP IMMEDIATELY** |
| `exit_interview_done`| 85.4% | Exit interviews occur exclusively during the offboarding window. | **DROP IMMEDIATELY** |
| `fnf_settlement_amount`| 85.4% | Full-and-Final payroll settlement paid out upon departure. | **DROP IMMEDIATELY** |
| `voluntary_left_within_12m`| 0.0% | Sub-category of the target itself (almost identical to label). | **DROP IMMEDIATELY** |

> **Audit Decision:** All six columns above are strictly banned from feature engineering and model training.
> **Safe Column Check:** `dept_attrition_rate_prior_12m` uses the 12 months *prior* to the observation snapshot; it is historical and safe to retain.

---

## 4. Missing Value Analysis & Decisions

Missing values were measured across all columns:

| Column | Missing Count | Missing % | Audit Decision & Rationale |
| :--- | :---: | :---: | :--- |
| `months_since_last_promotion` | 1,459 | 61.1% | **Do not drop.** Missing means the employee was *never promoted*. Impute with median and add flag. |
| `last_engagement_score` | 528 | 22.1% | **Impute with training median.** Non-participants in optional surveys. |
| `last_raise_pct` | 202 | 8.5% | **Impute with training median.** Employees who haven't received recent raise adjustments. |
| `last_performance_rating` | 182 | 7.6% | **Impute with training median.** Newer hires who missed the annual cycle. |
| `commute_distance_km` | 55 | 2.3% | **Impute with training median.** Randomly unrecorded home addresses. |
| `highest_education` | 15 | 0.6% | **Impute with category "Missing".** Preserves categorical completeness. |
| Exit columns (`exit_date`, etc.) | 2,040 | 85.4% | **Drop.** Missing corresponds to employees who stayed; dropped as leakage. |

---

### Think for Yourself: A Column has 40% Missing Values. What do you do?

> **Question:** *A column has 40% missing values. Do you drop the column, fill it, or add a “was missing” flag? Explain your choice.*

#### Expert Answer & Non-Technical Explanation:
**You should NOT automatically drop the column, nor should you blindly fill it with the average.** The right strategy depends on *why* the data is missing:

1. **Understand the Mechanism (Missing at Random vs. Missing Not at Random):**
   - If a column like `commute_distance` is missing 40% purely because a sensor or database form failed randomly, the missingness contains no useful business signal. Filling with median or dropping might be acceptable.
   - If a column like `bonus_received_last_year` or `disciplinary_actions_count` has 40% missing, the absence itself is the message (e.g., missing means zero bonus or no disciplinary action). In statistics, this is called *Missing Not At Random (MNAR)*.

2. **The Best Practice Solution: Impute + "Was Missing" Indicator Flag**
   - **Step A:** Create a new binary column called `column_was_missing` (1 if missing, 0 if present). This preserves the critical business signal that the data was absent.
   - **Step B:** Impute the original column with the training median or a constant placeholder so downstream mathematical models don't crash.
   - **Why this wins:** If the fact that an employee didn't submit an engagement survey correlates with their likelihood of quitting, the model can learn this pattern from the flag, while still utilizing the numerical values of employees who did submit it.

---

## 5. Outlier Detection: Comparing IQR vs. Z-Score

We audited three continuous operational features—`annual_fixed_pay` (Salary), `avg_hours_worked_3m` (Working Hours), and `tenure_months` (Tenure)—using two standard statistical techniques:
1. **Interquartile Range (IQR):** Flags values below $Q_1 - 1.5 \times \text{IQR}$ or above $Q_3 + 1.5 \times \text{IQR}$. Highly robust to skewed distributions.
2. **Z-Score (Standard Deviations):** Flags values with $|z| > 3.0$ standard deviations from the mean. Sensitive to extreme values.

### Comparison Results:

| Feature | IQR Outlier Flags | Z-Score Outlier Flags ($|z| > 3$) | Key Anomalies Found |
| :--- | :---: | :---: | :--- |
| **Annual Fixed Pay** | 94 rows | 26 rows | **Planted 10x Typos:** `EMP-0034`, `EMP-0433`, `EMP-0601` had an extra '0' typed at entry (e.g. ₹1.86 crore instead of ₹18.6 lakh). Senior executives have legitimate high pay. |
| **Average Hours Worked**| 48 rows | 15 rows | Employees averaging > 9.4 hours/day. Realistic overtime; no negative numbers. |
| **Tenure (Months)** | 74 rows | 42 rows | **Planted Negative Values:** 4 employees (`EMP-0248`, `EMP-0280`, `EMP-0568`, `EMP-0596`) have tenure between -1 and -5 months. Max tenure is 164 months (legitimate ~13.6 years). |

### Additional Operational Checks:
- **Attendance Rate > 100%:** Two records (`HIST-0048`, `HIST-0851`) showed `attendance_rate_3m = 1.12` (112%). An employee cannot attend more than 100% of scheduled work days.

### Audit Decisions on Outliers:
1. **Never delete rows simply because they are outliers:** High salaries and long tenures reflect vital senior staff whose attrition patterns differ from juniors.
2. **Fix negative tenure:** Clip `tenure_months` at 0 (or correct observation date offset).
3. **Fix impossible attendance rates:** Clip `attendance_rate_3m` at 1.0 (100%).
4. **Use Median & Robust Scaling:** Use median imputation in the pipeline so extreme salary values do not distort imputed averages.

---

## 6. Categorical Inconsistencies & Duplicates

1. **Exact Duplicate Rows:**
   - 6 exact duplicate rows were discovered appended at the bottom of the dataset (`EMP-0060`, `EMP-0237`, `EMP-0238`, `EMP-0325`, `EMP-0483`, `EMP-0570`).
   - **Decision:** Deduplicate immediately. Leaving them causes sample weight distortion and test leakage.
2. **Office Location (`work_location`) Standardization:**
   - The raw data had fragmented naming conventions:
     - `'Bangalore'` and `'bengaluru'` $\rightarrow$ Standardized to **`Bengaluru`**
     - `'Madras'` and `'chennai'` $\rightarrow$ Standardized to **`Chennai`**
     - `'delhi ncr'` and `'Delhi-NCR'` $\rightarrow$ Standardized to **`Delhi NCR`**
   - **Decision:** Normalize in `clean_data()` to avoid generating redundant one-hot columns.

---

## 7. Class Imbalance: Why Accuracy is a Trap

- **Employees Who Stayed:** 2,034 (85.4%)
- **Employees Who Left:** 349 (14.6%)

### The "Dummy Model" Paradox:
If an engineer builds a naive model that simply predicts every employee will stay, that model will boast an **85.4% accuracy**. However, it will detect **0% of departing employees**. For HR retention, such a model is 100% useless.

### Decision on Evaluation Metrics:
1. **Primary Metric: RECALL.** Missing an employee who leaves is costly (replacement cost = 50%–150% of annual salary). Reaching out to an employee who was going to stay anyway costs almost nothing.
2. **Secondary Metrics:** F1-Score (balance of precision and recall) and ROC-AUC (ranking power across all decision thresholds).
3. **Algorithmic Balancing:** Use class weighting (`class_weight='balanced'`) during model training to penalize the model heavily whenever it misses a departing employee.

---

## 8. Master Decision Matrix for Task 2 Reusable Pipeline

| Issue Category | Specific Finding | Pipeline Implementation Rule |
| :--- | :--- | :--- |
| **Duplicates** | 6 duplicate rows | Deduplicate raw data in `clean_data()`. |
| **Data Leakage** | 6 exit/settlement columns | Permanently drop from feature space before splitting. |
| **IDs** | `employee_id`, `row_id`, `observation_date` | Drop from feature space before model input. |
| **Typos: Locations**| Multiple casings/synonyms for Indian cities | Clean with mapping dictionary before One-Hot Encoding. |
| **Typos: Tenure** | 4 records with negative tenure | Clip `tenure_months` to a minimum of 0. |
| **Typos: Attendance**| Records with attendance > 100% | Clip `attendance_rate_3m` to a maximum of 1.0. |
| **Missing Numbers** | 0.6% to 61% missing across numeric features | Median imputation fitted strictly on training data. |
| **Missing Text** | Minor missing values in education | Impute with constant label "Missing" $\rightarrow$ One-Hot Encode. |
| **Feature Engineering**| Pay parity is a known turnover driver | Build custom `SalaryVsDeptMedian` transformer. |
| **Model Leakage Prevention**| Feature statistics computed on whole dataset | Split train/test (80/20 stratified) **before** fitting any pipeline step. |

---

## 9. Conclusion & Next Steps

Task 1 Data Audit is complete and documented. The rules above have been codified into [`src/features/build_features.py`](file:///c:/Users/Koushik%20Mandal/OneDrive/Desktop/employee-data-audit/src/features/build_features.py) for Task 2, and all baseline models in Task 3 and anomaly detectors in Task 4 now strictly adhere to these verified standards.

