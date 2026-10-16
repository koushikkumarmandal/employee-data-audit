# Data Audit & Exploration Report (Task 1)

**System:** CortexHR — AI Intelligence Layer (WorkforceOS Section 2)  
**Author:** AI/ML Team  
**Master Dataset:** `02_CortexHR_AI_ML_Dataset.xlsx` (Sheet: `attrition_dataset`)  
**Target Variable:** `left_within_12m`  
**Status:** Completed & Validated  



## 1. Size and Types

The master training dataset contains:
- **Total Rows:** 2,389 employee observations (383 real employee historical snapshots + 2,000 synthetic extension rows + 6 duplicate rows).
- **Total Columns:** 42 features covering demographics, compensation, job role, attendance, performance, and exit records.
- **Data Types Breakdown:**
  - `int64` (12 columns): e.g., `job_level`, `tenure_months`, `annual_fixed_pay`, `raises_received`, `left_within_12m`.
  - `float64` (15 columns): e.g., `total_experience_years`, `comp_ratio`, `attendance_rate_3m`, `last_engagement_score`.
  - `object` (text/categorical, 11 columns): e.g., `department_id`, `role_id`, `work_location`, `work_mode`, `employment_type`.
  - `datetime64[ns]` (4 columns): `observation_date`, `exit_date`, `resignation_date`, `source`.

---

## 2. Missing Values Analysis & Default Handling

Missing values were evaluated across every column:

| Column Name | Missing Count | Missing % | Handling Rule |
| :--- | :---: | :---: | :--- |
| `months_since_last_promotion` | 1,459 | 61.1% | Impute with median + add **`was_missing` flag** (unpromoted employees). |
| `last_engagement_score` | 528 | 22.1% | Impute with training median + add **`was_missing` flag** (> 2% missing). |
| `last_raise_pct` | 202 | 8.5% | Impute with training median + add **`was_missing` flag** (> 2% missing). |
| `last_performance_rating` | 182 | 7.6% | Impute with training median + add **`was_missing` flag** (> 2% missing). |
| `commute_distance_km` | 55 | 2.3% | Impute with training median + add **`was_missing` flag** (> 2% missing). |
| `highest_education` | 15 | 0.6% | Categorical imputation with `"Missing"` category (<= 2% missing). |
| Exit columns (`exit_date`, `resignation_date`, etc.) | 2,040 | 85.4% | **Drop completely** (data leakage candidates; missing = employee stayed). |

### Default Policy for Missing Values (> 2% Missing):
As established in the CortexHR Developer Guide:
1. For numeric columns with **more than 2% missing values**, we fill with the training set **median** and generate a binary `"was_missing"` indicator flag.
2. The flag retains the predictive business signal (e.g., employees who did not take an optional engagement survey or have never been promoted), while median imputation prevents models from failing due to NaNs.

---

## 3. Outlier Analysis (IQR vs. Z-Score)

Continuous features (`annual_fixed_pay`, `avg_hours_worked_3m`, `tenure_months`) were inspected using two methods:
1. **IQR Method:** Outliers outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
2. **Z-Score Method:** Outliers where $|z| > 3.0$ standard deviations.

| Feature | IQR Outlier Flags | Z-Score Flags ($|z|>3$) | Analysis & Root Cause |
| :--- | :---: | :---: | :--- |
| `annual_fixed_pay` | 94 | 26 | **Planted 10x Typos:** `EMP-0034`, `EMP-0433`, `EMP-0601` have pay 10x too high due to an extra '0' entered during manual entry (comp_ratio is normal ~1.0). Executive salaries are legitimate high values. |
| `avg_hours_worked_3m` | 48 | 15 | Long daily hours (> 9.4 hrs/day). Reflects heavy project deadlines; valid operational variance. |
| `tenure_months` | 74 | 42 | **Negative Tenure Typos:** 4 rows have negative tenure (-1 to -5 months). Max tenure is 164 months (valid 13.6 years). |

---

## 4. Class Balance

* **Total Clean Records:** 2,383
* **Stayers (`left_within_12m = 0`):** 2,034 (**85.4%**)
* **Leavers (`left_within_12m = 1`):** 349 (**14.6%**)

**Imbalance Insight:** The class ratio is roughly 6:1. An untuned model that predicts "nobody leaves" achieves **85.4% accuracy**, yet catches 0 leavers. Therefore, accuracy is deceptive; models must be evaluated on **Recall, PR-AUC, F1, and ROC-AUC** with balanced class weights.

---

## 5. Duplicate Rows

* **Planted Duplicates:** 6 exact duplicate records were identified appended at the end of the sheet:
  `EMP-0325`, `EMP-0238`, `EMP-0483`, `EMP-0060`, `EMP-0237`, `EMP-0570`.
* **Action:** Dropped immediately during initial data cleaning.

---

## 6. Inconsistent Category Spellings

Office locations in `work_location` exhibited multiple non-standard casing and legacy naming variants:
* `'Bangalore'`, `'bengaluru'`, `'BENGALURU'` $\rightarrow$ Standardized to **`Bengaluru`** (896 records)
* `'Madras'`, `'chennai'` $\rightarrow$ Standardized to **`Chennai`** (446 records)
* `'delhi ncr'`, `'Delhi-NCR'` $\rightarrow$ Standardized to **`Delhi NCR`** (139 records)

---

## 7. Impossible Values

1. **Negative Tenure:** `EMP-0248` (-4 mos), `EMP-0280` (-5 mos), `EMP-0568` (-2 mos), `EMP-0596` (-1 mos). Caused by observation date preceding join date typo. Clipped to `0`.
2. **Attendance Rate > 100%:** `HIST-0048` (1.12) and `HIST-0851` (1.12). Clipped to `1.0` (100%).

---

## 8. Leakage Candidates (Columns Existing Only After Exit)

The following 6 columns contain information generated exclusively during or after employee offboarding:
1. `exit_date` (Date employee departed)
2. `resignation_date` (Date resignation was tendered)
3. `exit_reason` (Self-reported departure reason)
4. `exit_interview_done` (Offboarding interview completion status)
5. `fnf_settlement_amount` (Full and final settlement payout amount)
6. `voluntary_left_within_12m` (Duplicate indicator of the target)

**Decision:** Permanently excluded from feature engineering. Retaining any of these causes near 1.0 ROC-AUC (fake leakage score).

---

## 9. Master Decisions Table

| # | Data Problem | Affected Columns / Rows | Action Decision | Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Exact Duplicates | 6 rows at end of sheet | **Drop rows** | Prevent duplicate sample weighting. |
| 2 | Data Leakage | 6 offboarding columns | **Drop columns** | Future information unavailable at prediction time. |
| 3 | Identifiers | `row_id`, `employee_id`, `observation_date` | **Drop columns** | Prevent overfitting on arbitrary database IDs. |
| 4 | City Spelling Variants | `work_location` ('Bangalore', 'Madras', 'Delhi-NCR') | **Standardize names** | Prevent redundant One-Hot columns. |
| 5 | Negative Tenure | `tenure_months` (4 rows) | **Clip to 0** | Impossible negative time; entry typo. |
| 6 | Attendance > 100% | `attendance_rate_3m` (2 rows) | **Clip to 1.0** | Impossible attendance above 100%. |
| 7 | Missing Numeric (> 2%) | `months_since_last_promotion`, `engagement`, `raise_pct`, `rating`, `commute` | **Median Impute + Was-Missing Flag** | Preserve missingness signal while handling NaNs. |
| 8 | Missing Categorical (<= 2%)| `highest_education` (15 rows) | **Fill with "Missing"** | Retain row completeness in One-Hot encoding. |
| 9 | High Salary Typos | `annual_fixed_pay` (`EMP-0034`, `0433`, `0601`) | **Robust Median Imputer** | Median imputation is impervious to 10x typos. |
| 10 | Class Imbalance | `left_within_12m` (85.4% vs 14.6%) | **Balanced class weights** | Penalize false negatives heavily for leavers. |

---

## 10. What I Missed (Answer Key Comparison)

After completing our manual audit, we cross-checked against `answer_key_planted_dirt` and `answer_key_notes`:
1. **Confirmed Catches:**
   - All 6 duplicate rows (`EMP-0060`, `0237`, `0238`, `0325`, `0483`, `0570`) caught.
   - All 4 negative tenure rows (`EMP-0248`, `0280`, `0568`, `0596`) caught.
   - All 2 impossible attendance records (`HIST-0048`, `HIST-0851`) caught.
   - All 3 10x salary typos (`EMP-0034`, `0433`, `0601`) caught.
   - All location variants (`Madras`, `Bangalore`, `bengaluru`, `delhi ncr`, `Delhi-NCR`) identified.
2. **Key Nuance Verified:**
   - `dept_attrition_rate_prior_12m` looks suspicious at first glance, but the note confirms it is historical (calculated over the 12 months *prior* to the observation date), making it completely safe and leak-free.

