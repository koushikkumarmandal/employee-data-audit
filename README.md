# CortexHR — AI/ML Developer Submission 
**WorkforceOS · Section 2 · The AI Intelligence Layer**  
**Tasks 1 to 4: Data Foundations, Reusable Pipeline, Baseline Models & Attendance Anomaly Detection**

---

## 1. Submission & Repository Links

* **Official GitHub Repository:** [https://github.com/koushikkumarmandal/employee-data-audit](https://github.com/koushikkumarmandal/employee-data-audit)
* **Release Version Tag:** [`v1.0.0`](https://github.com/koushikkumarmandal/employee-data-audit/releases/tag/v1.0.0)
* **Packaged Submission Archive (ZIP):** [`CortexHR_Phase1_Submission.zip`](./CortexHR_Phase1_Submission.zip) (Contains the complete standalone bundle of reports, notebooks, models, code, and Dockerfile ready for review portal upload).

### Task Branches for Pull Requests:
* **Task 1 Branch:** [`task-1-data-audit`](https://github.com/koushikkumarmandal/employee-data-audit/tree/task-1-data-audit)
* **Task 2 Branch:** [`task-2-feature-pipeline`](https://github.com/koushikkumarmandal/employee-data-audit/tree/task-2-feature-pipeline)
* **Task 3 Branch:** [`task-3-attrition-baselines`](https://github.com/koushikkumarmandal/employee-data-audit/tree/task-3-attrition-baselines)
* **Task 4 Branch:** [`task-4-attendance-anomalies`](https://github.com/koushikkumarmandal/employee-data-audit/tree/task-4-attendance-anomalies)

---

## 2. Project Overview

This repository contains the completed, tested, and containerized implementation for **Phase 1 (Tasks 1 to 4)** of the CortexHR AI/ML project.

* **Task 1 — Data Audit & Exploration:** Audited the master dataset (`2,389` rows, `42` columns), cataloged 6 data leakage columns, 6 duplicate rows, impossible values, typos, and class imbalance.
* **Task 2 — Reusable Feature Pipeline:** Built the `cortexhr_features` package with 6 custom engineered features (`salary_vs_dept_median`, `tenure_years`, `raise_overdue`, `unpaid_per_month`, `attendance_gap`, `overtime_load`). Fits on training data only with zero data leakage.
* **Task 3 — Attrition Baseline Models:** Plugs the Task 2 pipeline directly into Dummy, Logistic Regression, and Decision Tree baselines. Evaluated with 5-fold Stratified Cross-Validation and locked 20% test set. Threshold tuned to `0.490` for $\ge 60\%$ recall.
* **Task 4 — Attendance Anomaly Detection:** Trained Isolation Forest with 300 trees on unpaid-based attendance features (protecting maternity leave). Tuned threshold to 95th percentile, achieving **100% catch rate** with **5.0% false-alarm rate**.

---

## 3. Handoff Package (Required Reports)

As specified in Section 4 of the *CortexHR Developer Guide*, the handoff documents are published in [`reports/`](./reports/):

| Report Name | Local Path | Direct GitHub Link | Description |
| :--- | :--- | :--- | :--- |
| **Data Audit Report** | [`reports/data-audit.md`](./reports/data-audit.md) | [View on GitHub](https://github.com/koushikkumarmandal/employee-data-audit/blob/main/reports/data-audit.md) | Full audit findings, decisions table, 40% missingness explanation, and "What I missed" comparison. |
| **Baseline Report** | [`reports/baseline-report.md`](./reports/baseline-report.md) | [View on GitHub](https://github.com/koushikkumarmandal/employee-data-audit/blob/main/reports/baseline-report.md) | 5-fold CV & locked test set metrics (ROC-AUC, PR-AUC, F1, Recall), threshold tuning, and HR trade-off analysis. |
| **Phase 1 Review** | [`reports/phase-1-review.md`](./reports/phase-1-review.md) | [View on GitHub](https://github.com/koushikkumarmandal/employee-data-audit/blob/main/reports/phase-1-review.md) | Official Phase 1 Review following the Section 10 guide template. |

---

## 4. Assumptions (CortexHR Guide Requirement)

Per developer guide rules, all assumptions made during development are documented here:
1. **Missing Data Policy (> 2% Missing):** For numeric columns with $> 2\%$ missing values (`months_since_last_promotion`, `last_engagement_score`, `last_raise_pct`, `last_performance_rating`, `commute_distance_km`), values are filled with the training median and a `"was_missing"` binary indicator is generated to preserve predictive business signal.
2. **Data Leakage Exclusion:** Columns generated during or after employee exit (`exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount`, `voluntary_left_within_12m`) are completely dropped prior to model input. `dept_attrition_rate_prior_12m` is retained as it reflects historical performance prior to the observation window.
3. **Planted Dirt Sanitation:**
   - Negative tenure values (`EMP-0248`, `0280`, `0568`, `0596`) clipped to `0`.
   - Attendance rate $> 1.0$ (`HIST-0048`, `0851`) clipped to `1.0`.
   - City synonyms (`Bangalore` $\rightarrow$ `Bengaluru`, `Madras` $\rightarrow$ `Chennai`, `delhi ncr` / `Delhi-NCR` $\rightarrow$ `Delhi NCR`) mapped cleanly before one-hot encoding.
4. **Maternity & Legitimate Leave Protection:** In Task 4, anomaly detection features exclude `paid_leave_days` so employees taking legitimate extended paid leave (such as maternity leave) are not penalized or flagged as anomalous.

---

## 4. Key Results Summary

### Attrition Baseline Models (Task 3)
* **Dataset:** 2,383 clean records (85.4% Stayers, 14.6% Leavers).
* **Test Set:** 477 rows (20% stratified, locked until final evaluation).

| Model | CV ROC-AUC | CV PR-AUC | Test Accuracy | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Always Stays)** | 0.500 | 0.146 | 85.3% | 0.0% | 0.000 | 0.500 |
| **Logistic Regression (Standard)** | 0.709 | 0.312 | 85.7% | 12.9% | 0.209 | 0.683 |
| **Logistic Regression (Balanced)** | **0.709** | **0.309** | 66.9% | **58.6%** | **0.342** | **0.680** |
| **Decision Tree (Depth 4)** | 0.622 | 0.237 | 85.1% | 5.7% | 0.101 | 0.621 |
| **Decision Tree (Balanced)** | 0.634 | 0.228 | 59.5% | 45.7% | 0.249 | 0.591 |

* **Tuned Threshold:** `0.490` achieves **60.0% Recall** with **24.4% Precision** for HR proactive intervention.

### Attendance Anomaly Detection (Task 4)
* **Model:** Isolation Forest (300 trees, seed=42).
* **Dataset:** 7,339 monthly attendance records (219 synthetic test anomalies across 3 types).
* **Chosen Threshold:** `0.02730` (95th percentile).
  * **Catch Rate (Recall):** **100.0%** (219 / 219 caught).
  * **False-Alarm Rate:** **5.0%** (within the 5% operational ceiling to prevent alert fatigue).

---

## 5. Quickstart & Verification Commands

### Run Locally:
```powershell
# 1. Run unit tests
pytest

# 2. Run Task 2 (Feature Pipeline)
python -m cortexhr_features.build

# 3. Run Task 3 (Attrition Baseline Models)
python -m src.models.attrition_baseline

# 4. Run Task 4 (Attendance Anomaly Detection)
python -m src.models.attendance_anomaly
```

### Run with Docker:
```powershell
# Start the live container on port 8888
docker run -d -p 8888:8888 --name employee-app employee-data-audit

# Run tests inside Docker
docker exec -it employee-app pytest

# Open JupyterLab in browser
# http://localhost:8888/lab?token=student
```

---

## 6. Project Structure

```text
employee-data-audit/
├── cortexhr_features/          # Task 2 packaged feature pipeline
│   ├── __init__.py
│   └── build.py
├── data/
│   ├── raw/                    # Master Excel dataset (02_CortexHR_AI_ML_Dataset.xlsx)
│   └── processed/              # Processed model inputs (X_train, X_test, etc.)
├── models/                     # Saved joblib pipelines and metadata
├── models_store/               # Model registry (<name>/<version>/)
│   ├── attendance_anomaly/1.0.0/
│   ├── attrition_baseline/1.0.0/
│   └── features/1.0.0/
├── notebooks/                  # Interactive walkthrough notebooks
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_pipeline.ipynb
│   ├── 03_attrition_baseline_models.ipynb
│   └── 04_attendance_anomaly.ipynb
├── reports/                    # Required handoff markdown reports and plots
│   ├── data-audit.md
│   ├── baseline-report.md
│   ├── task4_report.md
│   ├── phase-1-review.md
│   ├── task3_confusion_matrix.png
│   ├── task3_roc_curve.png
│   ├── task4_precision_recall.png
│   └── task4_false_alarm_rate.png
├── src/             
├── tests/
├── Dockerfile 
├── pyproject.toml 
├── requirements.txt 
└── README.md 
```

