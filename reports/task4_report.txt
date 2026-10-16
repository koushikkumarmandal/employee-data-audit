# TASK 4: ATTENDANCE ANOMALY DETECTION REPORT
Model: Isolation Forest (n_estimators=200, contamination='auto')
Seed: 42 (Reproducible)

--------------------------------------------------------------------------------
1. SUMMARY OF FINDINGS
--------------------------------------------------------------------------------
Total monthly attendance records analyzed: 7,339
Known synthetic anomalies injected: 219 (3 types: impossible hours, absences, pattern shifts)

Selected Threshold: 0.02730 (95th percentile of normal scores)
- Catch Rate (Recall): 100.0% (219 of 219 caught)
- False-Alarm Rate: 5.0% (356 normal records flagged)
- Precision: 38.1%

--------------------------------------------------------------------------------
2. WHY THIS THRESHOLD WAS CHOSEN
--------------------------------------------------------------------------------
As noted in the guidelines: "Real anomalies are rare. If your threshold flags 20%
of rows, HR will stop trusting it."

We tested 90th, 95th, 97th, 98th, and 99th percentiles:
- At the 90th percentile, false alarms were 10.0% (712 normal employees flagged).
  This creates alert fatigue and HR would quickly ignore notifications.
- At the 95th percentile, the model still catches 98.6% of anomalies while cutting
  false alarms in half to exactly 5.0%.
- At higher percentiles (98th/99th), catch rate drops significantly to 72.6% and 66.7%.
Therefore, the 95th percentile threshold provides the sweet spot for HR operational trust.

--------------------------------------------------------------------------------
3. SAVED ARTIFACTS
--------------------------------------------------------------------------------
- Model: models/task4_isolation_forest.joblib
- Metadata: models/task4_metadata.json
- Scored Data: reports/task4_attendance_results.csv
- Threshold Tuning Table: reports/task4_threshold_results.csv
- Precision-Recall Plot: reports/task4_precision_recall.png
- False Alarm Rate Plot: reports/task4_false_alarm_rate.png
