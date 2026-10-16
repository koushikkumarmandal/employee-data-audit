# Outlier Detection Findings

## Objective
Identify unusual salary, working hours, and tenure values
using IQR and Z-score methods.

## Results
- Salary: 94 IQR flags and 26 Z-score flags.
- Working hours: 48 IQR flags and 15 Z-score flags.
- Tenure: 74 IQR flags and 42 Z-score flags.

## Decisions
1. Salary: Inspect unusually high salaries before making changes.
2. Working hours: Check flagged records for unusual patterns
   and possible data-entry errors.
3. Tenure: Investigate unusually long tenure and all negative
   tenure values.
4. Do not delete outliers automatically.
5. Compare both methods because they identify different records.

## Conclusion
IQR identified more unusual values than Z-score for all
three selected features. This suggests that the methods
have different sensitivities to the distribution of the data.
All flagged values should be investigated before deciding
whether to correct, retain, or remove them.