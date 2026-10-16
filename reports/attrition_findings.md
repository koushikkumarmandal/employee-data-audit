
# Attrition Class Imbalance

## Objective
Calculate the percentage of employees who left
within 12 months and identify class imbalance.

## Findings
- Total records: 2,389
- Employees who stayed: 2,040 (85.39%)
- Employees who left: 349 (14.61%)

## Decision
The dataset is imbalanced because most employees
belong to the stayed class.

Accuracy alone will not be enough to evaluate
an attrition prediction model.

I will also use precision, recall, F1-score,
ROC-AUC, and a confusion matrix.

## Conclusion
The model must be evaluated carefully to ensure
that it can identify employees who leave, not
just predict that most employees will stay.
