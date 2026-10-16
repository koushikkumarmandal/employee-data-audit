
# Task 1: Initial Dataset Exploration

## Objective
Understand the employee dataset before performing
the detailed data audit.

## Dataset
The workbook contains multiple sheets. For this task,
I selected the attrition_dataset sheet because it
contains employee information used for attrition analysis.

## Initial Findings
- Total records: 2,389
- Total columns: 42
- Duplicate employee IDs: 6 repeated occurrences
- Missing values exist in several columns.
- The dataset contains numeric, text, and date columns.
- left_within_12m is the target column.

## Important Note
Repeated employee IDs must be investigated before
removing any records. Different rows may represent
different observations of the same employee.

The initial exploration is complete. Missing values,
outliers, class imbalance, and data leakage will be
examined in the following audit steps.
