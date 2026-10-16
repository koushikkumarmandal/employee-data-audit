# Missing Value Analysis

## Objective
Identify missing values in every column and decide
how to handle them.

## Method
I used pandas isnull() and sum() to count missing
values. I calculated the missing percentage for
each column.

## Decisions
1. Numeric columns: consider median imputation.
2. Categorical columns: use Unknown where appropriate.
3. Columns with high missing percentages: investigate
   before dropping.
4. Exit-related columns: exclude from prediction features.
5. Missing target labels: exclude from supervised training.

## Conclusion
Missing values must be handled according to the meaning
of each column. I will not fill every missing value
with zero or remove columns without investigation.