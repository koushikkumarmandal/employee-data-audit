# TASK 3: ATTRITION BASELINE MODELS REPORT
Prepared for: HR Leadership & Data Team
Status: Completed

--------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
The objective is to identify employees at risk of leaving within the next 12 months.
Because 85.4% of employees stay and only 14.6% leave, class imbalance is present.

- A naive "Dummy" model that simply predicts everyone stays achieves 85.4% accuracy,
  yet catches ZERO leaving employees (0% recall). This proves accuracy is deceptive.
- By integrating our Task 2 reusable pipeline, we tested Logistic Regression and
  Decision Tree models with and without class weights.
- The best baseline model is: **Logistic Regression Balanced**.
  On the locked test set (477 employees never seen during training), it caught
  approximately 58.6% of all leavers
  with a ROC-AUC score of 0.680.

--------------------------------------------------------------------------------
2. 5-FOLD CROSS-VALIDATION RESULTS (TRAINING DATA ONLY)
--------------------------------------------------------------------------------
                       Model  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  precision_mean  precision_std  recall_mean  recall_std  f1_mean  f1_std
         Logistic Regression         0.709        0.027        0.312       0.037           0.443          0.163        0.093       0.024    0.152   0.037
Logistic Regression Balanced         0.709        0.022        0.309       0.036           0.265          0.026        0.609       0.023    0.369   0.027
               Decision Tree         0.622        0.044        0.237       0.047           0.361          0.078        0.101       0.044    0.152   0.052
      Decision Tree Balanced         0.634        0.028        0.228       0.030           0.223          0.023        0.509       0.052    0.310   0.030

--------------------------------------------------------------------------------
3. LOCKED TEST SET RESULTS (UNSEEN 20% EVALUATION)
--------------------------------------------------------------------------------
                       Model  Accuracy  Precision  Recall    F1  ROC_AUC  PR_AUC
        Dummy (Always Stays)     0.853      0.000   0.000 0.000    0.500   0.146
         Logistic Regression     0.857      0.562   0.129 0.209    0.683   0.336
Logistic Regression Balanced     0.669      0.241   0.586 0.342    0.680   0.341
               Decision Tree     0.851      0.444   0.057 0.101    0.621   0.211
      Decision Tree Balanced     0.595      0.171   0.457 0.249    0.591   0.206

--------------------------------------------------------------------------------
4. THINK FOR YOURSELF: PRECISION VS. RECALL FOR HR
--------------------------------------------------------------------------------
Question:
"HR would rather catch more leavers even if some stayers are flagged by mistake.
Which metric matters most, precision or recall? Explain."

Answer:
**RECALL MATTERS MOST FOR THIS HR USE CASE.**

Here is the business reason in plain words:
1. **Cost of a False Negative (Missing a real leaver):**
   Replacing an employee who leaves unexpectedly costs between 50% and 150% of their
   annual salary (recruiting fees, onboarding, training, and lost team productivity).
   If the model misses someone who leaves (False Negative), that cost is locked in.

2. **Cost of a False Positive (Flagging a stayer by mistake):**
   If the model flags a loyal stayer who wasn't actually planning to leave, the action
   taken is a supportive 1-on-1 check-in, reviewing career growth, or offering training.
   Having a positive career conversation with a good employee has very low cost and
   usually boosts morale anyway!

3. **Conclusion:**
   Because missing a leaver is far more costly than an unnecessary check-in, HR should
   prioritize **High Recall** (catching as many potential leavers as possible) while
   maintaining acceptable precision so HR's time is not overwhelmed.

--------------------------------------------------------------------------------
5. NEXT STEPS
--------------------------------------------------------------------------------
All future advanced models (Random Forests, Gradient Boosting, neural nets) must beat
this baseline of Logistic Regression Balanced (F1 = 0.342, ROC-AUC = 0.680).
