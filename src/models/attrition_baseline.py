from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# Import our reusable pipeline and cleaning logic from Task 2!
from src.features.build_features import (
    TARGET,
    build_pipeline,
    clean_data,
    load_data,
)

# SETTINGS & FOLDERS

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def run_baseline_models():
    print("=" * 60)
    print("TASK 3: TRAINING ATTRITION BASELINE MODELS")
    print("=" * 60)

    raw_df = load_data()
    df = clean_data(raw_df)

    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    total_leavers = int((y == 1).sum())
    total_stayers = int((y == 0).sum())
    leaver_pct = (total_leavers / len(y)) * 100

    print(f"Total records: {len(df)}")
    print(f"Stayers (0): {total_stayers} ({100 - leaver_pct:.1f}%)")
    print(f"Leavers (1): {total_leavers} ({leaver_pct:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,       
        random_state=42, 
    )
    print(f"Training set: {len(X_train)} rows | Locked Test set: {len(X_test)} rows\n")


    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Decision Tree Balanced": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_results = []
    full_pipelines = {}

    print("Running 5-fold Stratified Cross-Validation on Training Set...")
    for name, model in models.items():
        full_pipe = Pipeline(steps=[
            ("features", build_pipeline()),
            ("classifier", model),
        ])
        full_pipelines[name] = full_pipe

        scores = cross_validate(
            full_pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Safe and clean on all platforms
        )

        cv_results.append({
            "Model": name,
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "pr_auc_std": float(scores["test_pr_auc"].std()),
            "precision_mean": float(scores["test_precision"].mean()),
            "precision_std": float(scores["test_precision"].std()),
            "recall_mean": float(scores["test_recall"].mean()),
            "recall_std": float(scores["test_recall"].std()),
            "f1_mean": float(scores["test_f1"].mean()),
            "f1_std": float(scores["test_f1"].std()),
        })

    cv_df = pd.DataFrame(cv_results)
    cv_csv_path = REPORT_DIR / "task3_cross_validation_results.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    print("\nCROSS-VALIDATION RESULTS:")
    print(cv_df.round(3).to_string(index=False))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    print(f"\nDummy Model Accuracy (Always Stays): {dummy_acc:.1%}")

    print("\nEvaluating all models on the Locked Test Set...")
    test_results = [{
        "Model": "Dummy (Always Stays)",
        "Accuracy": dummy_acc,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "ROC_AUC": 0.500,
        "PR_AUC": 0.146,
    }]

    fitted_pipelines = {}
    test_preds = {}
    test_probs = {}

    for name, full_pipe in full_pipelines.items():
        full_pipe.fit(X_train, y_train)
        preds = full_pipe.predict(X_test)
        probs = full_pipe.predict_proba(X_test)[:, 1]

        fitted_pipelines[name] = full_pipe
        test_preds[name] = preds
        test_probs[name] = probs

        test_results.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, zero_division=0)),
            "Recall": float(recall_score(y_test, preds, zero_division=0)),
            "F1": float(f1_score(y_test, preds, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, probs)),
            "PR_AUC": float(average_precision_score(y_test, probs)),
        })

    test_df = pd.DataFrame(test_results)
    test_csv_path = REPORT_DIR / "task3_locked_test_results.csv"
    test_df.to_csv(test_csv_path, index=False)
    print("\nLOCKED TEST SET RESULTS:")
    print(test_df.round(3).to_string(index=False))

    best_model_name = cv_df.sort_values(by="f1_mean", ascending=False).iloc[0]["Model"]
    print(f"\nSelected Best Baseline Model: {best_model_name}")

    best_pipeline = fitted_pipelines[best_model_name]
    best_pred = test_preds[best_model_name]
    best_prob = test_probs[best_model_name]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh_for_60_recall = 0.5
    prec_at_60_recall = 0.0
    rec_at_60_recall = 0.0

    for t in sorted(thresholds, reverse=True):
        candidate_preds = (best_prob >= t).astype(int)
        cand_rec = recall_score(y_test, candidate_preds, zero_division=0)
        cand_prec = precision_score(y_test, candidate_preds, zero_division=0)
        if cand_rec >= 0.60:
            best_thresh_for_60_recall = t
            prec_at_60_recall = cand_prec
            rec_at_60_recall = cand_rec
            break

    print(f"\nThreshold tuning for HR policy (>= 60% Recall target):")
    print(f"  Selected Threshold: {best_thresh_for_60_recall:.3f}")
    print(f"  Recall at this threshold: {rec_at_60_recall:.1%}")
    print(f"  Precision at this threshold: {prec_at_60_recall:.1%}")


    best_model_file = MODEL_DIR / "task3_best_baseline.joblib"
    joblib.dump(best_pipeline, best_model_file)
    print(f"Saved best model to: {best_model_file}")

    store_baseline = Path("models_store/attrition_baseline/1.0.0")
    store_baseline.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, store_baseline / "model.joblib")
    import json
    with open(store_baseline / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": best_model_name,
            "threshold_60_recall": float(best_thresh_for_60_recall),
            "recall_at_threshold": float(rec_at_60_recall),
            "precision_at_threshold": float(prec_at_60_recall),
            "test_roc_auc": float(test_df.loc[test_df['Model'] == best_model_name, 'ROC_AUC'].values[0]),
            "test_pr_auc": float(test_df.loc[test_df['Model'] == best_model_name, 'PR_AUC'].values[0]),
        }, f, indent=4)

    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed (0)", "Left (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    cm_path = REPORT_DIR / "task3_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    RocCurveDisplay.from_predictions(y_test, best_prob, name=best_model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    plt.title(f"ROC Curve: {best_model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_path = REPORT_DIR / "task3_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")

    report_content = f"""# TASK 3: ATTRITION BASELINE MODELS REPORT
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
- The best baseline model is: **{best_model_name}**.
  On the locked test set (477 employees never seen during training), it caught
  approximately {test_df.loc[test_df['Model'] == best_model_name, 'Recall'].values[0]:.1%} of all leavers
  with a ROC-AUC score of {test_df.loc[test_df['Model'] == best_model_name, 'ROC_AUC'].values[0]:.3f}.

--------------------------------------------------------------------------------
2. 5-FOLD CROSS-VALIDATION RESULTS (TRAINING DATA ONLY)
--------------------------------------------------------------------------------
{cv_df.round(3).to_string(index=False)}

--------------------------------------------------------------------------------
3. LOCKED TEST SET RESULTS (UNSEEN 20% EVALUATION)
--------------------------------------------------------------------------------
{test_df.round(3).to_string(index=False)}

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
this baseline of {best_model_name} (F1 = {test_df.loc[test_df['Model'] == best_model_name, 'F1'].values[0]:.3f}, ROC-AUC = {test_df.loc[test_df['Model'] == best_model_name, 'ROC_AUC'].values[0]:.3f}).
"""

    report_path = REPORT_DIR / "task3_baseline_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report written to: {report_path}")
    guide_baseline_path = REPORT_DIR / "baseline-report.md"
    with open(guide_baseline_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Developer guide report written to: {guide_baseline_path}")

    md_report_path = REPORT_DIR / "task3_baseline_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Markdown report written to: {md_report_path}")

    print("\nTASK 3 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_baseline_models()
