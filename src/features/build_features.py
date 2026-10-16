import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PIPELINE_VERSION = "features-1.0.0"   # version tag


DATA_FILE = "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"

OUTPUT_DIR = "data/processed"   #  finished data goes
MODEL_DIR = "models"            #  saved pipeline goes

TARGET = "left_within_12m"     

# Columns that know the FUTURE
LEAKAGE_COLUMNS = [
    "exit_date",                  
    "resignation_date",           
    "exit_reason",              
    "exit_interview_done",       
    "fnf_settlement_amount",      
    "voluntary_left_within_12m",  
]


ID_COLUMNS = ["row_id", "employee_id", "observation_date", "source"]

# Inputs
NUMERIC_FEATURES = [
    "attendance_rate_3m",               
    "late_arrivals_3m",                 
    "avg_overtime_hours_per_month_3m",   
    "tenure_months",                     
    "salary_vs_dept_median",             
    "tenure_years",                     
    "raise_overdue",                     
    "unpaid_per_month",              
    "attendance_gap",                    
    "overtime_load",                    
    "months_since_last_raise",         
    "job_level",
    "total_experience_years",
    "previous_employers_count",
    "commute_distance_km",
    "annual_fixed_pay",
    "comp_ratio",
    "raises_received",
    "last_raise_pct",
    "promotions_received",
    "internal_transfers",
    "last_performance_rating",
    "last_engagement_score",
    "manager_changes_12m",
    "training_hours_12m",
    "avg_hours_worked_3m",
    "wfh_share_3m",
    "unpaid_leave_days_6m",
    "paid_leave_days_6m",
    "dept_attrition_rate_prior_12m",
]


CATEGORICAL_FEATURES = [
    "department_id",
    "role_id",
    "work_location",
    "work_mode",
    "employment_type",
    "highest_education",
]



# LOAD THE RAW DATA

def load_data(path=DATA_FILE):
    """Open the Excel file and read the attrition_dataset sheet."""
    df = pd.read_excel(path, sheet_name="attrition_dataset")
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from: {path}")
    return df


# clean data
def clean_data(df):
    """Apply the fixes we found during the Task 1 audit."""

    df = df.copy() 

    for col in ["work_location", "work_mode", "employment_type",
                "department_id", "role_id", "highest_education"]:
        if col in df.columns and hasattr(df[col], "str"):
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    if "work_location" in df.columns:
        city_mapping = {
            "bangalore": "Bengaluru",
            "bengaluru": "Bengaluru",
            "chennai": "Chennai",
            "madras": "Chennai",
            "delhi ncr": "Delhi NCR",
            "delhi-ncr": "Delhi NCR",
        }
        df["work_location"] = (
            df["work_location"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: city_mapping.get(x, x.title() if x != "nan" else np.nan))
        )

    if "tenure_months" in df.columns:
        df["tenure_months"] = df["tenure_months"].clip(lower=0)

    if "attendance_rate_3m" in df.columns:
        df["attendance_rate_3m"] = df["attendance_rate_3m"].clip(upper=1.0)

    for col in LEAKAGE_COLUMNS + ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows. Rows left: {len(df)}")

    return df


# new features

class SalaryVsDeptMedian(BaseEstimator, TransformerMixin):
    """
    A simple custom machine that builds our Task 2 engineered features:
      1. salary_vs_dept_median = salary / department median salary
      2. tenure_years = tenure_months / 12
      3. raise_overdue = 1 if months_since_last_raise > 12 else 0
      4. unpaid_per_month = unpaid_leave_days_6m / 6
      5. attendance_gap = 1 - attendance_rate_3m
      6. overtime_load = overtime_hours / (hours_worked * 21)

    We only fit() on the train data -> no leakage. :-)
    """

    def fit(self, X, y=None):
        if "department_id" in X.columns and "annual_fixed_pay" in X.columns:
            self.dept_medians_ = (
                X.groupby("department_id")["annual_fixed_pay"].median().to_dict()
            )
            self.overall_median_ = X["annual_fixed_pay"].median()
        else:
            self.dept_medians_ = {}
            self.overall_median_ = 1.0
        return self

    def transform(self, X):
        X = X.copy()
        if "department_id" in X.columns and "annual_fixed_pay" in X.columns:
            medians = X["department_id"].map(self.dept_medians_).fillna(self.overall_median_)
            X["salary_vs_dept_median"] = X["annual_fixed_pay"] / medians.replace(0, 1.0)
        else:
            X["salary_vs_dept_median"] = 1.0

        if "tenure_months" in X.columns:
            X["tenure_years"] = X["tenure_months"] / 12.0
        else:
            X["tenure_years"] = 0.0


        if "months_since_last_raise" in X.columns:
            X["raise_overdue"] = (X["months_since_last_raise"] > 12).astype(float)
        else:
            X["raise_overdue"] = 0.0


        if "unpaid_leave_days_6m" in X.columns:
            X["unpaid_per_month"] = X["unpaid_leave_days_6m"] / 6.0
        else:
            X["unpaid_per_month"] = 0.0

        if "attendance_rate_3m" in X.columns:
            X["attendance_gap"] = 1.0 - X["attendance_rate_3m"]
        else:
            X["attendance_gap"] = 0.0

        if "avg_overtime_hours_per_month_3m" in X.columns and "avg_hours_worked_3m" in X.columns:
            standard_monthly = X["avg_hours_worked_3m"] * 21.0
            X["overtime_load"] = X["avg_overtime_hours_per_month_3m"] / standard_monthly.replace(0, 1.0)
        else:
            X["overtime_load"] = 0.0

        return X


# Pipeline
def build_pipeline():
    """
    The pipeline works in this order:

      1. add the salary_vs_dept_median column   (our custom step)
      2. number columns : fill missing -> put on the same scale
      3. text columns   : fill missing -> turn text into 0/1 columns
    """

    numeric_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median")), 
        ("scale", StandardScaler()),                         
    ])


    categorical_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    both_parts = ColumnTransformer(
        transformers=[
            ("numbers", numeric_steps, NUMERIC_FEATURES),
            ("texts", categorical_steps, CATEGORICAL_FEATURES),
        ],
        remainder="drop", 
    )

    full_pipeline = Pipeline(steps=[
        ("add_salary_ratio", SalaryVsDeptMedian()),
        ("clean_and_scale", both_parts),
    ])
    return full_pipeline


def get_column_names(pipe):
    """The list of columns the pipeline produces at the end."""
    return list(pipe.named_steps["clean_and_scale"].get_feature_names_out())

# save & load pipeline
def save_pipeline(pipe, X_train, folder=MODEL_DIR):
    """Save the fitted pipeline + its info (metadata) into ONE file."""

    info = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_rows": len(X_train),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "dropped_columns": LEAKAGE_COLUMNS + ID_COLUMNS,
        "note": "Fitted on TRAIN data only. Never on the full dataset.",
    }

    Path(folder).mkdir(parents=True, exist_ok=True)   
    file = Path(folder) / f"feature_pipeline_{PIPELINE_VERSION}.joblib"
    joblib.dump({"pipeline": pipe, "info": info}, file) 
    
    meta_path = Path(folder) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        
   
    store_dir = Path("models_store/features/1.0.0")
    store_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, store_dir / "model.joblib")
    with open(store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"Pipeline saved to: {file}")
    return file


def load_pipeline(file):
    """Load a saved pipeline. Gives back (pipeline, info)."""
    saved = joblib.load(file)
    return saved["pipeline"], saved["info"]


# run everything
def run_all():
   
    raw = load_data()

  
    df = clean_data(raw)

   
    X = df.drop(columns=[TARGET])   
    y = df[TARGET]                 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,    
        stratify=y,       
        random_state=42,   
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)               
    X_train_ready = pipe.transform(X_train) 
    X_test_ready = pipe.transform(X_test)   

    columns = get_column_names(pipe)
    print(f"Features created: {len(columns)}")

    missing = int(np.isnan(X_train_ready).sum() + np.isnan(X_test_ready).sum())
    print(f"Missing values after transform: {missing} (must be 0)")

    save_pipeline(pipe, X_train)


    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_ready, columns=columns).to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    pd.DataFrame(X_test_ready, columns=columns).to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_frame().to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_frame().to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)
    print(f"Ready data saved in: {OUTPUT_DIR}/")

    print("\nDONE! Raw data -> model-ready features.")

if __name__ == "__main__":
    run_all()