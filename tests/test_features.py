"""
Tests for our feature pipeline.

IMPORTANT: the data in these tests is FAKE (made-up numbers).
It only checks that our CODE works. It says NOTHING about the company.
(This follows our project rule about synthetic test data.)

Run from the project root:   pytest
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.features.build_features import (
    clean_data, build_pipeline, get_column_names,
    save_pipeline, load_pipeline,
    TARGET, LEAKAGE_COLUMNS, ID_COLUMNS, PIPELINE_VERSION,
)


# --------------------------------------------------------
# Make FAKE data with the same column names as the real file
# --------------------------------------------------------
def make_fake_data(rows=100):
    rng = np.random.default_rng(42)   # fixed seed = same fake data every time
    return pd.DataFrame({
        # id + leakage columns (should get REMOVED by clean_data):
        "employee_id": [f"EMP-{i}" for i in range(rows)],
        "exit_date": np.nan,
        # text columns:
        "department_id": rng.choice(["D01", "D02"], rows),
        "role_id": rng.choice(["R001", "R002"], rows),
        "work_location": rng.choice(["Pune", "delhi ncr"], rows),
        "work_mode": rng.choice(["Remote", "Hybrid"], rows),
        "employment_type": rng.choice(["Permanent", "Contract"], rows),
        "highest_education": rng.choice(["B.E.", "MCA", np.nan], rows),  # some missing
        # number columns:
        "job_level": rng.integers(1, 6, rows),
        "tenure_months": rng.integers(1, 200, rows),
        "total_experience_years": rng.uniform(0, 30, rows),
        "previous_employers_count": rng.integers(0, 6, rows),
        "commute_distance_km": rng.uniform(1, 60, rows),
        "annual_fixed_pay": rng.integers(300000, 2500000, rows),
        "comp_ratio": rng.uniform(0.7, 1.3, rows),
        "raises_received": rng.integers(0, 7, rows),
        "last_raise_pct": rng.choice([5.0, 8.0, np.nan], rows),
        "months_since_last_raise": rng.integers(0, 48, rows),
        "promotions_received": rng.integers(0, 3, rows),
        "internal_transfers": rng.integers(0, 3, rows),
        "last_performance_rating": rng.choice([3.0, 4.0, np.nan], rows),
        "last_engagement_score": rng.choice([2.5, 4.0, np.nan], rows),
        "manager_changes_12m": rng.integers(0, 4, rows),
        "training_hours_12m": rng.uniform(0, 60, rows),
        "attendance_rate_3m": rng.uniform(0.7, 1.0, rows),
        "late_arrivals_3m": rng.integers(0, 20, rows),
        "avg_overtime_hours_per_month_3m": rng.uniform(0, 25, rows),
        "avg_hours_worked_3m": rng.uniform(6, 10, rows),
        "wfh_share_3m": rng.uniform(0, 1, rows),
        "unpaid_leave_days_6m": rng.integers(0, 10, rows),
        "paid_leave_days_6m": rng.integers(0, 15, rows),
        "dept_attrition_rate_prior_12m": rng.uniform(0, 0.4, rows),
        # the answer:
        "left_within_12m": rng.choice([0, 1], rows, p=[0.85, 0.15]),
    })


# --------------------------------------------------------
# Helper: fit the pipeline on fake TRAIN data (like real life)
# --------------------------------------------------------
def fit_on_fake_data():
    clean = clean_data(make_fake_data())
    X = clean.drop(columns=[TARGET])
    y = clean[TARGET]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)   # train only, like real life
    return pipe, X_test


# --------------------------------------------------------
# The tests
# --------------------------------------------------------
def test_1_bad_columns_are_removed():
    """Leakage and id columns must be gone after cleaning."""
    clean = clean_data(make_fake_data())
    for col in LEAKAGE_COLUMNS + ID_COLUMNS:
        assert col not in clean.columns, f"{col} was not removed!"


def test_2_output_shape_is_correct():
    """Same rows in = same rows out, and columns = our final column list."""
    pipe, X_test = fit_on_fake_data()
    out = pipe.transform(X_test)
    assert out.shape[0] == len(X_test)
    assert out.shape[1] == len(get_column_names(pipe))


def test_3_no_missing_values_after_transform():
    """The pipeline must fill every gap."""
    pipe, X_test = fit_on_fake_data()
    out = pipe.transform(X_test)
    assert int(np.isnan(out).sum()) == 0, "Found missing values after transform!"


def test_4_same_input_gives_same_output():
    """Running it twice must give identical numbers."""
    pipe, X_test = fit_on_fake_data()
    out_1 = pipe.transform(X_test.head(10))
    out_2 = pipe.transform(X_test.head(10))
    assert np.array_equal(out_1, out_2)


def test_5_saved_pipeline_works_the_same(tmp_path):
    """Load the saved file -> it must behave exactly like the original."""
    pipe, X_test = fit_on_fake_data()
    file = save_pipeline(pipe, X_test.head(75), folder=tmp_path)

    loaded_pipe, info = load_pipeline(file)

    assert info["pipeline_version"] == PIPELINE_VERSION      # version saved
    assert np.array_equal(pipe.transform(X_test),
                          loaded_pipe.transform(X_test))     # same output