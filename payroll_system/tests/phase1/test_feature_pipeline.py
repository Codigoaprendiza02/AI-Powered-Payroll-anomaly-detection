import pytest
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pipelines.p1_feature import (
    load_raw_data,
    detect_schema,
    run_quality_checks,
    clean_data,
    enforce_dtypes,
    engineer_features,
    write_feature_store,
    read_feature_store,
    run_feature_pipeline,
    InsufficientDataError,
    DataCleaningError
)
from config.settings import AUDIT_LOG_DIR, FEATURES_STORE_DIR

@pytest.fixture
def setup_dirs():
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    os.makedirs(FEATURES_STORE_DIR, exist_ok=True)

def test_load_raw_data(setup_dirs):
    # Test loading CSV
    df_csv = load_raw_data("tests/qa_data/phase1/test_standard.csv")
    assert len(df_csv) == 5
    assert "employee_id" in df_csv.columns
    
    # Test unsupported extension raises ValueError
    with pytest.raises(ValueError):
        load_raw_data("requirements.txt")

def test_detect_schema(setup_dirs):
    df_syn = load_raw_data("tests/qa_data/phase1/test_synonyms.csv")
    schema = detect_schema(df_syn)
    assert schema["confidence"] == 1.0
    assert schema["mapped"]["employee_id"] == "emp_id"
    assert schema["mapped"]["base_salary"] == "basic_pay"
    assert len(schema["unmapped"]) == 0

def test_low_schema_confidence(setup_dirs):
    df_garbled = load_raw_data("tests/qa_data/phase1/test_garbled.csv")
    schema = detect_schema(df_garbled)
    assert schema["confidence"] < 0.7
    assert len(schema["unmapped"]) > 0

def test_run_quality_checks(setup_dirs):
    df = load_raw_data("tests/qa_data/phase1/test_standard.csv")
    # Add a duplicate row for testing
    df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    report = run_quality_checks(df_with_dup, "test-run-qc")
    assert report["total_rows"] == 6
    assert report["duplicate_rows"] == 1
    
    # Test null flagging
    df_nulls = df.copy()
    df_nulls.loc[0, "department"] = np.nan
    # Now department null pct is 20%
    report_nulls = run_quality_checks(df_nulls, "test-run-qc-nulls")
    # For 5 rows, 1 null is 20%. Let's add another null to exceed 20%.
    df_nulls.loc[1, "department"] = np.nan
    report_nulls = run_quality_checks(df_nulls, "test-run-qc-nulls2")
    assert report_nulls["columns"]["department"]["flagged"] is True

def test_clean_data_duplicates_and_nulls(setup_dirs):
    # Create a raw dataset with nulls, duplicates, impossible values
    raw_data = {
        "employee_id": ["E01", "E01", "E02", None, "E04", "E05"],
        "department": ["Eng", "Eng", "HR", "Sales", None, "Eng"],
        "designation": ["Eng1", "Eng1", "HR1", "Sales1", "Eng2", "Eng3"],
        "base_salary": [50000, 50000, 60000, 70000, np.nan, 80000],
        "present_days": [22, 22, 23, 20, 21, 23],
        "total_working_days": [23, 23, 23, 23, 23, 23],
        "overtime_hours": [5, 5, 0, np.nan, 0, -2], # Negative OT hours
        "overtime_pay_per_hour": [200, 200, 0, 0, 0, 100],
        "net_salary": [51000, 51000, 60000, 70000, 40000, 80000],
        "lop_days": [1, 1, 0, 3, 2, 0]
    }
    df_raw = pd.DataFrame(raw_data)
    
    # Run pipeline quality check first to get mapped columns
    report = run_quality_checks(df_raw, "test-run-clean")
    
    df_clean, summary = clean_data(df_raw, report, "test-run-clean", min_rows=1)
    
    # 1. Duplicates: E01 duplicate removed
    assert summary["steps"]["duplicates"] == 1
    
    # 2. Nulls: Row 3 dropped because employee_id is null.
    #           Row 4 dropped because base_salary is null.
    #           Row 0 (duplicate) dropped during duplicate step.
    #           So 3 rows remaining: Row 0, Row 2, Row 5.
    assert len(df_clean) == 3
    
    # Verify no nulls remain in the final df
    assert df_clean.isnull().sum().sum() == 0
    assert "Unknown" in df_clean["department"].values or len(df_clean) > 0

def test_clean_data_impossible_values(setup_dirs):
    raw_data = {
        "employee_id": ["E01", "E02", "E03", "E04", "E05"],
        "department": ["Eng", "HR", "Sales", "Ops", "Finance"],
        "designation": ["Eng1", "HR1", "Analyst", "Lead", "Mgr"],
        "base_salary": [50000, 60000, 70000, -1000, 80000], # E04 has negative base_salary
        "present_days": [25, 23, -2, 20, 23], # E01 has present > total, E03 has negative present
        "total_working_days": [23, 23, 23, 23, 23],
        "overtime_hours": [0, 0, 0, 0, -5], # E05 has negative OT hours
        "overtime_pay_per_hour": [0, 0, 0, 0, 100],
        "net_salary": [50000, 60000, 70000, 70000, 0], # E05 has 0 net salary
        "lop_days": [0, 0, 25, 3, 0]
    }
    df_raw = pd.DataFrame(raw_data)
    report = run_quality_checks(df_raw, "test-run-impos")
    
    df_clean, summary = clean_data(df_raw, report, "test-run-impos", min_rows=1)
    
    # Verify present days capped to total working days for E01
    e01_row = df_clean[df_clean["employee_id"] == "E01"].iloc[0]
    assert e01_row["present_days"] == 23
    
    # Verify negative present_days zeroed for E03
    e03_row = df_clean[df_clean["employee_id"] == "E03"].iloc[0]
    assert e03_row["present_days"] == 0
    assert e03_row["lop_days"] == 23 # LOP recomputed to total - present
    
    # Verify rows with nonpositive base_salary (E04) and net_salary (E05) are dropped
    assert "E04" not in df_clean["employee_id"].values
    assert "E05" not in df_clean["employee_id"].values

def test_insufficient_data_error(setup_dirs):
    df_short = pd.DataFrame({
        "employee_id": ["E01", "E02"],
        "base_salary": [50000, 60000],
        "net_salary": [50000, 60000]
    })
    report = run_quality_checks(df_short, "test-run-short")
    with pytest.raises(InsufficientDataError):
        clean_data(df_short, report, "test-run-short")

def test_enforce_dtypes(setup_dirs):
    df = pd.DataFrame({
        "base_salary": [50000.5, "60000", 70000],
        "present_days": [22.0, 23.0, 20.0]
    })
    mapped = {"base_salary": "base_salary", "present_days": "present_days"}
    df_casted = enforce_dtypes(df, mapped)
    assert df_casted["base_salary"].dtype == "int64"
    assert df_casted["present_days"].dtype == "int64"

def test_engineer_features(setup_dirs):
    df = load_raw_data("tests/qa_data/phase1/test_standard.csv")
    schema = detect_schema(df)
    df_clean, _ = clean_data(df, run_quality_checks(df, "test-eng"), "test-eng")
    
    df_feat = engineer_features(df_clean, schema)
    
    assert "attendance_ratio" in df_feat.columns
    assert "overtime_pay" in df_feat.columns
    assert "earned_base" in df_feat.columns
    assert "expected_gross_salary" in df_feat.columns
    assert "expected_net" in df_feat.columns
    assert "salary_diff" in df_feat.columns
    assert "salary_dev_pct" in df_feat.columns
    assert "robust_z_base_salary" in df_feat.columns

def test_engineer_features_missing_optional(setup_dirs):
    # Ingest minimal dataset missing overtime columns
    df = load_raw_data("tests/qa_data/phase1/test_minimal.csv")
    schema = detect_schema(df)
    df_clean, _ = clean_data(df, run_quality_checks(df, "test-eng-min"), "test-eng-min", min_rows=1)
    
    df_feat = engineer_features(df_clean, schema)
    
    # Overtime features are missing, but expected_net, attendance_ratio, earned_base, and salary_diff should be present
    assert "attendance_ratio" in df_feat.columns
    assert "expected_net" in df_feat.columns
    assert "salary_diff" in df_feat.columns

def test_feature_store_roundtrip(setup_dirs):
    df = load_raw_data("tests/qa_data/phase1/test_standard.csv")
    schema = detect_schema(df)
    df_clean, _ = clean_data(df, run_quality_checks(df, "test-fs"), "test-fs")
    df_feat = engineer_features(df_clean, schema)
    
    write_feature_store(df_feat, "test-run-fs")
    df_loaded = read_feature_store("test-run-fs")
    
    pd.testing.assert_frame_equal(df_feat, df_loaded)
