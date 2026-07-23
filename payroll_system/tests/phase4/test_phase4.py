import pytest
import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    AUDIT_LOG_DIR, FEATURES_STORE_DIR, MODELS_REGISTRY_DIR,
    DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR
)
from pipelines.pipeline_runner import run_full_pipeline, AlertManagerStub
from pipelines.p6_drift_monitor import compute_psi, run_drift_monitoring
from pipelines.p7_drift_handle import run_drift_handling
from models.registry_manager import rollback_model
from models.registry import list_model_versions

@pytest.fixture(autouse=True)
def cleanup_and_setup_dirs():
    # Setup clean directories before each test
    for d in [MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR, DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        os.makedirs(d, exist_ok=True)
    yield
    # Cleanup after test runs
    for d in [MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR, DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        os.makedirs(d, exist_ok=True)

@pytest.fixture(scope="session", autouse=True)
def generate_qa_data_files():
    # Ensure phase4 testing folder exists
    p4_dir = Path("tests/qa_data/phase4")
    p4_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test_similar.csv if not exists
    similar_path = p4_dir / "test_similar.csv"
    if not similar_path.exists():
        df = pd.read_csv("tests/qa_data/canonical/file1_baseline_healthy.csv")
        rng = np.random.default_rng(42)
        factors = rng.uniform(0.95, 1.05, size=len(df))
        df["base_salary"] = (df["base_salary"] * factors).round().astype(int)
        df["net_salary"]  = (df["net_salary"]  * factors).round().astype(int)
        df.to_csv(similar_path, index=False)
        
    # Create test_moderate_drift.csv
    mod_path = p4_dir / "test_moderate_drift.csv"
    df = pd.read_csv("tests/qa_data/canonical/file1_baseline_healthy.csv")
    df["base_salary"] = (df["base_salary"] * 1.11).round().astype(int)
    df["net_salary"]  = (df["net_salary"]  * 1.11).round().astype(int)
    df.to_csv(mod_path, index=False)

def test_psi_computation():
    # Identical distributions should return PSI ≈ 0
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(100, 15, 1000))
    curr_identical = pd.Series(rng.normal(100, 15, 1000))
    
    psi_identical = compute_psi(ref, curr_identical)
    assert psi_identical < 0.1
    
    # Extremely different distributions should return a value > 2
    curr_different = pd.Series(rng.normal(150, 25, 1000))
    psi_different = compute_psi(ref, curr_different)
    assert psi_different > 2.0

def test_cold_start():
    # Fresh run on baseline healthy data
    run_id = run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    
    # Verify alerts contains COLD_START_NOTICE
    alerts_path = Path(AUDIT_LOG_DIR) / f"{run_id}_alerts.json"
    assert alerts_path.exists()
    with open(alerts_path, "r") as f:
        alerts = json.load(f)
        
    cold_starts = [a for a in alerts if a["alert_type"] == "COLD_START_NOTICE"]
    drift_alerts = [a for a in alerts if "DRIFT" in a["alert_type"]]
    
    assert len(cold_starts) == 1
    assert len(drift_alerts) == 0
    assert cold_starts[0]["severity"] == "INFO"
    
    # Verify latest_reference.json was written
    latest_ref = Path(DRIFT_STORE_DIR) / "latest_reference.json"
    assert latest_ref.exists()
    with open(latest_ref, "r") as f:
        ref_data = json.load(f)
    assert ref_data["run_id"] == run_id

def test_no_drift_on_similar_data():
    # Run baseline then similar
    run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    sim_run = run_full_pipeline("tests/qa_data/phase4/test_similar.csv")
    
    # Load drift report
    report_path = Path(AUDIT_LOG_DIR) / f"{sim_run}_drift_report.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
        
    assert report["overall_severity"] == "No Drift"
    assert report["max_psi"] < 0.1
    assert report["drift_detected"] is False
    
    # Check drift handling log
    log_path = Path(AUDIT_LOG_DIR) / f"{sim_run}_drift_handling_log.json"
    assert log_path.exists()
    with open(log_path, "r") as f:
        log = json.load(f)
    assert log["strategy_applied"] == "skipped — no drift"
    assert log["retrain_triggered"] is False

def test_moderate_drift():
    # Run baseline then moderate drift
    run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    mod_run = run_full_pipeline("tests/qa_data/phase4/test_moderate_drift.csv")
    
    # Load drift report
    report_path = Path(AUDIT_LOG_DIR) / f"{mod_run}_drift_report.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
        
    assert report["overall_severity"] == "Moderate Drift"
    assert report["drift_detected"] is True
    
    # Check drift handling log shows recalibration only, no retrain
    log_path = Path(AUDIT_LOG_DIR) / f"{mod_run}_drift_handling_log.json"
    assert log_path.exists()
    with open(log_path, "r") as f:
        log = json.load(f)
    assert log["strategy_applied"] == "threshold_recalibration"
    assert log["retrain_triggered"] is False
    
    # Check thresholds_current.json was updated
    thresholds_path = Path("config/thresholds_current.json")
    assert thresholds_path.exists()
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
    assert "base_salary" in thresholds
    assert thresholds["base_salary"]["mean"] > 0

def test_significant_drift_and_retrain():
    # Run baseline then significant drift file
    run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    
    versions_before = list_model_versions("anomaly_detector")
    assert len(versions_before) == 1
    
    drift_run = run_full_pipeline("tests/qa_data/canonical/file3_drift.csv")
    
    # Check drift report
    report_path = Path(AUDIT_LOG_DIR) / f"{drift_run}_drift_report.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
        
    assert report["overall_severity"] == "Significant Drift"
    assert report["max_psi"] > 0.25
    assert report["drift_detected"] is True
    
    # Check drift handling log shows retrain was triggered
    log_path = Path(AUDIT_LOG_DIR) / f"{drift_run}_drift_handling_log.json"
    assert log_path.exists()
    with open(log_path, "r") as f:
        log = json.load(f)
    assert log["strategy_applied"] == "full_retrain"
    assert log["retrain_triggered"] is True
    
    # Check new model version is registered
    versions_after = list_model_versions("anomaly_detector")
    assert len(versions_after) > 1

def test_model_rollback():
    from models.registry import register_model
    from sklearn.ensemble import IsolationForest
    
    # 1. Register version A as production
    model_a = IsolationForest()
    model_a.fit([[1, 2], [3, 4]])
    register_model("anomaly_detector", "version_a", model_a, {"run_id": "run_a", "metrics": {"f1": 0.5}}, ["f1", "f2"], mark_production=True)
    
    # 2. Register version B as production (simulating a model swap)
    model_b = IsolationForest()
    model_b.fit([[1, 2], [3, 4]])
    register_model("anomaly_detector", "version_b", model_b, {"run_id": "run_b", "metrics": {"f1": 0.6}}, ["f1", "f2"], mark_production=True)
    
    # Verify version B is production
    versions = list_model_versions("anomaly_detector")
    assert len(versions) == 2
    assert versions[0]["version_id"] == "version_b"
    assert versions[0]["metadata"].get("is_production") is True
    assert versions[1]["version_id"] == "version_a"
    assert versions[1]["metadata"].get("is_production") is False
    
    # Perform rollback
    res = rollback_model("anomaly_detector")
    assert res["model_version"] == "version_a"
    
    # Verify metadata on disk is updated correctly
    versions_after = list_model_versions("anomaly_detector")
    versions_on_disk = {v["version_id"]: v["metadata"].get("is_production") for v in versions_after}
    assert versions_on_disk["version_a"] is True
    assert versions_on_disk["version_b"] is False

def test_psi_history_accumulation():
    # Run multiple times and verify psi_history accumulation
    run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    run_full_pipeline("tests/qa_data/phase4/test_similar.csv")
    run_full_pipeline("tests/qa_data/phase4/test_moderate_drift.csv")
    
    history_path = Path(DRIFT_STORE_DIR) / "psi_history.json"
    assert history_path.exists()
    with open(history_path, "r") as f:
        history = json.load(f)
        
    assert len(history) >= 2
    # Verify that first entry doesn't get overwritten
    run_ids = [h["run_id"] for h in history]
    assert len(set(run_ids)) == len(run_ids)
