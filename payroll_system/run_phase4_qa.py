import os
import json
import shutil
import glob
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    AUDIT_LOG_DIR, FEATURES_STORE_DIR, MODELS_REGISTRY_DIR,
    DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR
)
from pipelines.pipeline_runner import run_full_pipeline
from models.registry import register_model, list_model_versions
from models.registry_manager import rollback_model

def clean_all_stores():
    for d in [MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR, DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        os.makedirs(d, exist_ok=True)

def generate_qa_files():
    p4_dir = Path("tests/qa_data/phase4")
    p4_dir.mkdir(parents=True, exist_ok=True)
    
    similar_path = p4_dir / "test_similar.csv"
    if not similar_path.exists():
        df = pd.read_csv("tests/qa_data/canonical/file1_baseline_healthy.csv")
        rng = np.random.default_rng(42)
        factors = rng.uniform(0.95, 1.05, size=len(df))
        df["base_salary"] = (df["base_salary"] * factors).round().astype(int)
        df["net_salary"]  = (df["net_salary"]  * factors).round().astype(int)
        df.to_csv(similar_path, index=False)
        
    mod_path = p4_dir / "test_moderate_drift.csv"
    # For manual QA tests we can generate moderate drift with 1.11 to make overall_severity Moderate Drift
    df = pd.read_csv("tests/qa_data/canonical/file1_baseline_healthy.csv")
    df["base_salary"] = (df["base_salary"] * 1.11).round().astype(int)
    df["net_salary"]  = (df["net_salary"]  * 1.11).round().astype(int)
    df.to_csv(mod_path, index=False)

def run_qa():
    print("# Phase 4 Human QA Testing Results\n")
    generate_qa_files()
    
    # ----------------------------------------------------
    # QA-4.1: Cold Start
    # ----------------------------------------------------
    print("## QA-4.1 — Cold Start: Drift Monitoring Skipped on First Run")
    clean_all_stores()
    try:
        run_id = run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
        alerts = json.load(open(f"store/audit_log/{run_id}_alerts.json"))
        cold_alerts = [a for a in alerts if a["alert_type"] == "COLD_START_NOTICE"]
        drift_alerts = [a for a in alerts if "DRIFT" in a["alert_type"]]
        latest_ref_exists = Path("store/drift/latest_reference.json").exists()
        
        status = "PASS" if (len(cold_alerts) == 1 and len(drift_alerts) == 0 and latest_ref_exists) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **COLD_START_NOTICE alerts**: {len(cold_alerts)} (Expected: 1)")
        print(f"- **Drift alerts**: {len(drift_alerts)} (Expected: 0)")
        print(f"- **Reference written (latest_reference.json)**: {latest_ref_exists} (Expected: True)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-4.2: Similar Data
    # ----------------------------------------------------
    print("## QA-4.2 — No Drift Detected on Similar Data")
    clean_all_stores()
    try:
        run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
        sim_run = run_full_pipeline("tests/qa_data/phase4/test_similar.csv")
        
        d = json.load(open(f"store/audit_log/{sim_run}_drift_report.json"))
        severity = d.get("overall_severity")
        max_psi = d.get("max_psi")
        
        alerts = json.load(open(f"store/audit_log/{sim_run}_alerts.json"))
        drift_alerts = [a for a in alerts if "DRIFT" in a["alert_type"]]
        
        status = "PASS" if (severity == "No Drift" and max_psi < 0.1 and len(drift_alerts) == 0) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Overall severity**: {severity} (Expected: No Drift)")
        print(f"- **Max PSI**: {max_psi:.4f} (Expected: < 0.1)")
        print(f"- **Drift alerts**: {len(drift_alerts)} (Expected: 0)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-4.3: Significant Drift
    # ----------------------------------------------------
    print("## QA-4.3 — Significant Drift Detected and Retrain Triggered")
    clean_all_stores()
    try:
        run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
        drift_run = run_full_pipeline("tests/qa_data/canonical/file3_drift.csv")
        
        d = json.load(open(f"store/audit_log/{drift_run}_drift_report.json"))
        severity = d.get("overall_severity")
        max_psi = d.get("max_psi")
        
        h = json.load(open(f"store/audit_log/{drift_run}_drift_handling_log.json"))
        strategy = h.get("strategy_applied")
        retrain = h.get("retrain_triggered")
        
        status = "PASS" if (severity == "Significant Drift" and max_psi > 0.25 and strategy == "full_retrain" and retrain is True) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Overall severity**: {severity} (Expected: Significant Drift)")
        print(f"- **Max PSI**: {max_psi:.4f} (Expected: > 0.25)")
        print(f"- **strategy_applied**: {strategy} (Expected: full_retrain)")
        print(f"- **retrain_triggered**: {retrain} (Expected: True)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-4.4: Moderate Drift
    # ----------------------------------------------------
    print("## QA-4.4 — Moderate Drift: Threshold Recalibration Only, No Retrain")
    clean_all_stores()
    try:
        run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
        mod_run = run_full_pipeline("tests/qa_data/phase4/test_moderate_drift.csv")
        
        d = json.load(open(f"store/audit_log/{mod_run}_drift_report.json"))
        severity = d.get("overall_severity")
        
        h = json.load(open(f"store/audit_log/{mod_run}_drift_handling_log.json"))
        strategy = h.get("strategy_applied")
        retrain = h.get("retrain_triggered")
        
        thresholds_updated = Path("config/thresholds_current.json").exists()
        
        status = "PASS" if (severity == "Moderate Drift" and strategy == "threshold_recalibration" and retrain is False and thresholds_updated) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Overall severity**: {severity} (Expected: Moderate Drift)")
        print(f"- **strategy_applied**: {strategy} (Expected: threshold_recalibration)")
        print(f"- **retrain_triggered**: {retrain} (Expected: False)")
        print(f"- **thresholds_current.json created**: {thresholds_updated} (Expected: True)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-4.5: Model Rollback
    # ----------------------------------------------------
    print("## QA-4.5 — Model Rollback Works")
    clean_all_stores()
    try:
        # Register two mock models manually
        from sklearn.ensemble import IsolationForest
        model_a = IsolationForest()
        model_a.fit([[1, 2], [3, 4]])
        register_model("anomaly_detector", "version_a", model_a, {"run_id": "run_a", "metrics": {"f1": 0.5}}, ["f1", "f2"], mark_production=True)
        
        model_b = IsolationForest()
        model_b.fit([[1, 2], [3, 4]])
        register_model("anomaly_detector", "version_b", model_b, {"run_id": "run_b", "metrics": {"f1": 0.6}}, ["f1", "f2"], mark_production=True)
        
        # Verify B is production
        versions = list_model_versions("anomaly_detector")
        v_b_prod = (versions[0]["version_id"] == "version_b" and versions[0]["metadata"].get("is_production") is True)
        
        # Rollback
        res = rollback_model("anomaly_detector")
        restored_ok = res["model_version"] == "version_a"
        
        # Check active on disk
        versions_after = list_model_versions("anomaly_detector")
        versions_on_disk = {v["version_id"]: v["metadata"].get("is_production") for v in versions_after}
        disk_ok = (versions_on_disk["version_a"] is True and versions_on_disk["version_b"] is False)
        
        status = "PASS" if (v_b_prod and restored_ok and disk_ok) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Version B initial production**: {v_b_prod} (Expected: True)")
        print(f"- **Rollback restored Version A**: {restored_ok} (Expected: True)")
        print(f"- **On-disk metadata state verified**: {disk_ok} (Expected: True)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")

if __name__ == "__main__":
    run_qa()
