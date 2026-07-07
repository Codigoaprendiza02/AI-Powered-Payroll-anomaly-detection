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
from pipelines.p4_forecasting import run_forecasting_pipeline
from pipelines.p5_risk import run_risk_pipeline

@pytest.fixture(autouse=True)
def cleanup_registry_and_stores():
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

def test_qa_3_1_forecast_produced():
    # Run the full pipeline to generate feature store data and retrain models
    run_id = run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    
    forecast_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.json"
    assert forecast_path.exists()
    
    with open(forecast_path, "r") as f:
        forecast = json.load(f)
        
    assert "total_payroll" in forecast
    assert forecast["total_payroll"] > 0
    assert "per_department" in forecast
    assert isinstance(forecast["per_department"], dict)
    assert len(forecast["per_department"]) > 0
    assert "overtime_hours" in forecast
    assert forecast["overtime_hours"] >= 0
    assert "drift_adjusted" in forecast
    assert forecast["drift_adjusted"] is False  # No drift on first run

def test_qa_3_2_risk_register_persists():
    # Run twice with different files
    run_a = run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    run_b = run_full_pipeline("tests/qa_data/canonical/file2_bias_logic_failure.csv")
    
    register_path = Path(RISK_REGISTER_DIR) / "register.json"
    assert register_path.exists()
    
    with open(register_path, "r") as f:
        register = json.load(f)
        
    # Check E010 exists and has history entries from both runs
    e010 = next((e for e in register if e["employee_id"] == "EMP0010"), None)
    if not e010:
        # Fallback to E010 if employee_id synonyms map differently
        e010 = next((e for e in register if e["employee_id"] in ["E010", "EMP0010"]), None)
        
    assert e010 is not None
    assert e010["last_run_id"] == run_b
    assert len(e010["history"]) >= 2
    
    # Assert run IDs match in history
    history_run_ids = [h["run_id"] for h in e010["history"]]
    assert run_a in history_run_ids
    assert run_b in history_run_ids

def test_qa_3_3_risk_escalation_alert():
    # Create the three escalation CSV files
    p3_test_dir = Path("tests/qa_data/phase3")
    p3_test_dir.mkdir(parents=True, exist_ok=True)
    
    csv_1 = p3_test_dir / "run_esc_1.csv"
    csv_2 = p3_test_dir / "run_esc_2.csv"
    csv_3 = p3_test_dir / "run_esc_3.csv"
    
    content_1 = """employee_id,department,designation,base_salary,present_days,total_working_days,overtime_hours,overtime_pay_per_hour,net_salary,lop_days
E001,Engineering,Senior Engineer,85000,22,23,10,500,87500,1
E002,HR,HR Manager,60000,23,23,0,0,60000,0
E003,Finance,Analyst,45000,20,23,5,300,46500,3
E004,Engineering,Junior Engineer,35000,14,23,0,0,24130,9
E005,Sales,Sales Lead,55000,23,23,15,400,91000,0
E006,Engineering,Senior Engineer,85000,22,23,0,0,85000,1
E007,Finance,Analyst,45000,23,23,0,0,45000,0
E008,HR,HR Manager,60000,10,23,0,0,26087,13
E009,Sales,Sales Lead,55000,23,23,20,400,63000,0
E010,Engineering,Junior Engineer,35000,23,23,0,0,180000,0
E_WATCH,Engineering,Analyst,50000,22,23,5,300,51500,1
"""

    content_2 = """employee_id,department,designation,base_salary,present_days,total_working_days,overtime_hours,overtime_pay_per_hour,net_salary,lop_days
E001,Engineering,Senior Engineer,85000,22,23,10,500,87500,1
E002,HR,HR Manager,60000,23,23,0,0,60000,0
E003,Finance,Analyst,45000,20,23,5,300,46500,3
E004,Engineering,Junior Engineer,35000,14,23,0,0,24130,9
E005,Sales,Sales Lead,55000,23,23,15,400,91000,0
E006,Engineering,Senior Engineer,85000,22,23,0,0,85000,1
E007,Finance,Analyst,45000,23,23,0,0,45000,0
E008,HR,HR Manager,60000,10,23,0,0,26087,13
E009,Sales,Sales Lead,55000,23,23,20,400,63000,0
E010,Engineering,Junior Engineer,35000,23,23,0,0,180000,0
E_WATCH,Engineering,Analyst,50000,22,23,5,300,80000,1
"""

    content_3 = """employee_id,department,designation,base_salary,present_days,total_working_days,overtime_hours,overtime_pay_per_hour,net_salary,lop_days
E001,Engineering,Senior Engineer,85000,22,23,10,500,87500,1
E002,HR,HR Manager,60000,23,23,0,0,60000,0
E003,Finance,Analyst,45000,20,23,5,300,46500,3
E004,Engineering,Junior Engineer,35000,14,23,0,0,24130,9
E005,Sales,Sales Lead,55000,23,23,15,400,91000,0
E006,Engineering,Senior Engineer,85000,22,23,0,0,85000,1
E007,Finance,Analyst,45000,23,23,0,0,45000,0
E008,HR,HR Manager,60000,10,23,0,0,26087,13
E009,Sales,Sales Lead,55000,23,23,20,400,63000,0
E010,Engineering,Junior Engineer,35000,23,23,0,0,180000,0
E_WATCH,Engineering,Analyst,50000,22,23,5,300,150000,1
"""
    
    csv_1.write_text(content_1)
    csv_2.write_text(content_2)
    csv_3.write_text(content_3)
    
    # Run the three files in sequence
    run1 = run_full_pipeline(str(csv_1))
    run2 = run_full_pipeline(str(csv_2))
    run3 = run_full_pipeline(str(csv_3))
    
    # Check alerts in run 3 for RISK_ESCALATION_ALERT
    run3_alerts_path = Path(AUDIT_LOG_DIR) / f"{run3}_alerts.json"
    assert run3_alerts_path.exists()
    
    with open(run3_alerts_path, "r") as f:
        run3_alerts = json.load(f)
        
    esc_alerts = [a for a in run3_alerts if a["alert_type"] == "RISK_ESCALATION_ALERT" and a["affected_entity"] == "E_WATCH"]
    assert len(esc_alerts) == 1
    assert esc_alerts[0]["severity"] == "HIGH"
    
    # Check first two runs have no RISK_ESCALATION_ALERT for E_WATCH
    for r in [run1, run2]:
        alerts_path = Path(AUDIT_LOG_DIR) / f"{r}_alerts.json"
        if alerts_path.exists():
            with open(alerts_path, "r") as f:
                alerts = json.load(f)
            esc_alerts_prior = [a for a in alerts if a["alert_type"] == "RISK_ESCALATION_ALERT" and a["affected_entity"] == "E_WATCH"]
            assert len(esc_alerts_prior) == 0

def test_qa_3_4_forecast_spike_alert():
    # Run baseline healthy data first
    run_full_pipeline("tests/qa_data/canonical/file1_baseline_healthy.csv")
    
    # Run bias/logic failure which has multiple inflated/manipulated salaries pushing forecast up
    spike_run = run_full_pipeline("tests/qa_data/canonical/file2_bias_logic_failure.csv")
    
    alerts_path = Path(AUDIT_LOG_DIR) / f"{spike_run}_alerts.json"
    assert alerts_path.exists()
    
    with open(alerts_path, "r") as f:
        alerts = json.load(f)
        
    spike_alerts = [a for a in alerts if a["alert_type"] == "FORECAST_PAYROLL_SPIKE_ALERT"]
    assert len(spike_alerts) > 0
    assert spike_alerts[0]["severity"] == "HIGH"
    assert spike_alerts[0]["affected_entity"] == "company_payroll"
    assert spike_alerts[0]["trigger_value"] > spike_alerts[0]["threshold_value"]
