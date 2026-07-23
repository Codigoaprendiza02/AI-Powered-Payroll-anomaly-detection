import os
import json
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from config.settings import (
    API_KEY, MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR, 
    DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR
)

client = TestClient(app)

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

# Helper to generate synthetic 50-employee CSV with synonyms
def generate_synthetic_csv(file_path: Path, salary_multiplier: float = 1.0):
    rng = np.random.default_rng(99)
    depts = ['Engineering', 'Finance', 'HR', 'Sales', 'Operations']
    desigs = ['Junior Engineer', 'Senior Engineer', 'Analyst', 'Manager', 'Lead']
    base_salaries = {'Junior Engineer': 35000, 'Senior Engineer': 85000,
                     'Analyst': 45000, 'Manager': 65000, 'Lead': 55000}
    
    rows = []
    for i in range(1, 51):
        desig = desigs[i % 5]
        dept = depts[i % 5]
        base = int(base_salaries[desig] * rng.uniform(0.95, 1.05) * salary_multiplier)
        twd = 23
        pd_ = int(rng.uniform(20, 24))  # 20 to 23 present days
        if pd_ > twd:
            pd_ = twd
        ot_h = int(rng.uniform(0, 10))
        ot_r = 400 if desig in ['Lead', 'Manager'] else 300
        net = int(base * pd_ / twd + ot_h * ot_r)
        lop = twd - pd_
        rows.append([f'E{i:03d}', dept, desig, base, pd_, twd, ot_h, ot_r, net, lop])
        
    df = pd.DataFrame(rows, columns=[
        'emp_id', 'dept', 'title', 'basic_pay', 
        'days_present', 'business_days', 'ot_hours', 
        'ot_rate', 'take_home', 'loss_of_pay'
    ])
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)

def test_integration_workflow():
    p5_test_dir = Path("tests/qa_data/phase5")
    p5_test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = p5_test_dir / "test_50_employees.csv"
    generate_synthetic_csv(csv_path, salary_multiplier=1.0)
    
    headers = {"X-API-Key": API_KEY}
    
    # 1. GET health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] in ["healthy", "degraded"]
    
    # 2. GET without API key -> 401
    res_unauth = client.post("/data/ingest", files={"file": ("test.csv", b"")})
    assert res_unauth.status_code == 401
    
    # 3. POST ingest data -> 200
    with open(csv_path, "rb") as f:
        res_ingest = client.post("/data/ingest", headers=headers, files={"file": ("test_50_employees.csv", f, "text/csv")})
        
    assert res_ingest.status_code == 200
    ingest_data = res_ingest.json()
    assert "run_id" in ingest_data
    assert ingest_data["status"] == "COMPLETE"
    run_id = ingest_data["run_id"]
    
    # 4. GET run status
    res_run = client.get(f"/runs/{run_id}", headers=headers)
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert run_data["status"] == "COMPLETE"
    assert len(run_data["pipeline_steps"]) == 8
    
    # 5. GET run alerts
    res_alerts = client.get(f"/runs/{run_id}/alerts", headers=headers)
    assert res_alerts.status_code == 200
    alerts_data = res_alerts.json()
    assert isinstance(alerts_data, list)
    
    # 6. GET run report
    res_report = client.get(f"/runs/{run_id}/report", headers=headers)
    assert res_report.status_code == 200
    report_data = res_report.json()
    assert report_data["run_id"] == run_id
    assert "inference" in report_data
    
    # 7. GET employees high-risk
    res_hr = client.get("/employees/high-risk", headers=headers)
    assert res_hr.status_code == 200
    hr_data = res_hr.json()
    assert isinstance(hr_data, list)
    
    # 8. GET forecasts latest
    res_forecast = client.get("/forecasts/latest", headers=headers)
    assert res_forecast.status_code == 200
    forecast_data = res_forecast.json()
    assert "total_payroll" in forecast_data
    assert forecast_data["total_payroll"] > 0
    
    # 9. GET monitoring drift (should be cold start on first run)
    res_drift = client.get("/monitoring/drift", headers=headers)
    assert res_drift.status_code == 200
    drift_data = res_drift.json()
    assert drift_data.get("cold_start") is True
    
    # 10. GET model health
    res_health_m = client.get("/monitoring/model-health", headers=headers)
    assert res_health_m.status_code == 200
    health_m_data = res_health_m.json()
    assert "model_version" in health_m_data
    assert health_m_data["run_id"] == run_id
    
    # 11. GET schema report
    res_schema = client.get("/schema/report", headers=headers)
    assert res_schema.status_code == 200
    schema_data = res_schema.json()
    assert schema_data["schema_confidence"] >= 0.7
    
    # 12. Idempotency test (POST identical file -> same run_id)
    with open(csv_path, "rb") as f:
        res_ingest_dup = client.post("/data/ingest", headers=headers, files={"file": ("test_50_employees.csv", f, "text/csv")})
    assert res_ingest_dup.status_code == 200
    assert res_ingest_dup.json()["run_id"] == run_id
    
    # 13. Test drift (POST file with shifted salaries)
    csv_drift_path = p5_test_dir / "test_50_employees_drift.csv"
    generate_synthetic_csv(csv_drift_path, salary_multiplier=2.1) # Double the salaries to guarantee drift
    
    with open(csv_drift_path, "rb") as f:
        res_ingest_drift = client.post("/data/ingest", headers=headers, files={"file": ("test_50_employees_drift.csv", f, "text/csv")})
    assert res_ingest_drift.status_code == 200
    drift_run_id = res_ingest_drift.json()["run_id"]
    assert drift_run_id != run_id
    
    # GET drift report after second run
    res_drift_2 = client.get("/monitoring/drift", headers=headers)
    assert res_drift_2.status_code == 200
    drift_2_data = res_drift_2.json()
    assert drift_2_data.get("cold_start") is False
    assert drift_2_data["max_psi"] > 0.0
    
    # 14. Test PII Masking
    res_hr_masked = client.get("/employees/high-risk?mask_pii=true", headers=headers)
    assert res_hr_masked.status_code == 200
    for emp in res_hr_masked.json():
        assert "***" in emp["employee_id"]
        
    # 15. Rollback test
    # First, register two models manually to populate history
    from models.registry import register_model
    from sklearn.ensemble import IsolationForest
    model_a = IsolationForest().fit([[1, 2], [3, 4]])
    register_model("anomaly_detector", "version_a", model_a, {"run_id": "run_a", "metrics": {"f1": 0.5}}, ["f1", "f2"], mark_production=True)
    model_b = IsolationForest().fit([[1, 2], [3, 4]])
    register_model("anomaly_detector", "version_b", model_b, {"run_id": "run_b", "metrics": {"f1": 0.6}}, ["f1", "f2"], mark_production=True)
    
    res_rollback = client.post("/models/rollback", headers=headers, json={"model_name": "anomaly_detector"})
    assert res_rollback.status_code == 200
    rollback_data = res_rollback.json()
    assert rollback_data["status"] == "success"

