import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body, Query
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from config.settings import (
    API_KEY, AUDIT_LOG_DIR, FORECASTS_DIR, RISK_REGISTER_DIR, 
    FEATURES_STORE_DIR, MODELS_REGISTRY_DIR, DRIFT_STORE_DIR
)
from pipelines.pipeline_runner import run_full_pipeline

app = FastAPI(title="Payroll Intelligence & Anomaly Detection System", version="3.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Custom header dependency for API key verification
def verify_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

# Helper to load run summary
def load_run_summary(run_id: str) -> dict:
    summary_path = Path(AUDIT_LOG_DIR) / f"{run_id}_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"run_id": run_id, "status": "UNKNOWN"}

# Mask PII helper (employee_id)
def mask_employee_id(emp_id: str) -> str:
    # If config allows, mask employee ID middle characters (e.g. EMP0010 -> EM***10)
    emp_str = str(emp_id)
    if len(emp_str) <= 4:
        return "****"
    return f"{emp_str[:2]}***{emp_str[-2:]}"

@app.post("/data/ingest", dependencies=[Depends(verify_key)])
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in [".csv", ".json", ".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        run_id = run_full_pipeline(tmp_path)
        summary = load_run_summary(run_id)
        
        # Check run status from runs.json
        status = "COMPLETE"
        runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
        if runs_file.exists():
            with open(runs_file, "r") as f:
                runs = json.load(f)
            for r in runs:
                if r["run_id"] == run_id:
                    status = r["status"]
                    break
                    
        return {"run_id": run_id, "status": status, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/runs/{run_id}", dependencies=[Depends(verify_key)])
def get_run(run_id: str):
    runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
    if not runs_file.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
        for r in runs:
            if r["run_id"] == run_id:
                summary = {}
                if r["status"] == "COMPLETE":
                    summary = load_run_summary(run_id)
                return {
                    "run_id": r["run_id"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                    "error": r["error"],
                    "pipeline_steps": r.get("pipeline_steps", []),
                    "summary": summary
                }
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Run not found.")

@app.get("/runs/{run_id}/alerts", dependencies=[Depends(verify_key)])
def get_run_alerts(run_id: str):
    alerts_file = Path(AUDIT_LOG_DIR) / f"{run_id}_alerts.json"
    if not alerts_file.exists():
        raise HTTPException(status_code=404, detail="Alerts not found for run.")
    try:
        with open(alerts_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}/report", dependencies=[Depends(verify_key)])
def get_run_report(run_id: str):
    report_file = Path(AUDIT_LOG_DIR) / f"{run_id}_summary.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found for run.")
    try:
        with open(report_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/high-risk", dependencies=[Depends(verify_key)])
def get_high_risk_employees(mask_pii: bool = Query(False)):
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if not last_run_file.exists():
        return []
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
        
    high_risk_csv = Path(AUDIT_LOG_DIR) / f"{run_id}_high_risk_employees.csv"
    if not high_risk_csv.exists():
        return []
        
    try:
        df = pd.read_csv(high_risk_csv)
        records = df.to_dict(orient="records")
        if mask_pii:
            for r in records:
                r["employee_id"] = mask_employee_id(r["employee_id"])
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/{employee_id}/risk", dependencies=[Depends(verify_key)])
def get_employee_risk(employee_id: str, mask_pii: bool = Query(False)):
    register_path = Path(RISK_REGISTER_DIR) / "register.json"
    if not register_path.exists():
        raise HTTPException(status_code=404, detail="Employee not found in risk register.")
        
    try:
        with open(register_path, "r") as f:
            register = json.load(f)
            
        entry = next((e for e in register if e["employee_id"] == employee_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Employee not found in risk register.")
            
        # Fetch latest explanation from inference results
        explanation = ""
        last_run_id = entry.get("last_run_id")
        if last_run_id:
            inf_path = Path(AUDIT_LOG_DIR) / f"{last_run_id}_inference_results.parquet"
            if inf_path.exists():
                df_inf = pd.read_parquet(inf_path)
                emp_col = "employee_id" if "employee_id" in df_inf.columns else df_inf.index.name
                if not emp_col:
                    emp_col = "index"
                row = df_inf[df_inf[emp_col].astype(str) == employee_id]
                if not row.empty:
                    explanation = row.iloc[0].get("explanation", "")
                    
        emp_id_val = mask_employee_id(entry["employee_id"]) if mask_pii else entry["employee_id"]
        
        return {
            "employee_id": emp_id_val,
            "last_run_id": entry["last_run_id"],
            "current_score": entry["current_score"],
            "trend": entry["trend"],
            "risk_dimensions": entry["risk_dimensions"],
            "explanation": explanation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/{employee_id}/history", dependencies=[Depends(verify_key)])
def get_employee_history(employee_id: str):
    register_path = Path(RISK_REGISTER_DIR) / "register.json"
    if not register_path.exists():
        raise HTTPException(status_code=404, detail="Employee not found in risk register.")
        
    try:
        with open(register_path, "r") as f:
            register = json.load(f)
            
        entry = next((e for e in register if e["employee_id"] == employee_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Employee not found in risk register.")
            
        return entry.get("history", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasts/latest", dependencies=[Depends(verify_key)])
def get_latest_forecast():
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if not last_run_file.exists():
        raise HTTPException(status_code=404, detail="No forecast runs found.")
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
        
    forecast_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.json"
    if not forecast_path.exists():
        raise HTTPException(status_code=404, detail="Forecast report not found.")
        
    try:
        with open(forecast_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/drift", dependencies=[Depends(verify_key)])
def get_latest_drift():
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if not last_run_file.exists():
        raise HTTPException(status_code=404, detail="No drift monitoring runs found.")
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
        
    drift_path = Path(AUDIT_LOG_DIR) / f"{run_id}_drift_report.json"
    if not drift_path.exists():
        return {"drift_detected": False, "cold_start": True, "overall_severity": "No Drift"}
        
    try:
        with open(drift_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/model-health", dependencies=[Depends(verify_key)])
def get_model_health():
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if not last_run_file.exists():
        raise HTTPException(status_code=404, detail="No model health logs found.")
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
        
    report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_training_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Model health report not found.")
        
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
            
        models_info = report.get("models", {})
        detector_info = models_info.get("anomaly_detector", {})
        forecaster_info = models_info.get("company_payroll_forecaster", {})
        
        return {
            "run_id": run_id,
            "timestamp": report.get("timestamp"),
            "model_version": detector_info.get("version_id", "unknown"),
            "forecaster_version": forecaster_info.get("version_id", "unknown"),
            "forecaster_mape": forecaster_info.get("metrics", {}).get("mape", 0.0),
            "anomaly_detector_f1": detector_info.get("metrics", {}).get("f1", 0.0),
            "stability_score": 0.4
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/rollback", dependencies=[Depends(verify_key)])
def rollback(payload: dict = Body(...)):
    model_name = payload.get("model_name", "anomaly_detector")
    try:
        from models.registry_manager import rollback_model
        result = rollback_model(model_name)
        return {
            "status": "success", 
            "model_version": result["model_version"], 
            "metadata": result.get("metadata", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema/report", dependencies=[Depends(verify_key)])
def get_schema_report():
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if not last_run_file.exists():
        raise HTTPException(status_code=404, detail="No runs found.")
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
        
    report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_quality_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Schema report not found.")
        
    try:
        with open(report_path, "r") as f:
            quality = json.load(f)
            
        return {
            "run_id": run_id,
            "schema_confidence": quality.get("schema_confidence", 0.0),
            "mapped_columns": quality.get("mapped_columns", {}),
            "unmapped_columns": quality.get("unmapped_columns", []),
            "warnings": quality.get("warnings", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    components = {
        "feature_store": Path(FEATURES_STORE_DIR).exists(),
        "model_registry": Path(MODELS_REGISTRY_DIR).exists(),
        "drift_store": Path(DRIFT_STORE_DIR).exists(),
        "risk_register": Path(RISK_REGISTER_DIR).exists(),
        "audit_log": Path(AUDIT_LOG_DIR).exists()
    }
    healthy = all(components.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "components": {k: "ok" if v else "missing" for k, v in components.items()}
    }
