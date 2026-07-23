import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import AUDIT_LOG_DIR, FORECASTS_DIR

class AlertManager:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.alerts = []
        self._seen_keys = set()

    def emit(self, alert_type: str, severity: str, affected_entity: str,
             trigger_value=None, threshold_value=None, recommended_action: str = ""):
        key = f"{alert_type}::{affected_entity}"
        if key in self._seen_keys:
            return   # deduplicate within a run
        self._seen_keys.add(key)
        
        # Convert numeric values to standard python types to avoid serialization issues
        if isinstance(trigger_value, (np.integer, np.floating)):
            trigger_value = float(trigger_value)
        if isinstance(threshold_value, (np.integer, np.floating)):
            threshold_value = float(threshold_value)
            
        self.alerts.append({
            "alert_type": alert_type,
            "severity": severity,
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "affected_entity": str(affected_entity),
            "trigger_value": trigger_value,
            "threshold_value": threshold_value,
            "recommended_action": recommended_action
        })

    def save(self):
        path = Path(AUDIT_LOG_DIR) / f"{self.run_id}_alerts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.alerts, f, indent=2)

    def get_all(self) -> list:
        return self.alerts

def generate_run_summary(run_id: str, alert_manager: AlertManager) -> dict:
    summary_path = Path(AUDIT_LOG_DIR) / f"{run_id}_summary.json"
    
    # 1. Load inference results
    inference_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
    inference_summary = {}
    if inference_path.exists():
        df_inf = pd.read_parquet(inference_path)
        
        # Count risk tiers
        risk_counts = df_inf["risk_tier"].value_counts().to_dict()
        # Convert numpy int64 counts to standard python ints
        risk_counts = {k: int(v) for k, v in risk_counts.items()}
        
        # Calculate anomaly rate (HIGH and CRITICAL)
        anom_mask = df_inf["risk_tier"].isin(["HIGH", "CRITICAL"])
        anom_rate = float(anom_mask.mean()) if len(df_inf) > 0 else 0.0
        
        inference_summary = {
            "total_records": len(df_inf),
            "anomaly_rate": anom_rate,
            "risk_tiers": risk_counts
        }
        
    # 2. Load risk report
    risk_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_risk_report.csv"
    risk_summary = {}
    if risk_report_path.exists():
        df_risk = pd.read_csv(risk_report_path)
        risk_summary = {
            "absenteeism_risk_count": int(df_risk["absenteeism_flag"].sum()) if "absenteeism_flag" in df_risk.columns else 0,
            "overtime_abuse_count": int(df_risk["overtime_abuse_flag"].sum()) if "overtime_abuse_flag" in df_risk.columns else 0,
            "salary_manipulation_count": int(df_risk["salary_manipulation_flag"].sum()) if "salary_manipulation_flag" in df_risk.columns else 0
        }
        
    # 3. Load forecast
    forecast_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.json"
    forecast_summary = {}
    if forecast_path.exists():
        try:
            with open(forecast_path, "r") as f:
                forecast_summary = json.load(f)
        except Exception:
            forecast_summary = {}
            
    # 4. Load drift report
    drift_path = Path(AUDIT_LOG_DIR) / f"{run_id}_drift_report.json"
    drift_summary = {}
    if drift_path.exists():
        try:
            with open(drift_path, "r") as f:
                drift_summary = json.load(f)
        except Exception:
            drift_summary = {}

    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "inference": inference_summary,
        "risk_monitoring": risk_summary,
        "forecasting": forecast_summary,
        "drift_monitoring": drift_summary,
        "alerts_count": len(alert_manager.get_all())
    }
    
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

def generate_risk_report(run_id: str) -> str:
    risk_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_risk_report.csv"
    high_risk_path = Path(AUDIT_LOG_DIR) / f"{run_id}_high_risk_employees.csv"
    
    # Load explanations from inference results
    explanations_map = {}
    inference_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
    if inference_path.exists():
        df_inf = pd.read_parquet(inference_path)
        emp_col = "employee_id" if "employee_id" in df_inf.columns else df_inf.index.name
        if not emp_col:
            emp_col = "index"
        for _, r in df_inf.iterrows():
            explanations_map[str(r[emp_col])] = r.get("explanation", "")
            
    df = None
    if risk_report_path.exists():
        df = pd.read_csv(risk_report_path)
    elif inference_path.exists():
        df_inf = pd.read_parquet(inference_path)
        df = pd.DataFrame({
            "employee_id": df_inf["employee_id"] if "employee_id" in df_inf.columns else df_inf.index,
            "department": df_inf["department"] if "department" in df_inf.columns else "Unknown",
            "designation": df_inf["designation"] if "designation" in df_inf.columns else "Unknown",
            "absenteeism_score": 0.0,
            "absenteeism_flag": False,
            "overtime_hours": 0.0,
            "overtime_abuse_flag": False,
            "salary_manipulation_score": 0.0,
            "salary_manipulation_flag": False,
            "final_anomaly_score": df_inf["final_anomaly_score"],
            "risk_tier": df_inf["risk_tier"]
        })
        
    if df is not None:
        # Add explanation column to report
        df["explanation"] = df["employee_id"].astype(str).map(explanations_map).fillna("")
        high_risk_df = df[df["risk_tier"].isin(["HIGH", "CRITICAL"])]
        high_risk_path.parent.mkdir(parents=True, exist_ok=True)
        high_risk_df.to_csv(high_risk_path, index=False)
    else:
        headers = ["employee_id", "department", "designation", "final_anomaly_score", "risk_tier", "explanation"]
        high_risk_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=headers).to_csv(high_risk_path, index=False)
        
    return str(high_risk_path)

def generate_forecast_report(run_id: str) -> dict:
    json_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.json"
    csv_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.csv"
    
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            rows = []
            rows.append({"metric": "total_payroll", "department": "All", "value": data.get("total_payroll")})
            rows.append({"metric": "overtime_hours", "department": "All", "value": data.get("overtime_hours")})
            rows.append({"metric": "drift_adjusted", "department": "All", "value": str(data.get("drift_adjusted"))})
            
            dept_preds = data.get("per_department", {})
            for dept, val in dept_preds.items():
                rows.append({"metric": "department_payroll", "department": dept, "value": val})
                
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception:
            pd.DataFrame(columns=["metric", "department", "value"]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["metric", "department", "value"]).to_csv(csv_path, index=False)
        
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path)
    }

def mark_run_complete(run_id: str):
    runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
    if not runs_file.exists():
        return
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
            
        for run in runs:
            if run["run_id"] == run_id:
                run["status"] = "COMPLETE"
                run["completed_at"] = datetime.utcnow().isoformat()
                for step in run.get("pipeline_steps", []):
                    if step["status"] == "RUNNING" or step["pipeline_step"] == "alerting":
                        step["status"] = "COMPLETE"
                        step["timestamp"] = datetime.utcnow().isoformat()
                break
                
        with open(runs_file, "w") as f:
            json.dump(runs, f, indent=2)
    except Exception:
        pass

def run_alerting_pipeline(run_id: str, alert_manager: AlertManager):
    """
    Step 8: Alerting & Output Pipeline.
    Aggregates all outputs and generates reports.
    """
    generate_run_summary(run_id, alert_manager)
    generate_risk_report(run_id)
    generate_forecast_report(run_id)
    alert_manager.save()
    mark_run_complete(run_id)
