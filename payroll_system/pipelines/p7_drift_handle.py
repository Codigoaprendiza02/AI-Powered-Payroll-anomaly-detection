import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config.settings import AUDIT_LOG_DIR
from pipelines.p1_feature import read_feature_store
from pipelines.p2_training import run_training_pipeline

def recalibrate_thresholds(df: pd.DataFrame) -> dict:
    """
    Computes per-column descriptive statistics and percentiles for numerical columns.
    """
    thresholds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = df[col]
        thresholds[col] = {
            "mean": float(series.mean()) if not pd.isna(series.mean()) else 0.0,
            "std": float(series.std()) if not pd.isna(series.std()) else 0.0,
            "median": float(series.median()) if not pd.isna(series.median()) else 0.0,
            "p25": float(series.quantile(0.25)) if not pd.isna(series.quantile(0.25)) else 0.0,
            "p75": float(series.quantile(0.75)) if not pd.isna(series.quantile(0.75)) else 0.0,
            "percentiles": {
                "25": float(series.quantile(0.25)) if not pd.isna(series.quantile(0.25)) else 0.0,
                "50": float(series.quantile(0.50)) if not pd.isna(series.quantile(0.50)) else 0.0,
                "75": float(series.quantile(0.75)) if not pd.isna(series.quantile(0.75)) else 0.0,
                "90": float(series.quantile(0.90)) if not pd.isna(series.quantile(0.90)) else 0.0,
                "95": float(series.quantile(0.95)) if not pd.isna(series.quantile(0.95)) else 0.0,
                "99": float(series.quantile(0.99)) if not pd.isna(series.quantile(0.99)) else 0.0
            }
        }
    return thresholds

def run_drift_handling(run_id: str, drift_result: dict, alert_manager):
    """
    Step 7: Drift Handling Pipeline. Responds to drift severity:
    - Skipped / Cold start / No drift -> Log skipped.
    - Moderate Drift -> Recalibrate thresholds.
    - Significant Drift / Accuracy Decay > 20% -> Recalibrate thresholds and retrain.
    """
    log_path = Path(AUDIT_LOG_DIR) / f"{run_id}_drift_handling_log.json"
    
    # 1. Skip if no drift or cold start
    if not drift_result.get("drift_detected") or drift_result.get("cold_start"):
        print("No drift — skipping.")
        skip_log = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "strategy_applied": "skipped — no drift",
            "retrain_triggered": False,
            "action": "skipped — no drift",
            "reason": "Cold start or no drift detected"
        }
        with open(log_path, "w") as f:
            json.dump(skip_log, f, indent=2)
        return
        
    severity = drift_result.get("severity", "No Drift")
    accuracy_decay_pct = drift_result.get("accuracy_decay_pct", 0.0)
    
    # Load current features for threshold recalibration
    df_feat = read_feature_store(run_id)
    
    strategy_applied = "skipped"
    retrain_triggered = False
    action = "none"
    retrain_outcome = {}
    
    # 2. Decide Handling Strategy
    # If accuracy decay > 20% or Significant drift: retrain + recalibrate
    if accuracy_decay_pct > 20.0 or severity == "Significant Drift":
        strategy_applied = "full_retrain"
        retrain_triggered = True
        action = "recalibrate_and_retrain"
        
        # Recalibrate Thresholds
        thresholds = recalibrate_thresholds(df_feat)
        thresholds_path = Path(__file__).parent.parent / "config" / "thresholds_current.json"
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
            
        # Trigger Retrain
        try:
            retrain_report = run_training_pipeline(run_id, alert_manager)
            retrain_outcome = {
                "status": "success",
                "promoted_models": list(retrain_report.get("models", {}).keys())
            }
        except Exception as e:
            retrain_outcome = {
                "status": "failed",
                "error": str(e)
            }
            
        if accuracy_decay_pct > 20.0:
            alert_manager.emit(
                alert_type="CRITICAL_MODEL_HEALTH_ALERT",
                severity="CRITICAL",
                affected_entity="company_payroll_forecaster",
                trigger_value=accuracy_decay_pct,
                threshold_value=20.0,
                recommended_action="Emergency model retrain completed due to extreme accuracy decay."
            )
            
    elif severity == "Moderate Drift" or (accuracy_decay_pct > 5.0 and accuracy_decay_pct <= 20.0):
        strategy_applied = "threshold_recalibration"
        retrain_triggered = False
        action = "threshold_recalibration"
        
        # Recalibrate Thresholds only
        thresholds = recalibrate_thresholds(df_feat)
        thresholds_path = Path(__file__).parent.parent / "config" / "thresholds_current.json"
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
            
    # 3. Save drift handling log
    handling_log = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_applied": strategy_applied,
        "retrain_triggered": retrain_triggered,
        "action": action,
        "severity": severity,
        "accuracy_decay_pct": accuracy_decay_pct,
        "retrain_outcome": retrain_outcome
    }
    with open(log_path, "w") as f:
        json.dump(handling_log, f, indent=2)
