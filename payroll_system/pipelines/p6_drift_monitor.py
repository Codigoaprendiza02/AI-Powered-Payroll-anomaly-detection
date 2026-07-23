import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import chi2_contingency

from config.settings import DRIFT_STORE_DIR, AUDIT_LOG_DIR, FORECASTS_DIR, FEATURES_STORE_DIR
from pipelines.p1_feature import read_feature_store, save_reference_distributions

def compute_psi(reference, current, bins=10):
    """
    Computes Population Stability Index (PSI) between reference and current series.
    """
    # Filter out NaNs and infs to avoid histogram issues
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    
    if len(reference) == 0 or len(current) == 0:
        return 0.0
        
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(len(reference), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(len(current), 1))
    
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def compute_chi2_pvalue(ref_counts: dict, cur_counts: dict) -> float:
    """
    Computes Chi-squared contingency test p-value between two value count dictionaries.
    """
    all_cats = sorted(list(set(ref_counts.keys()).union(set(cur_counts.keys()))))
    if len(all_cats) <= 1:
        return 1.0
        
    ref_vec = [ref_counts.get(cat, 0) for cat in all_cats]
    cur_vec = [cur_counts.get(cat, 0) for cat in all_cats]
    
    if sum(ref_vec) == 0 or sum(cur_vec) == 0:
        return 1.0
        
    table = np.array([ref_vec, cur_vec])
    # Add tiny epsilon to avoid absolute zero row/column contingency table issues
    table = table + 1e-9
    
    try:
        chi2, p_val, dof, expected = chi2_contingency(table)
        return float(p_val)
    except Exception:
        return 1.0

def run_drift_monitoring(run_id: str, alert_manager) -> dict:
    """
    Step 6: Drift Monitoring Pipeline. Computes PSI for numerical features,
    performs Chi-squared tests for categorical features, and tracks forecast accuracy decay.
    """
    latest_ref_path = Path(DRIFT_STORE_DIR) / "latest_reference.json"
    
    # 1. Check for Cold Start
    if not latest_ref_path.exists():
        # Generate the first run's reference distribution file
        # We need to make sure we also append final_anomaly_score if possible
        try:
            ref_path = Path(DRIFT_STORE_DIR) / f"reference_{run_id}.json"
            if ref_path.exists():
                with open(ref_path, "r") as f:
                    ref_data = json.load(f)
            else:
                # Fallback if reference file wasn't written
                df_feat = read_feature_store(run_id)
                save_reference_distributions(df_feat, run_id)
                with open(ref_path, "r") as f:
                    ref_data = json.load(f)
                    
            # Try to add final_anomaly_score distribution
            inf_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
            if inf_path.exists():
                inf_df = pd.read_parquet(inf_path)
                if "final_anomaly_score" in inf_df.columns:
                    score = inf_df["final_anomaly_score"]
                    ref_data["distributions"]["final_anomaly_score"] = {
                        "mean": float(score.mean()) if not pd.isna(score.mean()) else 0.0,
                        "median": float(score.median()) if not pd.isna(score.median()) else 0.0,
                        "std": float(score.std()) if not pd.isna(score.std()) else 0.0,
                        "p25": float(score.quantile(0.25)) if not pd.isna(score.quantile(0.25)) else 0.0,
                        "p75": float(score.quantile(0.75)) if not pd.isna(score.quantile(0.75)) else 0.0
                    }
                    with open(ref_path, "w") as f:
                        json.dump(ref_data, f, indent=2)
            
            # Copy to latest_reference
            with open(latest_ref_path, "w") as f:
                json.dump(ref_data, f, indent=2)
        except Exception:
            pass
            
        alert_manager.emit(
            alert_type="COLD_START_NOTICE",
            severity="INFO",
            affected_entity="model_monitoring",
            recommended_action="First run recorded as baseline reference distributions. Drift monitoring skipped."
        )
        return {"drift_detected": False, "cold_start": True}
        
    # 2. Load Reference distributions
    with open(latest_ref_path, "r") as f:
        ref_data = json.load(f)
    ref_run_id = ref_data["run_id"]
    
    # 3. Load dataframes
    curr_features = read_feature_store(run_id)
    ref_features_path = Path(FEATURES_STORE_DIR) / f"{ref_run_id}_features.parquet"
    if ref_features_path.exists():
        ref_features = pd.read_parquet(ref_features_path)
    else:
        ref_features = curr_features # fallback if missing
        
    curr_inf_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
    curr_inference = pd.read_parquet(curr_inf_path) if curr_inf_path.exists() else pd.DataFrame()
    
    ref_inf_path = Path(AUDIT_LOG_DIR) / f"{ref_run_id}_inference_results.parquet"
    ref_inference = pd.read_parquet(ref_inf_path) if ref_inf_path.exists() else pd.DataFrame()
    
    # 4. Compute PSI for numerical features
    feature_psi_map = {}
    numerical_cols = curr_features.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        if col in ref_features.columns:
            psi_val = compute_psi(ref_features[col], curr_features[col])
            feature_psi_map[col] = psi_val
            
    # Include final_anomaly_score in PSI computation
    if "final_anomaly_score" in curr_inference.columns and "final_anomaly_score" in ref_inference.columns:
        psi_val = compute_psi(ref_inference["final_anomaly_score"], curr_inference["final_anomaly_score"])
        feature_psi_map["final_anomaly_score"] = psi_val
        
    if not feature_psi_map:
        return {"drift_detected": False, "cold_start": False, "severity": "No Drift", "max_psi": 0.0}
        
    max_psi = max(feature_psi_map.values())
    
    # 5. Classify Severity
    if max_psi < 0.1:
        overall_severity = "No Drift"
    elif max_psi <= 0.25:
        overall_severity = "Moderate Drift"
    else:
        overall_severity = "Significant Drift"
        
    # 6. Categorical Drift
    cat_p_values = {}
    for col in ["department", "designation"]:
        ref_counts = ref_data.get("value_counts", {}).get(col, {})
        if col in curr_features.columns:
            cur_counts = curr_features[col].value_counts().to_dict()
            p_val = compute_chi2_pvalue(ref_counts, cur_counts)
            cat_p_values[col] = p_val
            
    # 7. Compute accuracy decay
    accuracy_decay_pct = 0.0
    prior_mape = None
    history_path = Path(FORECASTS_DIR) / "history.json"
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history_data = json.load(f)
                if len(history_data) >= 2:
                    prior_mape = history_data[-2].get("forecast_mape")
        except Exception:
            pass
            
    current_mape = None
    training_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_training_report.json"
    if training_report_path.exists():
        try:
            with open(training_report_path, "r") as f:
                tr = json.load(f)
                current_mape = tr.get("models", {}).get("company_payroll_forecaster", {}).get("metrics", {}).get("mape")
        except Exception:
            pass
            
    if prior_mape is not None and current_mape is not None:
        delta = current_mape - prior_mape
        accuracy_decay_pct = float(delta * 100.0)
        
        if delta > 0.05:
            alert_manager.emit(
                alert_type="FORECAST_ACCURACY_DECAY_ALERT",
                severity="HIGH",
                affected_entity="company_payroll_forecaster",
                trigger_value=current_mape,
                threshold_value=prior_mape + 0.05,
                recommended_action=f"Forecast MAPE increased by {delta*100:.2f}% (from {prior_mape*100:.2f}% to {current_mape*100:.2f}%). Recalibrating thresholds."
            )
        if delta > 0.20:
            alert_manager.emit(
                alert_type="CRITICAL_MODEL_HEALTH_ALERT",
                severity="CRITICAL",
                affected_entity="company_payroll_forecaster",
                trigger_value=current_mape,
                threshold_value=prior_mape + 0.20,
                recommended_action=f"Critical MAPE decay of {delta*100:.2f}% detected. Triggering emergency model retrain."
            )
            
    # 8. Top-3 drifted features
    sorted_features = sorted(feature_psi_map.items(), key=lambda x: x[1], reverse=True)
    top_features = []
    for feat, psi in sorted_features[:3]:
        # Compute mean change pct
        if feat == "final_anomaly_score":
            curr_mean = float(curr_inference["final_anomaly_score"].mean()) if not curr_inference.empty else 0.0
            ref_mean = float(ref_inference["final_anomaly_score"].mean()) if not ref_inference.empty else 0.0
        else:
            curr_mean = float(curr_features[feat].mean()) if feat in curr_features.columns else 0.0
            ref_mean = float(ref_features[feat].mean()) if feat in ref_features.columns else 0.0
            
        mean_change_pct = abs(curr_mean - ref_mean) / max(abs(ref_mean), 1e-6) * 100
        top_features.append({
            "feature": feat,
            "psi": psi,
            "mean_change_pct": float(mean_change_pct)
        })
        
    # 9. Root Cause
    root_cause = None
    if top_features:
        root_cause = {
            "feature": top_features[0]["feature"],
            "psi": top_features[0]["psi"],
            "mean_change_pct": top_features[0]["mean_change_pct"]
        }
        
    # 10. Emit alerts based on PSI thresholds
    for col, psi_val in feature_psi_map.items():
        # Salary-related feature check
        is_salary = "salary" in col.lower() or "pay" in col.lower() or "base" in col.lower()
        is_ot = "overtime" in col.lower() or "ot" in col.lower()
        
        if is_salary and psi_val > 0.25:
            alert_manager.emit(
                alert_type="SALARY_DRIFT_ALERT",
                severity="HIGH",
                affected_entity=col,
                trigger_value=psi_val,
                threshold_value=0.25,
                recommended_action=f"Salary feature '{col}' drifted with PSI={psi_val:.3f}. Auditing salary ranges."
            )
        if is_ot and psi_val > 0.25:
            alert_manager.emit(
                alert_type="OVERTIME_DRIFT_ALERT",
                severity="HIGH",
                affected_entity=col,
                trigger_value=psi_val,
                threshold_value=0.25,
                recommended_action=f"Overtime feature '{col}' drifted with PSI={psi_val:.3f}. Reviewing overtime policies."
            )
        if psi_val > 10.0:
            alert_manager.emit(
                alert_type="PAYROLL_DATA_DRIFT_ALERT",
                severity="CRITICAL",
                affected_entity=col,
                trigger_value=psi_val,
                threshold_value=10.0,
                recommended_action=f"Catastrophic drift in feature '{col}' (PSI={psi_val:.3f}). Triggering retrain."
            )
            
    # 11. Append to psi_history.json
    psi_history_path = Path(DRIFT_STORE_DIR) / "psi_history.json"
    psi_history = []
    if psi_history_path.exists():
        try:
            with open(psi_history_path, "r") as f:
                psi_history = json.load(f)
        except Exception:
            pass
            
    psi_history.append({
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_psi_map": feature_psi_map,
        "overall_severity": overall_severity
    })
    
    with open(psi_history_path, "w") as f:
        json.dump(psi_history, f, indent=2)
        
    # 12. Save latest_drift_severity.json
    # Map overall_severity to compact string for forecasting multiplier compat
    compact_severity = "Significant" if overall_severity == "Significant Drift" else ("Moderate" if overall_severity == "Moderate Drift" else "No Drift")
    latest_severity_path = Path(DRIFT_STORE_DIR) / "latest_drift_severity.json"
    with open(latest_severity_path, "w") as f:
        json.dump({
            "severity": compact_severity,
            "max_psi": max_psi,
            "top_features": top_features,
            "accuracy_decay_pct": accuracy_decay_pct
        }, f, indent=2)
        
    # 13. Save drift report to audit log
    drift_report = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "overall_severity": overall_severity,
        "max_psi": max_psi,
        "top_features": top_features,
        "accuracy_decay_pct": accuracy_decay_pct,
        "feature_psi_map": feature_psi_map,
        "categorical_drift_p_values": cat_p_values,
        "root_cause": root_cause,
        "drift_detected": overall_severity != "No Drift",
        "cold_start": False
    }
    drift_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_drift_report.json"
    with open(drift_report_path, "w") as f:
        json.dump(drift_report, f, indent=2)
        
    # 14. Update latest_reference.json with the current run's reference
    curr_ref_path = Path(DRIFT_STORE_DIR) / f"reference_{run_id}.json"
    if curr_ref_path.exists():
        try:
            with open(curr_ref_path, "r") as f:
                curr_ref_data = json.load(f)
            # Add final_anomaly_score distribution to latest reference
            if "final_anomaly_score" in curr_inference.columns:
                score = curr_inference["final_anomaly_score"]
                curr_ref_data["distributions"]["final_anomaly_score"] = {
                    "mean": float(score.mean()) if not pd.isna(score.mean()) else 0.0,
                    "median": float(score.median()) if not pd.isna(score.median()) else 0.0,
                    "std": float(score.std()) if not pd.isna(score.std()) else 0.0,
                    "p25": float(score.quantile(0.25)) if not pd.isna(score.quantile(0.25)) else 0.0,
                    "p75": float(score.quantile(0.75)) if not pd.isna(score.quantile(0.75)) else 0.0
                }
                # Overwrite run reference
                with open(curr_ref_path, "w") as f:
                    json.dump(curr_ref_data, f, indent=2)
            
            with open(latest_ref_path, "w") as f:
                json.dump(curr_ref_data, f, indent=2)
        except Exception:
            pass
            
    return {
        "drift_detected": overall_severity != "No Drift",
        "severity": overall_severity,
        "max_psi": max_psi,
        "top_features": top_features,
        "accuracy_decay_pct": accuracy_decay_pct,
        "cold_start": False
    }
