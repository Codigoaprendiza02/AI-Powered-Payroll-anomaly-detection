import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config.settings import AUDIT_LOG_DIR, RISK_REGISTER_DIR, MODELS_REGISTRY_DIR
from models.registry import load_production_model
from pipelines.p1_feature import read_feature_store

def run_risk_pipeline(run_id: str, alert_manager) -> list:
    """
    Step 5: Risk Monitoring Pipeline. Computes absenteeism risk, overtime abuse,
    and salary manipulation risk. Updates the persistent risk register.
    """
    # 1. Load inference results (contains features and anomaly scores)
    inference_results_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
    if not inference_results_path.exists():
        raise FileNotFoundError(f"Inference results not found for run: {run_id}")
    results_df = pd.read_parquet(inference_results_path)
    
    # 2. Get columns mapping
    emp_id_col = "employee_id" if "employee_id" in results_df.columns else results_df.index.name
    if not emp_id_col:
        emp_id_col = "index"
        
    # 3. Load persistent risk register
    register_path = Path(RISK_REGISTER_DIR) / "register.json"
    register = []
    if register_path.exists():
        try:
            with open(register_path, "r") as f:
                register = json.load(f)
        except Exception:
            register = []
            
    # Build lookup dict for prior entries
    prior_entries = {e["employee_id"]: e for e in register}
    
    # ==========================================
    # Dimension 1: Absenteeism Risk
    # ==========================================
    absenteeism_scores = pd.Series(0.0, index=results_df.index)
    absenteeism_flags = pd.Series(False, index=results_df.index)
    
    try:
        abs_model, abs_meta = load_production_model("absenteeism_classifier")
        # Load features list
        version_dir = Path(MODELS_REGISTRY_DIR) / "absenteeism_classifier" / abs_meta["version_id"]
        features_path = version_dir / "features.json"
        with open(features_path, "r") as f:
            abs_features = json.load(f)
            
        # Label encode department and designation using metadata mappings
        df_encoded = results_df.copy()
        mappings = abs_meta.get("mappings", {})
        for col in ["department", "designation"]:
            if col in df_encoded.columns:
                mapping = mappings.get(col, {})
                df_encoded[col] = df_encoded[col].map(mapping).fillna(-1).astype(int)
                
        # Build features matrix
        X_abs = pd.DataFrame(index=results_df.index)
        for col in abs_features:
            if col in df_encoded.columns:
                X_abs[col] = df_encoded[col]
            else:
                X_abs[col] = 0.0
                
        # Predict probability
        proba = abs_model.predict_proba(X_abs)
        if proba.shape[1] == 2:
            absenteeism_scores = pd.Series(proba[:, 1], index=results_df.index)
        else:
            classes = abs_model.classes_
            if 1 in classes:
                absenteeism_scores = pd.Series(proba[:, 0], index=results_df.index)
            else:
                absenteeism_scores = pd.Series(0.0, index=results_df.index)
                
        absenteeism_flags = absenteeism_scores > 0.7
    except FileNotFoundError:
        pass
        
    # ==========================================
    # Dimension 2: Overtime Abuse Risk
    # ==========================================
    overtime_abuse_flags = pd.Series(False, index=results_df.index)
    if "overtime_hours" in results_df.columns:
        if "department" in results_df.columns:
            dept_medians = results_df.groupby("department")["overtime_hours"].transform("median")
            dept_stds = results_df.groupby("department")["overtime_hours"].transform("std")
            global_median = results_df["overtime_hours"].median()
            global_std = results_df["overtime_hours"].std()
            
            dept_medians = dept_medians.fillna(global_median).fillna(0.0)
            dept_stds = dept_stds.fillna(global_std).fillna(0.0)
            
            overtime_abuse_calc_flags = results_df["overtime_hours"] > (dept_medians + 2 * dept_stds)
        else:
            global_median = results_df["overtime_hours"].median()
            global_std = results_df["overtime_hours"].std()
            global_std_val = global_std if (not pd.isna(global_std) and global_std > 0) else 0.0
            overtime_abuse_calc_flags = results_df["overtime_hours"] > (global_median + 2 * global_std_val)
            
        # Check consecutive runs check
        for idx, row in results_df.iterrows():
            emp_id = str(row[emp_id_col])
            overtime_history_flag = False
            
            if emp_id in prior_entries:
                history = prior_entries[emp_id].get("history", [])
                if len(history) >= 2:
                    last_two = history[-2:]
                    flagged_both = True
                    for h in last_two:
                        dims = h.get("dimensions", {})
                        if not dims.get("overtime_abuse", False):
                            flagged_both = False
                            break
                    if flagged_both:
                        overtime_history_flag = True
                        
            overtime_abuse_flags.loc[idx] = overtime_abuse_calc_flags.loc[idx] or overtime_history_flag
            
    # ==========================================
    # Dimension 3: Salary Manipulation Risk
    # ==========================================
    salary_manipulation_scores = pd.Series(0.0, index=results_df.index)
    salary_manipulation_flags = pd.Series(False, index=results_df.index)
    
    try:
        manip_model, manip_meta = load_production_model("salary_manipulation_detector")
        # Load features list
        version_dir = Path(MODELS_REGISTRY_DIR) / "salary_manipulation_detector" / manip_meta["version_id"]
        features_path = version_dir / "features.json"
        with open(features_path, "r") as f:
            manip_features = json.load(f)
            
        X_manip = pd.DataFrame(index=results_df.index)
        for col in manip_features:
            if col in results_df.columns:
                X_manip[col] = results_df[col]
            else:
                X_manip[col] = 0.0
                
        decision_scores = manip_model.decision_function(X_manip)
        salary_manipulation_scores = pd.Series(decision_scores, index=results_df.index)
        salary_manipulation_flags = pd.Series(decision_scores < 0, index=results_df.index)
    except FileNotFoundError:
        pass
        
    # ==========================================
    # Risk Register Update & Escalation Alerting
    # ==========================================
    updated_register = []
    escalation_flagged = []
    
    for idx, row in results_df.iterrows():
        emp_id = str(row[emp_id_col])
        current_score = float(row["final_anomaly_score"])
        
        abs_flag = bool(absenteeism_flags.loc[idx])
        ot_flag = bool(overtime_abuse_flags.loc[idx])
        manip_flag = bool(salary_manipulation_flags.loc[idx])
        
        # Look up prior entry
        if emp_id in prior_entries:
            entry = prior_entries[emp_id]
            prior_history = entry.get("history", [])
            prior_score = entry.get("current_score", 0.0)
            
            if current_score > prior_score:
                trend = "increasing"
            elif current_score < prior_score:
                trend = "decreasing"
            else:
                trend = "stable"
                
            is_escalation = False
            if len(prior_history) >= 2:
                last_score = prior_history[-1].get("score", 0.0)
                second_last_score = prior_history[-2].get("score", 0.0)
                if current_score > last_score and last_score > second_last_score:
                    is_escalation = True
                    
            history_entry = {
                "run_id": run_id,
                "score": current_score,
                "trend": trend,
                "dimensions": {
                    "absenteeism": abs_flag,
                    "overtime_abuse": ot_flag,
                    "salary_manipulation": manip_flag
                }
            }
            
            updated_entry = {
                "employee_id": emp_id,
                "last_run_id": run_id,
                "current_score": current_score,
                "trend": trend,
                "risk_dimensions": {
                    "absenteeism": abs_flag,
                    "overtime_abuse": ot_flag,
                    "salary_manipulation": manip_flag
                },
                "history": prior_history + [history_entry]
            }
        else:
            # First time employee appears
            trend = "stable"
            is_escalation = False
            history_entry = {
                "run_id": run_id,
                "score": current_score,
                "trend": trend,
                "dimensions": {
                    "absenteeism": abs_flag,
                    "overtime_abuse": ot_flag,
                    "salary_manipulation": manip_flag
                }
            }
            updated_entry = {
                "employee_id": emp_id,
                "last_run_id": run_id,
                "current_score": current_score,
                "trend": trend,
                "risk_dimensions": {
                    "absenteeism": abs_flag,
                    "overtime_abuse": ot_flag,
                    "salary_manipulation": manip_flag
                },
                "history": [history_entry]
            }
            
        updated_register.append(updated_entry)
        
        # Emit alerts for this employee
        if abs_flag:
            alert_manager.emit(
                alert_type="ABSENTEEISM_RISK_ALERT",
                severity="MEDIUM",
                affected_entity=emp_id,
                trigger_value=float(absenteeism_scores.loc[idx]),
                threshold_value=0.7,
                recommended_action=f"Absenteeism risk score {absenteeism_scores.loc[idx]:.2f} exceeds threshold 0.7 for employee {emp_id}."
            )
            
        if manip_flag:
            alert_manager.emit(
                alert_type="SALARY_MANIPULATION_ALERT",
                severity="CRITICAL",
                affected_entity=emp_id,
                trigger_value=float(salary_manipulation_scores.loc[idx]),
                recommended_action=f"Salary manipulation detected for employee {emp_id}."
            )
            
        if is_escalation:
            alert_manager.emit(
                alert_type="RISK_ESCALATION_ALERT",
                severity="HIGH",
                affected_entity=emp_id,
                trigger_value=current_score,
                recommended_action=f"Risk score has escalated consecutively over 3 runs for employee {emp_id}."
            )
            escalation_flagged.append(emp_id)
            
    # Save updated register
    with open(register_path, "w") as f:
        json.dump(updated_register, f, indent=2)
        
    # 4. Generate store/audit_log/{run_id}_risk_report.csv
    report_df = pd.DataFrame({
        "employee_id": results_df[emp_id_col],
        "department": results_df["department"] if "department" in results_df.columns else "Unknown",
        "designation": results_df["designation"] if "designation" in results_df.columns else "Unknown",
        "absenteeism_score": absenteeism_scores,
        "absenteeism_flag": absenteeism_flags,
        "overtime_hours": results_df["overtime_hours"] if "overtime_hours" in results_df.columns else 0.0,
        "overtime_abuse_flag": overtime_abuse_flags,
        "salary_manipulation_score": salary_manipulation_scores,
        "salary_manipulation_flag": salary_manipulation_flags,
        "final_anomaly_score": results_df["final_anomaly_score"],
        "risk_tier": results_df["risk_tier"]
    })
    
    report_csv_path = Path(AUDIT_LOG_DIR) / f"{run_id}_risk_report.csv"
    report_df.to_csv(report_csv_path, index=False)
    
    # Return high or critical risk employee list
    high_risk_employees = report_df[report_df["risk_tier"].isin(["HIGH", "CRITICAL"])]["employee_id"].tolist()
    
    return high_risk_employees
