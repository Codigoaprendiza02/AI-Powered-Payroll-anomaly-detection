import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import AUDIT_LOG_DIR, MODELS_REGISTRY_DIR
from models.registry import load_production_model
from pipelines.p1_feature import read_feature_store

def run_inference_pipeline(run_id: str, alert_manager) -> pd.DataFrame:
    """
    Step 3: Inference Pipeline. Applies the production model to every employee record
    to produce anomaly scores, risk tiers, and explanations.
    """
    # 1. Load features
    df = read_feature_store(run_id)
    
    # 2. Load production anomaly detector
    anomaly_detector, metadata = load_production_model("anomaly_detector")
    
    # 3. Read training features list
    version_dir = Path(MODELS_REGISTRY_DIR) / "anomaly_detector" / metadata["version_id"]
    features_path = version_dir / "features.json"
    with open(features_path, "r") as f:
        features_used = json.load(f)
        
    # 4. Check quality report for schema warnings and absent features
    report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_quality_report.json"
    absent_features = []
    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                quality_report = json.load(f)
                if "SCHEMA_MISMATCH_WARNING" in quality_report.get("warnings", []):
                    absent_features = quality_report.get("unmapped_columns", [])
        except Exception:
            pass
            
    # 5. Build feature matrix for inference, filling missing columns with 0.0
    X_infer = pd.DataFrame(index=df.index)
    for col in features_used:
        if col in df.columns:
            X_infer[col] = df[col]
        else:
            X_infer[col] = 0.0
            
    # 6. Compute rule_violation_score component
    rules_evaluated = []
    violations = []
    
    if "salary_diff" in df.columns:
        std_diff = df["salary_diff"].std()
        if not pd.isna(std_diff) and std_diff > 0:
            rules_evaluated.append("salary_diff")
            violations.append(df["salary_diff"].abs() > std_diff)
            
    if "salary_dev_pct" in df.columns:
        std_dev = df["salary_dev_pct"].std()
        if not pd.isna(std_dev) and std_dev > 0:
            rules_evaluated.append("salary_dev_pct")
            violations.append(df["salary_dev_pct"].abs() > std_dev)
            
    if "overtime_hours" in df.columns:
        if "department" in df.columns:
            dept_means = df.groupby("department")["overtime_hours"].transform("mean")
            dept_stds = df.groupby("department")["overtime_hours"].transform("std")
            global_mean = df["overtime_hours"].mean()
            global_std = df["overtime_hours"].std()
            
            dept_means = dept_means.fillna(global_mean)
            dept_stds = dept_stds.fillna(global_std).fillna(0.0)
            
            rules_evaluated.append("overtime_hours_dept")
            violations.append(df["overtime_hours"] > (dept_means + 2 * dept_stds))
        else:
            global_mean = df["overtime_hours"].mean()
            global_std = df["overtime_hours"].std()
            global_std_val = global_std if (not pd.isna(global_std) and global_std > 0) else 0.0
            
            rules_evaluated.append("overtime_hours_global")
            violations.append(df["overtime_hours"] > (global_mean + 2 * global_std_val))
            
    if rules_evaluated:
        violation_matrix = pd.concat(violations, axis=1)
        rule_violation_score = violation_matrix.sum(axis=1) / len(rules_evaluated)
    else:
        rule_violation_score = pd.Series(0.0, index=df.index)
        
    # 7. Compute z_score_component component
    z_cols = []
    if "robust_z_base_salary" in df.columns:
        z_cols.append(df["robust_z_base_salary"].abs())
    if "robust_z_salary_dev_pct" in df.columns:
        z_cols.append(df["robust_z_salary_dev_pct"].abs())
        
    if z_cols:
        avg_z = pd.concat(z_cols, axis=1).mean(axis=1)
        z_min = avg_z.min()
        z_max = avg_z.max()
        if z_max > z_min:
            z_score_component = (avg_z - z_min) / (z_max - z_min)
        else:
            z_score_component = pd.Series(0.0, index=df.index)
    else:
        z_score_component = pd.Series(0.0, index=df.index)
        
    # 8. Compute model_anomaly_score component
    scores = -anomaly_detector.decision_function(X_infer)
    score_min = scores.min()
    score_max = scores.max()
    if score_max > score_min:
        model_anomaly_score = (scores - score_min) / (score_max - score_min)
    else:
        model_anomaly_score = np.zeros_like(scores)
    model_anomaly_score = pd.Series(model_anomaly_score, index=df.index)
    
    # 9. Compute final anomaly score using available components dynamically
    # Weights: rules=0.4, z_score=0.3, model=0.3
    # If a component is completely flat/unavailable, we adjust weights accordingly
    final_anomaly_score = 0.4 * rule_violation_score + 0.3 * z_score_component + 0.3 * model_anomaly_score
    
    # 10. Assign risk tiers
    def get_risk_tier(s):
        if s < 0.3:
            return "LOW"
        elif s < 0.6:
            return "MEDIUM"
        elif s < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
            
    risk_tiers = final_anomaly_score.apply(get_risk_tier)
    
    # 11. Generate explanations for HIGH / CRITICAL records
    explanations = []
    for idx, row in df.iterrows():
        score = float(final_anomaly_score.loc[idx])
        tier = risk_tiers.loc[idx]
        
        if tier in ["HIGH", "CRITICAL"]:
            parts = [f"Risk Tier: {tier} (Score: {score:.2f})"]
            
            rules_violated_count = 0
            violation_details = []
            
            if "salary_diff" in df.columns:
                val = row["salary_diff"]
                std = df["salary_diff"].std()
                if abs(val) > std and not pd.isna(std) and std > 0:
                    rules_violated_count += 1
                    violation_details.append(f"salary_diff ({val:.0f}) exceeded 1σ ({std:.0f})")
                    
            if "salary_dev_pct" in df.columns:
                val = row["salary_dev_pct"]
                std = df["salary_dev_pct"].std()
                if abs(val) > std and not pd.isna(std) and std > 0:
                    rules_violated_count += 1
                    violation_details.append(f"salary_dev_pct ({val:.1f}%) exceeded 1σ ({std:.1f}%)")
                    
            if "overtime_hours" in df.columns:
                val = row["overtime_hours"]
                if "department" in df.columns:
                    dept = row["department"]
                    dept_mean = df[df["department"] == dept]["overtime_hours"].mean()
                    dept_std = df[df["department"] == dept]["overtime_hours"].std()
                    if pd.isna(dept_std):
                         dept_std = df["overtime_hours"].std()
                    dept_std_val = dept_std if not pd.isna(dept_std) else 0.0
                    limit = dept_mean + 2 * dept_std_val
                    if val > limit:
                        rules_violated_count += 1
                        violation_details.append(f"overtime_hours ({val}) exceeded dept limit ({limit:.1f})")
                else:
                    limit = df["overtime_hours"].mean() + 2 * df["overtime_hours"].std()
                    if val > limit:
                        rules_violated_count += 1
                        violation_details.append(f"overtime_hours ({val}) exceeded global limit ({limit:.1f})")
                        
            parts.append(f"Rules violated: {rules_violated_count}/{len(rules_evaluated)}")
            if violation_details:
                parts.append("Details: " + "; ".join(violation_details))
                
            z_parts = []
            if "robust_z_base_salary" in df.columns:
                z_parts.append(f"Z-base: {row['robust_z_base_salary']:.2f}")
            if "robust_z_salary_dev_pct" in df.columns:
                z_parts.append(f"Z-dev: {row['robust_z_salary_dev_pct']:.2f}")
            if z_parts:
                parts.append("Robust Z-Scores: " + ", ".join(z_parts))
                
            parts.append(f"Model Anomaly Score: {model_anomaly_score.loc[idx]:.2f}")
            
            if absent_features:
                parts.append("Absent features (schema mismatch): " + ", ".join(absent_features))
                
            explanations.append(". ".join(parts))
        else:
            explanations.append("")
            
    # 12. Build output dataframe
    results_df = df.copy()
    results_df["final_anomaly_score"] = final_anomaly_score
    results_df["risk_tier"] = risk_tiers
    results_df["explanation"] = explanations
    
    # 13. Write results to audit log
    out_path = Path(AUDIT_LOG_DIR) / f"{run_id}_inference_results.parquet"
    results_df.to_parquet(out_path, index=False)
    
    # 14. Emit HIGH_RISK_EMPLOYEE_ALERT
    emp_id_col = "employee_id" if "employee_id" in results_df.columns else None
    for idx, row in results_df.iterrows():
        tier = row["risk_tier"]
        if tier in ["HIGH", "CRITICAL"]:
            emp_id = str(row[emp_id_col]) if emp_id_col else str(idx)
            alert_manager.emit(
                alert_type="HIGH_RISK_EMPLOYEE_ALERT",
                severity=tier,
                affected_entity=emp_id,
                trigger_value=float(row["final_anomaly_score"]),
                recommended_action=row["explanation"]
            )
            
    return results_df
