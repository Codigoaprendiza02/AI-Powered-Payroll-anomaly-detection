import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression

from config.settings import FORECASTS_DIR, DRIFT_STORE_DIR, AUDIT_LOG_DIR
from pipelines.p1_feature import read_feature_store

def find_date_column(df: pd.DataFrame) -> str | None:
    """
    Detects if there is a date-like column in the dataframe.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
            
    date_indicators = ["date", "month", "year", "period", "timestamp", "time", "created_at", "pay_date"]
    for col in df.columns:
        col_lower = str(col).lower()
        if any(ind in col_lower for ind in date_indicators):
            try:
                pd.to_datetime(df[col].iloc[:5], errors='raise')
                return col
            except Exception:
                pass
                
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            continue
        try:
            pd.to_datetime(df[col].iloc[:5], errors='raise')
            return col
        except Exception:
            pass
            
    return None

def get_chronological_monthly_series(current_df: pd.DataFrame):
    """
    Aggregates company-level and department-level payroll and overtime historically.
    Supports both date-based grouping within a single run, and run-by-run grouping across multiple files.
    """
    from config.settings import FEATURES_STORE_DIR
    feature_dir = Path(FEATURES_STORE_DIR)
    if not feature_dir.exists():
        return [], [], {}
        
    parquet_files = sorted(list(feature_dir.glob("*_features.parquet")))
    
    company_payroll = []
    company_overtime = []
    dept_payroll = {} # dept -> list of float
    
    date_col = find_date_column(current_df)
    
    if date_col:
        # Group by year-month dynamically across all files to handle single file multi-month uploads
        monthly_data = {}
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                if df.empty:
                    continue
                d_col = find_date_column(df)
                if d_col:
                    df = df.copy()
                    df["__ym"] = pd.to_datetime(df[d_col]).dt.strftime("%Y-%m")
                    for ym, group in df.groupby("__ym"):
                        p_sum = float(group["net_salary"].sum()) if "net_salary" in group.columns else 0.0
                        o_sum = float(group["overtime_hours"].sum()) if "overtime_hours" in group.columns else 0.0
                        
                        dept_sums = {}
                        if "department" in group.columns and "net_salary" in group.columns:
                            dept_sums = group.groupby("department")["net_salary"].sum().to_dict()
                            
                        entry = monthly_data.setdefault(ym, {"pay": 0.0, "ot": 0.0, "depts": {}})
                        entry["pay"] += p_sum
                        entry["ot"] += o_sum
                        for dept, val in dept_sums.items():
                            entry["depts"][dept] = entry["depts"].get(dept, 0.0) + float(val)
                else:
                    # Fallback to single month for this file
                    file_ym = datetime.utcnow().strftime("%Y-%m")
                    p_sum = float(df["net_salary"].sum()) if "net_salary" in df.columns else 0.0
                    o_sum = float(df["overtime_hours"].sum()) if "overtime_hours" in df.columns else 0.0
                    
                    dept_sums = {}
                    if "department" in df.columns and "net_salary" in df.columns:
                        dept_sums = df.groupby("department")["net_salary"].sum().to_dict()
                        
                    entry = monthly_data.setdefault(file_ym, {"pay": 0.0, "ot": 0.0, "depts": {}})
                    entry["pay"] += p_sum
                    entry["ot"] += o_sum
                    for dept, val in dept_sums.items():
                        entry["depts"][dept] = entry["depts"].get(dept, 0.0) + float(val)
            except Exception:
                continue
                
        sorted_yms = sorted(list(monthly_data.keys()))
        for ym in sorted_yms:
            company_payroll.append(monthly_data[ym]["pay"])
            company_overtime.append(monthly_data[ym]["ot"])
            for dept, val in monthly_data[ym]["depts"].items():
                dept_payroll.setdefault(dept, []).append(val)
                
        # Fill missing months for depts with 0.0
        for dept in dept_payroll:
            dept_series = []
            for ym in sorted_yms:
                dept_series.append(monthly_data[ym]["depts"].get(dept, 0.0))
            dept_payroll[dept] = dept_series
            
    else:
        # Fall back to treating each features parquet file as a separate month chronologically
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                p_sum = float(df["net_salary"].sum()) if "net_salary" in df.columns else 0.0
                o_sum = float(df["overtime_hours"].sum()) if "overtime_hours" in df.columns else 0.0
                company_payroll.append(p_sum)
                company_overtime.append(o_sum)
                
                if "department" in df.columns and "net_salary" in df.columns:
                    dept_sums = df.groupby("department")["net_salary"].sum().to_dict()
                    all_depts = set(dept_payroll.keys()).union(set(dept_sums.keys()))
                    for dept in all_depts:
                        dept_list = dept_payroll.setdefault(dept, [])
                        while len(dept_list) < len(company_payroll) - 1:
                            dept_list.append(0.0)
                        dept_list.append(float(dept_sums.get(dept, 0.0)))
            except Exception:
                continue
                
        for dept, dept_list in dept_payroll.items():
            while len(dept_list) < len(company_payroll):
                dept_list.append(0.0)
                
    return company_payroll, company_overtime, dept_payroll

def run_forecasting_pipeline(run_id: str, alert_manager) -> dict:
    """
    Step 4: Forecasting Pipeline. Generates payroll expense and overtime forecasts.
    """
    # 1. Load current run features
    df = read_feature_store(run_id)
    
    # 2. Get company and department level chronological month series
    company_payroll, company_overtime, dept_payroll = get_chronological_monthly_series(df)
    n_months = len(company_payroll)
    
    # 3. Forecast company-wide total payroll (Linear Regression)
    if n_months == 0:
        pred_total_payroll = 0.0
    elif n_months == 1:
        pred_total_payroll = company_payroll[0]
    else:
        X = np.array(range(1, n_months + 1)).reshape(-1, 1)
        y = np.array(company_payroll)
        lr = LinearRegression()
        lr.fit(X, y)
        pred_total_payroll = float(lr.predict(np.array([[n_months + 1]]))[0])
        
    pred_total_payroll = max(0.0, pred_total_payroll)
    
    # 4. Forecast company-wide overtime hours
    if n_months == 0:
        pred_overtime_hours = 0.0
    elif n_months == 1:
        pred_overtime_hours = company_overtime[0]
    else:
        X_ot = np.array(range(1, n_months + 1)).reshape(-1, 1)
        y_ot = np.array(company_overtime)
        lr_ot = LinearRegression()
        lr_ot.fit(X_ot, y_ot)
        pred_overtime_hours = float(lr_ot.predict(np.array([[n_months + 1]]))[0])
        
    pred_overtime_hours = max(0.0, pred_overtime_hours)
    
    # 5. Compute company monthly growth rate (CAGR) for fallback
    # CAGR = (last_month / first_month) ^ (1/n) - 1
    if n_months <= 1:
        company_monthly_growth_rate = 0.0
    else:
        last_month = company_payroll[-1]
        first_month = company_payroll[0]
        n = n_months - 1
        if first_month > 0:
            val = last_month / first_month
            if val > 0:
                company_monthly_growth_rate = (val) ** (1.0 / n) - 1.0
            else:
                company_monthly_growth_rate = 0.0
        else:
            company_monthly_growth_rate = 0.0
            
    # 6. Forecast per-department total payroll
    pred_by_dept = {}
    
    # Get current active departments from the current df to predict for them
    current_depts = df["department"].unique() if "department" in df.columns else []
    
    for dept in current_depts:
        if not dept or dept == "Unknown":
            continue
        dept_history = dept_payroll.get(dept, [])
        # Find active history starting from first non-zero month
        non_zero_indices = [i for i, v in enumerate(dept_history) if v > 0]
        if non_zero_indices:
            first_non_zero = non_zero_indices[0]
            active_history = dept_history[first_non_zero:]
        else:
            active_history = []
            
        if len(active_history) >= 3:
            X_dept = np.array(range(1, len(active_history) + 1)).reshape(-1, 1)
            y_dept = np.array(active_history)
            lr_dept = LinearRegression()
            lr_dept.fit(X_dept, y_dept)
            pred_dept = float(lr_dept.predict(np.array([[len(active_history) + 1]]))[0])
        else:
            last_known = active_history[-1] if active_history else 0.0
            pred_dept = last_known * (1.0 + company_monthly_growth_rate)
            
        pred_by_dept[dept] = max(0.0, pred_dept)
        
    # 7. Apply drift multiplier adjustment from prior run
    drift_multiplier = 1.00
    drift_path = Path(DRIFT_STORE_DIR) / "latest_drift_severity.json"
    if drift_path.exists():
        try:
            with open(drift_path, "r") as f:
                drift_data = json.load(f)
                severity = drift_data.get("severity", "")
                if severity in ["Significant", "Significant Drift"]:
                    drift_multiplier = 1.10
                elif severity in ["Moderate", "Moderate Drift"]:
                    drift_multiplier = 1.05
        except Exception:
            pass
            
    final_pred_total_payroll = pred_total_payroll * drift_multiplier
    
    # 8. Check prior total for FORECAST_PAYROLL_SPIKE_ALERT
    prior_total = None
    history_path = Path(FORECASTS_DIR) / "history.json"
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history_data = json.load(f)
                if history_data and isinstance(history_data, list):
                    last_entry = history_data[-1]
                    prior_total = last_entry.get("last_actual_total")
                    if prior_total is None:
                        prior_total = last_entry.get("actual_total")
                    if prior_total is None:
                        prior_total = last_entry.get("predicted_total")
        except Exception:
            pass
            
    if prior_total is not None and prior_total > 0:
        if final_pred_total_payroll > prior_total * 1.20:
            alert_manager.emit(
                alert_type="FORECAST_PAYROLL_SPIKE_ALERT",
                severity="HIGH",
                affected_entity="company_payroll",
                trigger_value=final_pred_total_payroll,
                threshold_value=prior_total * 1.20,
                recommended_action=f"Upcoming payroll prediction {final_pred_total_payroll:.0f} is > 20% higher than prior total {prior_total:.0f}. Audit department budgets."
            )
            
    # 9. Build output dict
    forecast_report = {
        "total_payroll": final_pred_total_payroll,
        "per_department": pred_by_dept,
        "overtime_hours": pred_overtime_hours,
        "drift_adjusted": drift_multiplier > 1.00,
        "drift_multiplier_applied": drift_multiplier
    }
    
    # 10. Save forecast report to run file
    run_forecast_path = Path(FORECASTS_DIR) / f"{run_id}_forecast.json"
    with open(run_forecast_path, "w") as f:
        json.dump(forecast_report, f, indent=2)
        
    # 11. Append to forecasts history
    history_list = []
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history_list = json.load(f)
        except Exception:
            history_list = []
            
    # Look up forecast MAPE from training report
    forecast_mape = 0.0
    training_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_training_report.json"
    if training_report_path.exists():
        try:
            with open(training_report_path, "r") as f:
                tr = json.load(f)
                forecast_mape = tr.get("models", {}).get("company_payroll_forecaster", {}).get("metrics", {}).get("mape", 0.0)
        except Exception:
            pass

    actual_total_payroll = float(df["net_salary"].sum()) if "net_salary" in df.columns else 0.0
    
    history_entry = {
        "run_id": run_id,
        "predicted_total": final_pred_total_payroll,
        "predicted_by_dept": pred_by_dept,
        "predicted_overtime": pred_overtime_hours,
        "drift_multiplier_applied": drift_multiplier,
        "actual_total": actual_total_payroll,
        "last_actual_total": actual_total_payroll,
        "forecast_mape": forecast_mape,
        "timestamp": datetime.utcnow().isoformat()
    }
    history_list.append(history_entry)
    with open(history_path, "w") as f:
        json.dump(history_list, f, indent=2)
        
    return forecast_report
