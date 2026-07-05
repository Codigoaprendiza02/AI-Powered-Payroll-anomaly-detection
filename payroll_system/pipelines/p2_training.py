import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_absolute_percentage_error, mean_squared_error

from config.settings import RANDOM_SEED, AUDIT_LOG_DIR, FEATURES_STORE_DIR
from models.registry import is_cold_start, register_model, load_production_model
from pipelines.p1_feature import read_feature_store

def get_rule_anomaly_target(df: pd.DataFrame) -> pd.Series:
    """
    Computes a rule-based anomaly target boolean Series based on:
    - |salary_diff| > 1 sigma
    - |salary_dev_pct| > 1 sigma
    - overtime_hours > 2 sigma from department mean (with company-wide fallback)
    """
    target = pd.Series(False, index=df.index)
    
    if "salary_diff" in df.columns:
        std_diff = df["salary_diff"].std()
        if not pd.isna(std_diff) and std_diff > 0:
            target = target | (df["salary_diff"].abs() > std_diff)
            
    if "salary_dev_pct" in df.columns:
        std_dev = df["salary_dev_pct"].std()
        if not pd.isna(std_dev) and std_dev > 0:
            target = target | (df["salary_dev_pct"].abs() > std_dev)
            
    if "overtime_hours" in df.columns:
        if "department" in df.columns:
            dept_means = df.groupby("department")["overtime_hours"].transform("mean")
            dept_stds = df.groupby("department")["overtime_hours"].transform("std")
            global_mean = df["overtime_hours"].mean()
            global_std = df["overtime_hours"].std()
            
            # Fill NaNs for small departments
            dept_means = dept_means.fillna(global_mean)
            dept_stds = dept_stds.fillna(global_std).fillna(0.0)
            
            target = target | (df["overtime_hours"] > (dept_means + 2 * dept_stds))
        else:
            global_mean = df["overtime_hours"].mean()
            global_std = df["overtime_hours"].std()
            global_std_val = global_std if (not pd.isna(global_std) and global_std > 0) else 0.0
            target = target | (df["overtime_hours"] > (global_mean + 2 * global_std_val))
            
    return target

def get_historical_data() -> tuple:
    """
    Scans FEATURES_STORE_DIR for all versioned Parquet feature stores,
    sorts them chronologically by filename, and aggregates:
      - monthly company payroll (sum of net_salary)
      - monthly company overtime (sum of overtime_hours)
      - monthly department payrolls (dept -> list of payroll totals)
    Returns:
      - company_payroll: list of float
      - company_overtime: list of float
      - dept_payroll: dict of dept_name -> list of float
    """
    feature_dir = Path(FEATURES_STORE_DIR)
    if not feature_dir.exists():
        return [], [], {}
        
    parquet_files = sorted(list(feature_dir.glob("*_features.parquet")))
    
    company_payroll = []
    company_overtime = []
    dept_payroll = {}
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            
            # Company payroll
            p_sum = float(df["net_salary"].sum()) if "net_salary" in df.columns else 0.0
            company_payroll.append(p_sum)
            
            # Company overtime
            o_sum = float(df["overtime_hours"].sum()) if "overtime_hours" in df.columns else 0.0
            company_overtime.append(o_sum)
            
            # Department payroll
            if "department" in df.columns and "net_salary" in df.columns:
                dept_sums = df.groupby("department")["net_salary"].sum().to_dict()
                # Track all unique departments seen so far
                all_depts = set(dept_payroll.keys()).union(set(dept_sums.keys()))
                for dept in all_depts:
                    dept_list = dept_payroll.setdefault(dept, [])
                    dept_list.append(float(dept_sums.get(dept, 0.0)))
        except Exception:
            continue
            
    return company_payroll, company_overtime, dept_payroll

def evaluate_classifier_cv(model_class, model_kwargs, X, y) -> dict:
    """
    Evaluates classifier models using 5-fold Stratified CV if n >= 50, else LOOCV.
    Isolation Forest is evaluated as an anomaly classifier.
    """
    n_samples = len(X)
    if n_samples < 5:
        # Fallback to train-set evaluation
        model = model_class(**model_kwargs)
        if isinstance(model, IsolationForest):
            model.fit(X)
            preds = model.predict(X)
            # Map sklearn Outliers output (1 = inlier, -1 = outlier) to binary (1 = anomaly, 0 = normal)
            y_pred = np.where(preds == -1, 1, 0)
        else:
            model.fit(X, y)
            y_pred = model.predict(X)
            
        return {
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "stability_score": 1.0
        }
        
    if n_samples >= 50:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    else:
        cv = LeaveOneOut()
        
    f1s, precs, recs, accs = [], [], [], []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        model = model_class(**model_kwargs)
        
        if isinstance(model, IsolationForest):
            model.fit(X_train_cv)
            preds = model.predict(X_test_cv)
            y_pred = np.where(preds == -1, 1, 0)
        else:
            # Handle SMOTE on training fold if needed
            from imblearn.over_sampling import SMOTE
            minority_count = int(y_train_cv.sum())
            normal_count = len(y_train_cv) - minority_count
            
            if minority_count >= 2 and normal_count >= 2:
                k = min(5, minority_count - 1)
                try:
                    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
                    X_train_res, y_train_res = smote.fit_resample(X_train_cv, y_train_cv)
                    model.fit(X_train_res, y_train_res)
                except Exception:
                    model.fit(X_train_cv, y_train_cv)
            else:
                model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_test_cv)
            
        f1s.append(f1_score(y_test_cv, y_pred, zero_division=0))
        precs.append(precision_score(y_test_cv, y_pred, zero_division=0))
        recs.append(recall_score(y_test_cv, y_pred, zero_division=0))
        accs.append(accuracy_score(y_test_cv, y_pred))
        
    f1_mean = float(np.mean(f1s))
    prec_mean = float(np.mean(precs))
    rec_mean = float(np.mean(recs))
    acc_mean = float(np.mean(accs))
    
    stability = float(1.0 - np.std(f1s)) if len(f1s) > 1 else 1.0
    
    return {
        "f1": f1_mean,
        "precision": prec_mean,
        "recall": rec_mean,
        "accuracy": acc_mean,
        "stability_score": stability
    }

def evaluate_regressor_cv(X, y) -> dict:
    """
    Evaluates forecasting regressors using CV.
    """
    n_samples = len(X)
    if n_samples < 5:
        # Fallback to train-set evaluation
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        mape = float(mean_absolute_percentage_error(y, preds)) if not np.any(y == 0) else 0.0
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        return {"mape": mape, "rmse": rmse, "stability_score": 1.0}
        
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED) if n_samples >= 50 else LeaveOneOut()
    mapes, rmses = [], []
    
    for train_idx, test_idx in cv.split(X):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        model = LinearRegression()
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_test_cv)
        
        # Avoid zero division in MAPE
        if not np.any(y_test_cv == 0):
            mapes.append(mean_absolute_percentage_error(y_test_cv, preds))
        else:
            mapes.append(0.0)
        rmses.append(np.sqrt(mean_squared_error(y_test_cv, preds)))
        
    mape_mean = float(np.mean(mapes))
    rmse_mean = float(np.mean(rmses))
    stability = float(1.0 - np.std(mapes)) if len(mapes) > 1 else 1.0
    
    return {
        "mape": mape_mean,
        "rmse": rmse_mean,
        "stability_score": stability
    }

def run_training_pipeline(run_id: str, alert_manager) -> dict:
    """
    Step 2: Model Training Pipeline. Fits and evaluates all 5 ML models,
    executes Champion-Challenger testing, and updates the registry.
    """
    # 1. Load latest features
    df = read_feature_store(run_id)
    
    # 2. Extract anomaly targets via rules
    anomaly_target = get_rule_anomaly_target(df)
    estimated_anomaly_rate = float(anomaly_target.sum() / len(df)) if len(df) > 0 else 0.0
    
    # 3. Features selection
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_use = [col for col in num_cols if col != "net_salary"]
    
    # 4. Check cold start flag
    cold_start = is_cold_start("anomaly_detector")
    
    # Define models dict to log report
    training_report = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "cold_start": cold_start,
        "models": {}
    }
    
    # ==========================================
    # Model 1: Isolation Forest (Anomaly Detector)
    # ==========================================
    contamination = min(0.5, max(0.05, estimated_anomaly_rate))
    if_kwargs = {"n_estimators": 100, "contamination": contamination, "random_state": RANDOM_SEED}
    
    # CV evaluation
    if_metrics = evaluate_classifier_cv(IsolationForest, if_kwargs, df[features_to_use], anomaly_target)
    
    # Champion-Challenger or Direct Deploy
    promote_if = True
    if not cold_start:
        try:
            prod_if, prod_meta = load_production_model("anomaly_detector")
            old_f1 = prod_meta["metrics"]["f1"]
            if if_metrics["f1"] < old_f1 * 1.05:
                promote_if = False
                alert_manager.emit("RETRAIN_FAILED", "WARNING", "anomaly_detector", 
                                   trigger_value=if_metrics["f1"], threshold_value=old_f1 * 1.05,
                                   recommended_action="Retrained anomaly detector did not improve F1 score by 5%. Retained champion.")
        except FileNotFoundError:
            pass
            
    if promote_if:
        final_if = IsolationForest(**if_kwargs)
        final_if.fit(df[features_to_use])
        register_model("anomaly_detector", run_id, final_if, 
                       {"run_id": run_id, "metrics": if_metrics, "hyperparameters": if_kwargs}, 
                       features_to_use, mark_production=True)
    else:
        # Register new model but do not mark as production
        final_if = IsolationForest(**if_kwargs)
        final_if.fit(df[features_to_use])
        register_model("anomaly_detector", run_id, final_if, 
                       {"run_id": run_id, "metrics": if_metrics, "hyperparameters": if_kwargs}, 
                       features_to_use, mark_production=False)
                       
    training_report["models"]["anomaly_detector"] = {
        "status": "PROMOTED" if promote_if else "RETAINED_CHAMPION",
        "metrics": if_metrics
    }
    
    # ==========================================
    # Model 2: Random Forest Classifier (Absenteeism)
    # ==========================================
    rf_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 1.0, "stability_score": 1.0}
    rf_features = []
    
    if "present_days" in df.columns and "total_working_days" in df.columns:
        absenteeism_target = (df["present_days"] < df["total_working_days"] * 0.9).astype(int)
        
        # Prepare RF features (encode categoricals, exclude target leakage)
        leakage_cols = ["present_days", "total_working_days", "lop_days", "attendance_ratio", 
                        "earned_base", "expected_gross_salary", "expected_net", "salary_diff", "net_salary"]
        rf_num_features = [col for col in num_cols if col not in leakage_cols]
        
        # Label encode department and designation
        df_encoded = df.copy()
        mappings = {}
        for col in ["department", "designation"]:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique())
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df_encoded[col] = df[col].map(mapping).fillna(-1).astype(int)
                mappings[col] = mapping
                
        rf_features = rf_num_features + [col for col in ["department", "designation"] if col in df.columns]
        
        # SMOTE check for absenteeism
        abs_rate = absenteeism_target.mean()
        rf_kwargs = {"n_estimators": 100, "class_weight": "balanced", "random_state": RANDOM_SEED}
        
        rf_metrics = evaluate_classifier_cv(RandomForestClassifier, rf_kwargs, df_encoded[rf_features], absenteeism_target)
        
        promote_rf = True
        if not is_cold_start("absenteeism_classifier"):
            try:
                prod_rf, prod_meta = load_production_model("absenteeism_classifier")
                old_f1 = prod_meta["metrics"]["f1"]
                if rf_metrics["f1"] < old_f1 * 1.05:
                    promote_rf = False
                    alert_manager.emit("RETRAIN_FAILED", "WARNING", "absenteeism_classifier",
                                       trigger_value=rf_metrics["f1"], threshold_value=old_f1 * 1.05,
                                       recommended_action="Retrained absenteeism classifier did not improve F1 score by 5%. Retained champion.")
            except FileNotFoundError:
                pass
                
        # Fit final model
        final_rf = RandomForestClassifier(**rf_kwargs)
        # Apply SMOTE to final model fit if needed
        minority = absenteeism_target.sum()
        normal = len(absenteeism_target) - minority
        if abs_rate < 0.15 and minority >= 2 and normal >= 2:
            from imblearn.over_sampling import SMOTE
            k = min(5, minority - 1)
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
            X_res, y_res = smote.fit_resample(df_encoded[rf_features], absenteeism_target)
            final_rf.fit(X_res, y_res)
        else:
            final_rf.fit(df_encoded[rf_features], absenteeism_target)
            
        register_model("absenteeism_classifier", run_id, final_rf,
                       {"run_id": run_id, "metrics": rf_metrics, "hyperparameters": rf_kwargs, "mappings": mappings},
                       rf_features, mark_production=promote_rf)
                       
        training_report["models"]["absenteeism_classifier"] = {
            "status": "PROMOTED" if promote_rf else "RETAINED_CHAMPION",
            "metrics": rf_metrics
        }
        
    # ==========================================
    # Model 3: Salary Manipulation IF
    # ==========================================
    manip_features = [col for col in ["salary_diff", "salary_dev_pct"] if col in df.columns]
    # Include robust z-scores if available
    for c in ["robust_z_salary_diff", "robust_z_salary_dev_pct"]:
        if c in df.columns:
            manip_features.append(c)
            
    manip_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 1.0, "stability_score": 1.0}
    
    if manip_features:
        manip_kwargs = {"contamination": 0.1, "random_state": RANDOM_SEED}
        manip_metrics = evaluate_classifier_cv(IsolationForest, manip_kwargs, df[manip_features], anomaly_target)
        
        promote_manip = True
        if not is_cold_start("salary_manipulation_detector"):
            try:
                prod_m, prod_meta = load_production_model("salary_manipulation_detector")
                old_f1 = prod_meta["metrics"]["f1"]
                if manip_metrics["f1"] < old_f1 * 1.05:
                    promote_manip = False
                    alert_manager.emit("RETRAIN_FAILED", "WARNING", "salary_manipulation_detector",
                                       trigger_value=manip_metrics["f1"], threshold_value=old_f1 * 1.05,
                                       recommended_action="Retrained salary manipulation detector did not improve F1 score by 5%. Retained champion.")
            except FileNotFoundError:
                pass
                
        final_manip = IsolationForest(**manip_kwargs)
        final_manip.fit(df[manip_features])
        register_model("salary_manipulation_detector", run_id, final_manip,
                       {"run_id": run_id, "metrics": manip_metrics, "hyperparameters": manip_kwargs},
                       manip_features, mark_production=promote_manip)
                       
        training_report["models"]["salary_manipulation_detector"] = {
            "status": "PROMOTED" if promote_manip else "RETAINED_CHAMPION",
            "metrics": manip_metrics
        }
        
    # ==========================================
    # Model 4 & 5: Forecasting (Company & Department)
    # ==========================================
    comp_payroll, comp_overtime, dept_payroll = get_historical_data()
    n_history = len(comp_payroll)
    
    # Company Forecaster training
    if n_history > 0:
        X_hist = np.array(range(1, n_history + 1)).reshape(-1, 1)
        y_payroll = np.array(comp_payroll)
        y_overtime = np.array(comp_overtime)
        
        # Prepare CV data. If history is < 3, fit constant synthetically to avoid linear regression fit failure
        if n_history < 2:
            # Cold start fallback: fit on X = [[1], [2]], y = [val, val]
            X_fit = np.array([[1], [2]])
            y_pay_fit = np.array([comp_payroll[0], comp_payroll[0]])
            y_ot_fit = np.array([comp_overtime[0], comp_overtime[0]])
            
            payroll_metrics = {"mape": 0.0, "rmse": 0.0, "stability_score": 1.0}
            overtime_metrics = {"mape": 0.0, "rmse": 0.0, "stability_score": 1.0}
        else:
            X_fit = X_hist
            y_pay_fit = y_payroll
            y_ot_fit = y_overtime
            payroll_metrics = evaluate_regressor_cv(X_hist, y_payroll)
            overtime_metrics = evaluate_regressor_cv(X_hist, y_overtime)
            
        # Fit final company regressors
        lr_pay = LinearRegression()
        lr_pay.fit(X_fit, y_pay_fit)
        
        lr_ot = LinearRegression()
        lr_ot.fit(X_fit, y_ot_fit)
        
        # Champion-Challenger for company payroll forecaster
        promote_pay = True
        if not is_cold_start("company_payroll_forecaster"):
            try:
                _, prod_meta = load_production_model("company_payroll_forecaster")
                old_mape = prod_meta["metrics"]["mape"]
                # Lower MAPE is better
                if payroll_metrics["mape"] > old_mape * 0.95 and old_mape > 0:
                    promote_pay = False
                    alert_manager.emit("RETRAIN_FAILED", "WARNING", "company_payroll_forecaster",
                                       trigger_value=payroll_metrics["mape"], threshold_value=old_mape * 0.95,
                                       recommended_action="Retrained payroll forecaster did not improve MAPE by 5%. Retained champion.")
            except FileNotFoundError:
                pass
                
        register_model("company_payroll_forecaster", run_id, lr_pay,
                       {"run_id": run_id, "metrics": payroll_metrics, "history_length": n_history},
                       ["time_index"], mark_production=promote_pay)
                       
        register_model("company_overtime_forecaster", run_id, lr_ot,
                       {"run_id": run_id, "metrics": overtime_metrics, "history_length": n_history},
                       ["time_index"], mark_production=True) # Always promote overtime
                       
        training_report["models"]["company_payroll_forecaster"] = {
            "status": "PROMOTED" if promote_pay else "RETAINED_CHAMPION",
            "metrics": payroll_metrics
        }
        training_report["models"]["company_overtime_forecaster"] = {
            "status": "PROMOTED",
            "metrics": overtime_metrics
        }
        
    # Per-department forecasting
    dept_models = {}
    dept_metrics_report = {}
    for dept, history in dept_payroll.items():
        n_dept_hist = len(history)
        if n_dept_hist >= 3:
            X_dept = np.array(range(1, n_dept_hist + 1)).reshape(-1, 1)
            y_dept = np.array(history)
            
            lr_dept = LinearRegression()
            lr_dept.fit(X_dept, y_dept)
            dept_models[dept] = lr_dept
            
            dept_metrics_report[dept] = evaluate_regressor_cv(X_dept, y_dept)
            
    # Save all department models dict in a single registry version
    register_model("dept_payroll_forecaster", run_id, dept_models,
                   {"run_id": run_id, "metrics": dept_metrics_report, "num_departments": len(dept_models)},
                   ["time_index"], mark_production=True)
                   
    training_report["models"]["dept_payroll_forecaster"] = {
        "status": "PROMOTED",
        "metrics": {"num_dept_models_trained": len(dept_models)}
    }
    
    # Save training report as JSON
    report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_training_report.json"
    with open(report_path, "w") as f:
        json.dump(training_report, f, indent=2)
        
    return training_report
