import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from config.settings import FEATURES_STORE_DIR, AUDIT_LOG_DIR, DRIFT_STORE_DIR

# Custom Exceptions
class DataCleaningError(Exception):
    pass

class InsufficientDataError(Exception):
    pass

# Step 4 — Data Ingestion
def load_raw_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    # Strip whitespace and lowercase columns
    df.columns = df.columns.str.lower().str.strip()
    return df

# Step 3 — Column Mapping Registry
def detect_schema(df: pd.DataFrame) -> dict:
    # Resolve relative path for column_mapping.json from this file
    mapping_path = Path(__file__).parent.parent / "config" / "column_mapping.json"
    with open(mapping_path, "r") as f:
        column_mapping = json.load(f)
        
    mapped = {}
    unmapped = []
    
    # Iterate through canonical fields and map to df columns using synonyms
    for canonical, synonyms in column_mapping.items():
        found = False
        for synonym in synonyms:
            syn_clean = synonym.lower().strip()
            if syn_clean in df.columns:
                mapped[canonical] = syn_clean
                found = True
                break
        if not found:
            unmapped.append(canonical)
            
    confidence = len(mapped) / len(column_mapping)
    
    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "confidence": confidence
    }

# Step 5 — Data Quality Module
def run_quality_checks(df: pd.DataFrame, run_id: str) -> dict:
    schema_info = detect_schema(df)
    mapped_cols = schema_info["mapped"]
    
    total_rows = len(df)
    # Consider row a duplicate if ALL columns are identical
    duplicate_rows = int(df.duplicated().sum())
    
    # Find impossible value rows: rows with any negative value in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    impossible_value_rows = int((numeric_df < 0).any(axis=1).sum())
    
    columns_report = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = float((null_count / total_rows) * 100) if total_rows > 0 else 0.0
        dtype = str(df[col].dtype)
        # Flag if null percentage > 20%
        flagged = null_pct > 20.0
        
        columns_report[col] = {
            "null_count": null_count,
            "null_pct": null_pct,
            "dtype": dtype,
            "flagged": flagged
        }
        
    skipped_features = []
    if "overtime_hours" not in mapped_cols or "overtime_pay_per_hour" not in mapped_cols:
        skipped_features.append("overtime_pay")

    report = {
        "run_id": run_id,
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "columns": columns_report,
        "impossible_value_rows": impossible_value_rows,
        "schema_confidence": schema_info["confidence"],
        "mapped_columns": mapped_cols,
        "unmapped_columns": schema_info["unmapped"],
        "skipped_features": skipped_features,
        "warnings": []
    }
    
    if schema_info["confidence"] < 0.7:
        report["warnings"].append("SCHEMA_MISMATCH_WARNING")
        
    # Write quality report
    report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

# Step 6.1 — Duplicate Row Removal
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    # Keep first occurrence of identical duplicate rows
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed

# Step 6.2 — Null Value Handling
def handle_nulls(df: pd.DataFrame, mapped_cols: dict) -> tuple[pd.DataFrame, dict]:
    log = {}
    null_before = df.isnull().sum().to_dict()
    
    # Drop rows where identity or core salary columns are null
    # We resolve actual column names using the mapped columns from registry
    drop_actuals = []
    for canonical in ["employee_id", "base_salary", "net_salary"]:
        if canonical in mapped_cols:
            drop_actuals.append(mapped_cols[canonical])
            
    rows_before = len(df)
    if drop_actuals:
        df = df.dropna(subset=drop_actuals)
    log["rows_dropped_null_critical"] = rows_before - len(df)
    
    # Categorical fills: department, designation
    for canonical in ["department", "designation"]:
        if canonical in mapped_cols:
            actual = mapped_cols[canonical]
            n = df[actual].isnull().sum()
            if n > 0:
                df[actual] = df[actual].fillna("Unknown")
                log[f"{canonical}_filled_unknown"] = int(n)
                
    # Numeric fills
    if "present_days" in mapped_cols:
        actual = mapped_cols["present_days"]
        n = df[actual].isnull().sum()
        if n > 0:
            median_val = int(df[actual].median()) if not df[actual].isnull().all() else 23
            df[actual] = df[actual].fillna(median_val)
            log["present_days_filled_median"] = int(n)
            
    if "total_working_days" in mapped_cols:
        actual = mapped_cols["total_working_days"]
        n = df[actual].isnull().sum()
        if n > 0:
            mode_series = df[actual].mode()
            mode_val = int(mode_series[0]) if not mode_series.empty else 23
            df[actual] = df[actual].fillna(mode_val)
            log["total_working_days_filled_mode"] = int(n)
            
    for canonical in ["overtime_hours", "overtime_pay_per_hour"]:
        if canonical in mapped_cols:
            actual = mapped_cols[canonical]
            n = df[actual].isnull().sum()
            if n > 0:
                df[actual] = df[actual].fillna(0)
                log[f"{canonical}_filled_zero"] = int(n)
                
    # Recompute lop_days from present and total
    if "lop_days" in mapped_cols and "total_working_days" in mapped_cols and "present_days" in mapped_cols:
        df[mapped_cols["lop_days"]] = df[mapped_cols["total_working_days"]] - df[mapped_cols["present_days"]]
        log["lop_days_recomputed"] = True
        
    log["null_before"] = {k: int(v) for k, v in null_before.items() if v > 0}
    log["null_after"] = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    
    return df, log

# Step 6.3 — Impossible Value Correction
def fix_impossible_values(df: pd.DataFrame, mapped_cols: dict) -> tuple[pd.DataFrame, dict]:
    log = {}
    
    # present_days cannot exceed total_working_days
    if "present_days" in mapped_cols and "total_working_days" in mapped_cols:
        pres = mapped_cols["present_days"]
        tot = mapped_cols["total_working_days"]
        mask = df[pres] > df[tot]
        if mask.any():
            df.loc[mask, pres] = df.loc[mask, tot]
            log["present_days_capped_to_total"] = int(mask.sum())
            
    # present_days, lop_days and overtime_hours cannot be negative
    for canonical in ["present_days", "lop_days", "overtime_hours"]:
        if canonical in mapped_cols:
            actual = mapped_cols[canonical]
            mask = df[actual] < 0
            if mask.any():
                df.loc[mask, actual] = 0
                log[f"{canonical}_negative_zeroed"] = int(mask.sum())
                
    # base_salary and net_salary must be positive (>= 1)
    for canonical in ["base_salary", "net_salary"]:
        if canonical in mapped_cols:
            actual = mapped_cols[canonical]
            mask = df[actual] <= 0
            if mask.any():
                df = df[~mask]
                log[f"{canonical}_nonpositive_rows_dropped"] = int(mask.sum())
                
    # overtime_pay_per_hour cannot be negative
    if "overtime_pay_per_hour" in mapped_cols:
        actual = mapped_cols["overtime_pay_per_hour"]
        mask = df[actual] < 0
        if mask.any():
            df.loc[mask, actual] = 0
            log["overtime_pay_per_hour_negative_zeroed"] = int(mask.sum())
            
    # lop_days must equal total_working_days - present_days - recompute after all fixes
    if "lop_days" in mapped_cols and "total_working_days" in mapped_cols and "present_days" in mapped_cols:
        df[mapped_cols["lop_days"]] = df[mapped_cols["total_working_days"]] - df[mapped_cols["present_days"]]
        
    return df, log

# Step 6.4 — Type Casting
REQUIRED_DTYPES = {
    "base_salary": "int64",
    "present_days": "int64",
    "total_working_days": "int64",
    "overtime_hours": "int64",
    "overtime_pay_per_hour": "int64",
    "net_salary": "int64",
    "lop_days": "int64",
}

def enforce_dtypes(df: pd.DataFrame, mapped_cols: dict) -> pd.DataFrame:
    for canonical, dtype in REQUIRED_DTYPES.items():
        if canonical in mapped_cols:
            actual = mapped_cols[canonical]
            df[actual] = pd.to_numeric(df[actual], errors="coerce").fillna(0).astype(dtype)
    return df

# Step 6.5 — Minimum Viable Batch Check
def check_minimum_rows(df: pd.DataFrame, min_rows: int = 5) -> None:
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Only {len(df)} rows remain after cleaning. "
            f"Minimum required: {min_rows}."
        )

# Step 6 — Data Cleaning Wrapper
def clean_data(df: pd.DataFrame, quality_report: dict, run_id: str, min_rows: int = 5) -> tuple[pd.DataFrame, dict]:
    original_rows = len(df)
    summary = {"run_id": run_id, "original_rows": original_rows, "steps": {}}
    mapped_cols = quality_report["mapped_columns"]
    
    df, dup_log = remove_duplicates(df)
    summary["steps"]["duplicates"] = dup_log
    if dup_log > 0 and (dup_log / original_rows) > 0.05:
        # Append DUPLICATE_DATA_WARNING to quality report warnings
        quality_report["warnings"].append("DUPLICATE_DATA_WARNING")
        
    df, null_log = handle_nulls(df, mapped_cols)
    summary["steps"]["nulls"] = null_log
    
    df, impos_log = fix_impossible_values(df, mapped_cols)
    summary["steps"]["impossible_values"] = impos_log
    
    df = enforce_dtypes(df, mapped_cols)
    summary["steps"]["dtypes_enforced"] = [mapped_cols[k] for k in REQUIRED_DTYPES.keys() if k in mapped_cols]
    
    # Assert that no nulls remain in mapped columns
    for canonical in mapped_cols.values():
        if df[canonical].isnull().any():
            raise DataCleaningError(f"Nulls remain in mapped column: {canonical}")
            
    check_minimum_rows(df, min_rows)
    
    summary["rows_after_cleaning"] = len(df)
    summary["rows_removed_total"] = original_rows - len(df)
    summary["removal_pct"] = round((original_rows - len(df)) / original_rows * 100, 2) if original_rows > 0 else 0.0
    
    if summary["removal_pct"] > 10.0:
        summary["warning"] = (
            f"{summary['removal_pct']}% of rows were removed during cleaning. "
            f"Inspect the quality report for root cause."
        )
        
    # Write cleaning report
    cleaning_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_cleaning_report.json"
    with open(cleaning_report_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Also update the quality report on disk with any new warnings
    quality_report_path = Path(AUDIT_LOG_DIR) / f"{run_id}_quality_report.json"
    with open(quality_report_path, "w") as f:
        json.dump(quality_report, f, indent=2)
        
    return df, summary

# Step 7 — Feature Engineering
def engineer_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    mapped = col_map["mapped"]
    
    # Helper to check if a list of canonical names are all present in mapped
    def columns_available(*names):
        return all(name in mapped for name in names)
        
    # Core variables for formula computation
    emp_id_col = mapped.get("employee_id")
    dept_col = mapped.get("department")
    des_col = mapped.get("designation")
    base_sal_col = mapped.get("base_salary")
    pres_days_col = mapped.get("present_days")
    tot_days_col = mapped.get("total_working_days")
    ot_hours_col = mapped.get("overtime_hours")
    ot_rate_col = mapped.get("overtime_pay_per_hour")
    net_sal_col = mapped.get("net_salary")
    
    # We will output a new DataFrame with canonical column names
    out_df = pd.DataFrame()
    
    # Copy across identity and raw features (with canonical naming)
    if emp_id_col: out_df["employee_id"] = df[emp_id_col]
    if dept_col: out_df["department"] = df[dept_col]
    if des_col: out_df["designation"] = df[des_col]
    if base_sal_col: out_df["base_salary"] = df[base_sal_col]
    if pres_days_col: out_df["present_days"] = df[pres_days_col]
    if tot_days_col: out_df["total_working_days"] = df[tot_days_col]
    if ot_hours_col: out_df["overtime_hours"] = df[ot_hours_col]
    if ot_rate_col: out_df["overtime_pay_per_hour"] = df[ot_rate_col]
    if net_sal_col: out_df["net_salary"] = df[net_sal_col]
    if "lop_days" in mapped: out_df["lop_days"] = df[mapped["lop_days"]]
    
    # Compute: attendance_ratio
    if columns_available("present_days", "total_working_days"):
        out_df["attendance_ratio"] = df[pres_days_col] / df[tot_days_col]
        
    # Compute: overtime_pay
    if columns_available("overtime_hours", "overtime_pay_per_hour"):
        out_df["overtime_pay"] = df[ot_hours_col] * df[ot_rate_col]
    elif columns_available("overtime_hours") or columns_available("overtime_pay_per_hour"):
        # Handle partially missing overtime features
        pass
        
    # Compute: earned_base
    if columns_available("base_salary", "total_working_days", "present_days"):
        out_df["earned_base"] = (df[base_sal_col] / df[tot_days_col]) * df[pres_days_col]
        
    # Compute: expected_gross_salary
    if "earned_base" in out_df.columns and "overtime_pay" in out_df.columns:
        out_df["expected_gross_salary"] = out_df["earned_base"] + out_df["overtime_pay"]
    elif "earned_base" in out_df.columns:
        out_df["expected_gross_salary"] = out_df["earned_base"]
        
    # Compute: expected_net (with support for tax/pf deductions if they exist as raw columns)
    tax_col = None
    pf_col = None
    # Check if there are any other columns in the source df containing 'tax' or 'pf'
    for c in df.columns:
        if "tax" in c and "deduct" in c: tax_col = c
        if "pf" in c and "deduct" in c: pf_col = c
        
    tax_series = df[tax_col] if tax_col else 0
    pf_series = df[pf_col] if pf_col else 0
    
    if "expected_gross_salary" in out_df.columns:
        out_df["expected_net"] = out_df["expected_gross_salary"] - (tax_series + pf_series)
        
    # Compute: salary_diff
    if columns_available("net_salary") and "expected_net" in out_df.columns:
        out_df["salary_diff"] = df[net_sal_col] - out_df["expected_net"]
        
    # Compute: salary_dev_pct = (base_salary - designation_mean) / designation_mean * 100
    if columns_available("base_salary", "designation"):
        des_means = df.groupby(des_col)[base_sal_col].transform("mean")
        # Handle zero division: if mean is zero, set dev to 0
        out_df["salary_dev_pct"] = np.where(des_means != 0, (df[base_sal_col] - des_means) / des_means * 100, 0.0)
        
    # Compute Robust Z-Scores for all numerical features in the out_df
    numeric_cols = out_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = out_df[col]
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            out_df[f"robust_z_{col}"] = 0.0
        else:
            # Robust Z = (x - Median) / MAD
            out_df[f"robust_z_{col}"] = (series - median) / mad
            
    return out_df

# Step 8 — Feature Store
def write_feature_store(df: pd.DataFrame, run_id: str):
    out_path = Path(FEATURES_STORE_DIR) / f"{run_id}_features.parquet"
    df.to_parquet(out_path, index=False)

def read_feature_store(run_id: str) -> pd.DataFrame:
    in_path = Path(FEATURES_STORE_DIR) / f"{run_id}_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Feature file not found for run: {run_id}")
    return pd.read_parquet(in_path)

def save_reference_distributions(df: pd.DataFrame, run_id: str):
    distributions = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = df[col]
        distributions[col] = {
            "mean": float(series.mean()) if not pd.isna(series.mean()) else 0.0,
            "median": float(series.median()) if not pd.isna(series.median()) else 0.0,
            "std": float(series.std()) if not pd.isna(series.std()) else 0.0,
            "p25": float(series.quantile(0.25)) if not pd.isna(series.quantile(0.25)) else 0.0,
            "p75": float(series.quantile(0.75)) if not pd.isna(series.quantile(0.75)) else 0.0
        }
    
    value_counts = {
        "department": df["department"].value_counts().to_dict() if "department" in df.columns else {},
        "designation": df["designation"].value_counts().to_dict() if "designation" in df.columns else {}
    }
    
    ref_data = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "distributions": distributions,
        "value_counts": value_counts
    }
    
    ref_path = Path(DRIFT_STORE_DIR) / f"reference_{run_id}.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(ref_data, f, indent=2)

# Phase 1 Orchestrator
def run_feature_pipeline(file_path: str, run_id: str, alert_manager=None) -> pd.DataFrame:
    # 1. Ingestion
    df = load_raw_data(file_path)
    
    # 2. Schema detection
    schema_info = detect_schema(df)
    
    # 3. Quality checks
    quality_report = run_quality_checks(df, run_id)
    
    # 4. Cleaning
    df_cleaned, cleaning_summary = clean_data(df, quality_report, run_id)
    
    # 5. Feature Engineering
    df_features = engineer_features(df_cleaned, schema_info)
    
    # 6. Feature Store write
    write_feature_store(df_features, run_id)
    
    # 7. Save Reference Distributions (for drift monitoring)
    save_reference_distributions(df_features, run_id)
    
    return df_features
