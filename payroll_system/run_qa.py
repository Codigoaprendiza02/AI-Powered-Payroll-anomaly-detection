import os
import json
import pandas as pd
from pathlib import Path
from pipelines.p1_feature import run_feature_pipeline, read_feature_store

def run_qa_tests():
    print("# Phase 1 Human QA Testing Results\n")
    
    # ----------------------------------------------------
    # QA-1.1: Standard CSV Ingestion
    # ----------------------------------------------------
    print("## QA-1.1 — Standard CSV Ingestion")
    try:
        run_id = "test-run-001"
        # 1. Run pipeline
        run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id)
        
        # 2. Check feature store output
        df = read_feature_store(run_id)
        rows_ok = len(df) == 5
        cols = df.columns.tolist()
        required_cols = ["base_salary", "attendance_ratio", "salary_dev_pct", "earned_base", "robust_z_base_salary"]
        cols_ok = all(c in cols for c in required_cols)
        
        # 3. Check quality report
        report_path = Path("store/audit_log") / f"{run_id}_quality_report.json"
        with open(report_path, "r") as f:
            qc = json.load(f)
        dup_ok = qc.get("duplicate_rows") == 0
        flagged_ok = not any(c_info.get("flagged") for c_info in qc.get("columns", {}).values())
        
        status = "PASS" if (rows_ok and cols_ok and dup_ok and flagged_ok) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Rows**: {len(df)} (Expected: 5)")
        print(f"- **Engineered Columns Verified**: {cols_ok}")
        print(f"- **Duplicate Rows**: {qc.get('duplicate_rows')} (Expected: 0)")
        print(f"- **Columns Flagged for Nulls**: {not flagged_ok}")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-1.2: Non-Standard Column Names (Synonyms)
    # ----------------------------------------------------
    print("## QA-1.2 — Non-Standard Column Names (Synonyms)")
    try:
        run_id = "test-run-002"
        # 1. Run pipeline
        run_feature_pipeline("tests/qa_data/phase1/test_synonyms.csv", run_id)
        
        # 2. Confirm base_salary mapping values
        df = read_feature_store(run_id)
        salaries = df["base_salary"].tolist()
        salaries_expected = [90000, 70000, 25000, 95000, 72000]
        salaries_ok = salaries == salaries_expected
        
        # 3. Check schema confidence
        report_path = Path("store/audit_log") / f"{run_id}_quality_report.json"
        with open(report_path, "r") as f:
            qc = json.load(f)
        confidence_ok = qc.get("schema_confidence", 0.0) >= 0.7
        warnings_ok = "SCHEMA_MISMATCH_WARNING" not in qc.get("warnings", [])
        
        status = "PASS" if (salaries_ok and confidence_ok and warnings_ok) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Mapped Salaries**: {salaries} (Expected: {salaries_expected})")
        print(f"- **Schema Confidence**: {qc.get('schema_confidence'):.2f} (Expected: >= 0.7)")
        print(f"- **Warnings**: {qc.get('warnings')}")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-1.3: Low Schema Confidence Warning
    # ----------------------------------------------------
    print("## QA-1.3 — Low Schema Confidence Warning")
    try:
        run_id = "test-run-003"
        # 1. Run pipeline
        run_feature_pipeline("tests/qa_data/phase1/test_garbled.csv", run_id)
        
        # 2. Check Parquet creation
        parquet_exists = (Path("store/features") / f"{run_id}_features.parquet").exists()
        
        # 3. Check warnings and confidence
        report_path = Path("store/audit_log") / f"{run_id}_quality_report.json"
        with open(report_path, "r") as f:
            qc = json.load(f)
        confidence_low = qc.get("schema_confidence", 0.0) < 0.7
        warning_present = "SCHEMA_MISMATCH_WARNING" in qc.get("warnings", [])
        
        status = "PASS" if (parquet_exists and confidence_low and warning_present) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Parquet File Created**: {parquet_exists}")
        print(f"- **Schema Confidence**: {qc.get('schema_confidence'):.2f} (Expected: < 0.7)")
        print(f"- **Warnings**: {qc.get('warnings')} (Expected: containing SCHEMA_MISMATCH_WARNING)")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-1.4: Missing Optional Columns
    # ----------------------------------------------------
    print("## QA-1.4 — Missing Optional Columns")
    try:
        run_id = "test-run-004"
        # 1. Run pipeline
        run_feature_pipeline("tests/qa_data/phase1/test_minimal.csv", run_id)
        
        # 2. Check attendance_ratio computation
        df = read_feature_store(run_id)
        ratios = [round(r, 3) for r in df["attendance_ratio"].tolist()]
        ratios_expected = [round(22/23, 3), round(23/23, 3), round(20/23, 3), round(21/23, 3), round(23/23, 3)]
        ratios_ok = ratios == ratios_expected
        
        # 3. Check skip log
        report_path = Path("store/audit_log") / f"{run_id}_quality_report.json"
        with open(report_path, "r") as f:
            qc = json.load(f)
        skipped = qc.get("skipped_features", [])
        skipped_ok = "overtime_pay" in skipped
        
        status = "PASS" if (ratios_ok and skipped_ok) else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Computed Attendance Ratios**: {ratios} (Expected: {ratios_expected})")
        print(f"- **Skipped Features Log**: {skipped} (Expected: containing 'overtime_pay')")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")
        
    print("\n" + "="*40 + "\n")
    
    # ----------------------------------------------------
    # QA-1.5: Duplicate Run Idempotency
    # ----------------------------------------------------
    print("## QA-1.5 — Duplicate Run Idempotency")
    try:
        run_id = "test-run-005"
        # 1. Run pipeline twice
        run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id)
        run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id)
        
        # 2. Count output files
        feat_store = Path("store/features")
        matching_files = [f.name for f in feat_store.glob(f"*{run_id}*")]
        count_ok = len(matching_files) == 1
        
        status = "PASS" if count_ok else "FAIL"
        print(f"- **Status**: {status}")
        print(f"- **Feature Store Files Found**: {matching_files} (Expected exactly 1 file: ['{run_id}_features.parquet'])")
    except Exception as e:
        print(f"- **Status**: FAIL (Error: {e})")

if __name__ == "__main__":
    run_qa_tests()
