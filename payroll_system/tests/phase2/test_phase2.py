import pytest
import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import AUDIT_LOG_DIR, FEATURES_STORE_DIR, MODELS_REGISTRY_DIR
from models.registry import is_cold_start, register_model, load_production_model, list_model_versions
from pipelines.pipeline_runner import run_full_pipeline, AlertManagerStub
from pipelines.p1_feature import run_feature_pipeline, read_feature_store
from pipelines.p2_training import run_training_pipeline
from pipelines.p3_inference import run_inference_pipeline

@pytest.fixture(autouse=True)
def cleanup_registry_and_stores():
    # Setup clean directories before each test
    for d in [MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    yield
    # Cleanup after test runs
    for d in [MODELS_REGISTRY_DIR, FEATURES_STORE_DIR, AUDIT_LOG_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def test_model_registry():
    # Verify cold start
    assert is_cold_start("dummy_model") is True
    
    # Register model
    dummy_model_obj = "DUMMY_MODEL_OBJECT"
    register_model(
        model_name="dummy_model",
        version_id="run_1",
        model_obj=dummy_model_obj,
        metadata={"metrics": {"f1": 0.8}, "run_id": "run_1"},
        features=["feat1", "feat2"],
        mark_production=True
    )
    
    assert is_cold_start("dummy_model") is False
    
    # Load production model
    model_obj, meta = load_production_model("dummy_model")
    assert model_obj == dummy_model_obj
    assert meta["metrics"]["f1"] == 0.8
    assert meta["is_production"] is True
    assert meta["version_id"] == "run_1"
    
    # Register a second version (not production)
    register_model(
        model_name="dummy_model",
        version_id="run_2",
        model_obj="DUMMY_MODEL_2",
        metadata={"metrics": {"f1": 0.95}, "run_id": "run_2"},
        features=["feat1", "feat2"],
        mark_production=False
    )
    
    # Production model should still be run_1
    model_obj, meta = load_production_model("dummy_model")
    assert model_obj == dummy_model_obj
    assert meta["version_id"] == "run_1"
    
    # Register a third version (marking production)
    register_model(
        model_name="dummy_model",
        version_id="run_3",
        model_obj="DUMMY_MODEL_3",
        metadata={"metrics": {"f1": 0.97}, "run_id": "run_3"},
        features=["feat1", "feat2"],
        mark_production=True
    )
    
    # Production model should now be run_3
    model_obj, meta = load_production_model("dummy_model")
    assert model_obj == "DUMMY_MODEL_3"
    assert meta["version_id"] == "run_3"
    
    # Check that run_1 metadata is updated to is_production = False
    run_1_meta_file = Path(MODELS_REGISTRY_DIR) / "dummy_model" / "run_1" / "metadata.json"
    with open(run_1_meta_file, "r") as f:
        run_1_meta = json.load(f)
    assert run_1_meta["is_production"] is False
    
    # Check listing versions
    versions = list_model_versions("dummy_model")
    assert len(versions) == 3
    assert versions[0]["version_id"] == "run_3"

def test_training_pipeline_and_cold_start():
    run_id = "test_run_cold_start"
    alert_manager = AlertManagerStub(run_id)
    
    # 1. Run Feature Pipeline to generate parquet features
    run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id, alert_manager)
    
    # 2. Assert cold start is True before training
    assert is_cold_start("anomaly_detector") is True
    
    # 3. Run training
    report = run_training_pipeline(run_id, alert_manager)
    
    # 4. Assert cold start is False after training
    assert is_cold_start("anomaly_detector") is False
    assert is_cold_start("absenteeism_classifier") is False
    assert is_cold_start("salary_manipulation_detector") is False
    assert is_cold_start("company_payroll_forecaster") is False
    
    # 5. Verify production models registered successfully
    obj, meta = load_production_model("anomaly_detector")
    assert meta["run_id"] == run_id
    assert meta["is_production"] is True
    
    assert report["cold_start"] is True
    assert "anomaly_detector" in report["models"]
    assert report["models"]["anomaly_detector"]["status"] == "PROMOTED"

def test_champion_challenger():
    alert_manager = AlertManagerStub("cc_test")
    
    # We will simulate Champion-Challenger check
    # Let's train a model and mark as production (F1 = 0.5)
    register_model(
        model_name="anomaly_detector",
        version_id="run_champ",
        model_obj="CHAMP",
        metadata={"metrics": {"f1": 0.5}, "run_id": "run_champ"},
        features=["feat1"],
        mark_production=True
    )
    
    # Run Feature Pipeline for new run
    run_id = "run_challenger"
    run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id, alert_manager)
    
    # Run Training Pipeline
    # Our generated rules on test_standard might produce a certain F1.
    # To check the logic: if F1 score improves by 5%, it promoted. Else, it retains champion.
    # Let's see: we run the training pipeline
    report = run_training_pipeline(run_id, alert_manager)
    
    # Load production model to see who is production
    model_obj, meta = load_production_model("anomaly_detector")
    
    # If the new F1 was worse than old_f1 * 1.05 (0.5 * 1.05 = 0.525), it should have retained champ.
    # If it was better, it should have promoted challenger.
    # Let's verify that the decision was logged and stored correctly in report
    status = report["models"]["anomaly_detector"]["status"]
    assert status in ["PROMOTED", "RETAINED_CHAMPION"]
    
    if status == "RETAINED_CHAMPION":
        assert meta["version_id"] == "run_champ"
        # Check that warning was emitted in alert manager
        warnings = [a for a in alert_manager.alerts if a["alert_type"] == "RETRAIN_FAILED"]
        assert len(warnings) > 0
    else:
        assert meta["version_id"] == run_id

def test_inference_pipeline():
    run_id = "test_run_inference"
    alert_manager = AlertManagerStub(run_id)
    
    # Run Feature & Training to have a production model
    run_feature_pipeline("tests/qa_data/phase1/test_standard.csv", run_id, alert_manager)
    run_training_pipeline(run_id, alert_manager)
    
    # Run Inference
    results_df = run_inference_pipeline(run_id, alert_manager)
    
    # Verify outputs
    assert len(results_df) == 5
    assert "final_anomaly_score" in results_df.columns
    assert "risk_tier" in results_df.columns
    assert "explanation" in results_df.columns
    
    # Verify risk tiers are correctly assigned
    assert results_df["risk_tier"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()
    
    # Check explanations for HIGH or CRITICAL rows
    high_critical_rows = results_df[results_df["risk_tier"].isin(["HIGH", "CRITICAL"])]
    for _, row in high_critical_rows.iterrows():
        assert len(row["explanation"]) > 0
        assert "Risk Tier" in row["explanation"]

def test_pipeline_idempotency():
    # Calling run_full_pipeline twice with the same file returns the same run_id
    file_path = "tests/qa_data/phase1/test_standard.csv"
    
    rid1 = run_full_pipeline(file_path)
    rid2 = run_full_pipeline(file_path)
    
    assert rid1 == rid2

def test_pipeline_failure():
    from unittest.mock import patch
    file_path = "tests/qa_data/phase1/test_standard.csv"
    
    with patch("pipelines.pipeline_runner.run_training_pipeline", side_effect=ValueError("Simulated training failure")):
        with pytest.raises(ValueError, match="Simulated training failure"):
            run_full_pipeline(file_path)
            
    # Check runs.json for FAILED status
    runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
    assert runs_file.exists()
    with open(runs_file, "r") as f:
        runs = json.load(f)
        
    failed_runs = [r for r in runs if r["status"] == "FAILED"]
    assert len(failed_runs) > 0
    last_failed = failed_runs[-1]
    running_or_failed_steps = [s for s in last_failed["pipeline_steps"] if s["status"] == "FAILED"]
    assert len(running_or_failed_steps) > 0
