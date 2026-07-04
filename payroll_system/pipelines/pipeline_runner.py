import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from config.settings import generate_run_id, AUDIT_LOG_DIR
from pipelines.p1_feature import run_feature_pipeline

class AlertManagerStub:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.alerts = []
        self._seen = set()
        
    def emit(self, alert_type: str, severity: str, affected_entity: str,
             trigger_value=None, threshold_value=None, recommended_action: str = ""):
        key = f"{alert_type}::{affected_entity}"
        if key in self._seen:
            return
        self._seen.add(key)
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
        alerts_path = Path(AUDIT_LOG_DIR) / f"{self.run_id}_alerts.json"
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2)

def compute_md5(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks of 8KB
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_run_id_by_hash(file_hash: str) -> str or None:
    hash_file = Path(AUDIT_LOG_DIR) / "file_hashes.json"
    if not hash_file.exists():
        return None
    try:
        with open(hash_file, "r") as f:
            hashes = json.load(f)
            return hashes.get(file_hash)
    except Exception:
        return None

def log_run_start(run_id: str, file_path: str, file_hash: str):
    runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
    runs = []
    if runs_file.exists():
        try:
            with open(runs_file, "r") as f:
                runs = json.load(f)
        except Exception:
            runs = []
            
    # Add new run entry
    new_run = {
        "run_id": run_id,
        "file_path": file_path,
        "file_hash": file_hash,
        "status": "RUNNING",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "pipeline_steps": [
            {"pipeline_step": "feature", "status": "RUNNING", "timestamp": datetime.utcnow().isoformat()}
        ]
    }
    runs.append(new_run)
    with open(runs_file, "w") as f:
        json.dump(runs, f, indent=2)
        
    # Update hash mapping
    hash_file = Path(AUDIT_LOG_DIR) / "file_hashes.json"
    hashes = {}
    if hash_file.exists():
        try:
            with open(hash_file, "r") as f:
                hashes = json.load(f)
        except Exception:
            hashes = {}
    hashes[file_hash] = run_id
    with open(hash_file, "w") as f:
        json.dump(hashes, f, indent=2)
        
    # Update last_run_id
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    with open(last_run_file, "w") as f:
        f.write(run_id)

def log_run_complete(run_id: str):
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
                # Update pipeline step status
                for step in run.get("pipeline_steps", []):
                    if step["pipeline_step"] == "feature":
                        step["status"] = "COMPLETE"
                        step["timestamp"] = datetime.utcnow().isoformat()
                break
                
        with open(runs_file, "w") as f:
            json.dump(runs, f, indent=2)
    except Exception:
        pass

def log_run_failed(run_id: str, error_msg: str):
    runs_file = Path(AUDIT_LOG_DIR) / "runs.json"
    if not runs_file.exists():
        return
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
            
        for run in runs:
            if run["run_id"] == run_id:
                run["status"] = "FAILED"
                run["error"] = error_msg
                for step in run.get("pipeline_steps", []):
                    if step["pipeline_step"] == "feature":
                        step["status"] = "FAILED"
                        step["timestamp"] = datetime.utcnow().isoformat()
                break
                
        with open(runs_file, "w") as f:
            json.dump(runs, f, indent=2)
    except Exception:
        pass

def run_full_pipeline(file_path: str, alert_manager=None) -> str:
    file_hash = compute_md5(file_path)
    existing_run_id = get_run_id_by_hash(file_hash)
    if existing_run_id:
        return existing_run_id
        
    run_id = generate_run_id()
    if alert_manager is None:
        alert_manager = AlertManagerStub(run_id)
        
    log_run_start(run_id, file_path, file_hash)
    
    try:
        run_feature_pipeline(file_path, run_id, alert_manager)
        alert_manager.save()
        log_run_complete(run_id)
    except Exception as e:
        alert_manager.emit("PIPELINE_FAILURE_ALERT", "CRITICAL", "system",
                           trigger_value=str(e), recommended_action="Investigate pipeline logs immediately.")
        alert_manager.save()
        log_run_failed(run_id, str(e))
        raise
        
    return run_id
