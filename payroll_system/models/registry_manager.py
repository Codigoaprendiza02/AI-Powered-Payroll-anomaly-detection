import os
import json
from datetime import datetime
from pathlib import Path
from config.settings import MODELS_REGISTRY_DIR

def rollback_model(model_name: str) -> dict:
    """
    Rolls back the active production model to the second-to-last production version.
    Updates the on-disk metadata.json files and appends the restored version to
    the production history stack.
    """
    history_path = Path(MODELS_REGISTRY_DIR) / "production_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Production history file not found at {history_path}")
        
    with open(history_path, "r") as f:
        history_data = json.load(f)
        
    model_history = history_data.get(model_name, [])
    if len(model_history) < 2:
        raise ValueError(f"Insufficient history for model '{model_name}' to perform rollback. Required >= 2 entries, found {len(model_history)}.")
        
    # Second-to-last production version
    restored_entry = model_history[-2]
    restored_version = restored_entry["model_version"]
    
    # Update on-disk metadata.json files
    parent_dir = Path(MODELS_REGISTRY_DIR) / model_name
    if not parent_dir.exists():
        raise FileNotFoundError(f"Model registry directory not found for '{model_name}'.")
        
    for v_dir in parent_dir.iterdir():
        if v_dir.is_dir():
            meta_file = v_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta_data = json.load(f)
                    
                    is_restored = (v_dir.name == restored_version)
                    meta_data["is_production"] = is_restored
                    
                    with open(meta_file, "w") as f:
                        json.dump(meta_data, f, indent=2)
                except Exception as e:
                    # Ignore or log error
                    pass
                    
    # Append the restored version as the new active production version in history
    new_entry = restored_entry.copy()
    new_entry["timestamp"] = datetime.utcnow().isoformat()
    new_entry["is_production"] = True
    
    model_history.append(new_entry)
    # Keep the last 5 entries
    history_data[model_name] = model_history[-5:]
    
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
        
    return new_entry
