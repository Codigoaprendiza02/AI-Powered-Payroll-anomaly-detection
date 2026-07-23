import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from config.settings import MODELS_REGISTRY_DIR

def get_model_registry_dir(model_name: str) -> Path:
    return Path(MODELS_REGISTRY_DIR) / model_name

def register_model(
    model_name: str,
    version_id: str,
    model_obj,
    metadata: dict,
    features: list,
    mark_production: bool = True
):
    """
    Registers a model version under models/registry/{model_name}/{version_id}/.
    Saves:
      - model.pkl
      - metadata.json
      - features.json
    If mark_production is True, makes this version the current production model,
    setting is_production to False for all other versions of this model.
    """
    model_dir = get_model_registry_dir(model_name) / version_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model artifact
    model_path = model_dir / "model.pkl"
    joblib.dump(model_obj, model_path)
    
    # Save features list
    features_path = model_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)
        
    # Standardize and save metadata
    metadata_copy = metadata.copy()
    metadata_copy.setdefault("timestamp", datetime.utcnow().isoformat())
    metadata_copy["is_production"] = mark_production
    
    # Update other versions' metadata if marking this one as production
    if mark_production:
        parent_dir = get_model_registry_dir(model_name)
        if parent_dir.exists():
            for v_dir in parent_dir.iterdir():
                if v_dir.is_dir() and v_dir.name != version_id:
                    meta_file = v_dir / "metadata.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r") as f:
                                meta_data = json.load(f)
                            if meta_data.get("is_production", False):
                                meta_data["is_production"] = False
                                with open(meta_file, "w") as f:
                                    json.dump(meta_data, f, indent=2)
                        except Exception:
                            pass
                            
        # Maintain production history
        history_path = Path(MODELS_REGISTRY_DIR) / "production_history.json"
        history_data = {}
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)
            except Exception:
                pass
                
        model_history = history_data.setdefault(model_name, [])
        
        # Build entry
        entry = metadata_copy.copy()
        entry["model_version"] = version_id
        entry["model_name"] = model_name
        entry["is_production"] = True
        
        # Append and keep last 5
        model_history.append(entry)
        history_data[model_name] = model_history[-5:]
        
        try:
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)
        except Exception:
            pass
                            
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_copy, f, indent=2)

def load_production_model(model_name: str) -> tuple:
    """
    Loads the current production version of a model.
    Returns:
      - model_obj: loaded sklearn/IF model
      - metadata: dict of metadata
    Raises FileNotFoundError if no production version is found.
    """
    parent_dir = get_model_registry_dir(model_name)
    if not parent_dir.exists():
        raise FileNotFoundError(f"Model registry for '{model_name}' does not exist.")
        
    for v_dir in parent_dir.iterdir():
        if v_dir.is_dir():
            meta_file = v_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta_data = json.load(f)
                    if meta_data.get("is_production", False):
                        model_path = v_dir / "model.pkl"
                        if model_path.exists():
                            model_obj = joblib.load(model_path)
                            meta_data["version_id"] = v_dir.name
                            return model_obj, meta_data
                except Exception as e:
                    # Log error or continue to check others
                    continue
                    
    raise FileNotFoundError(f"No production model found for '{model_name}'.")

def list_model_versions(model_name: str) -> list:
    """
    Returns all registered versions of a model sorted by timestamp descending.
    Each item in the list is a dictionary containing version_id and its metadata.
    """
    parent_dir = get_model_registry_dir(model_name)
    if not parent_dir.exists():
        return []
        
    versions = []
    for v_dir in parent_dir.iterdir():
        if v_dir.is_dir():
            meta_file = v_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta_data = json.load(f)
                    versions.append({
                        "version_id": v_dir.name,
                        "metadata": meta_data
                    })
                except Exception:
                    pass
                    
    # Sort by timestamp descending. Handle missing timestamp gracefully.
    versions.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
    return versions

def is_cold_start(model_name: str) -> bool:
    """
    Returns True if no registered model folder exists for the given model_name,
    or if it exists but contains no subdirectories.
    """
    parent_dir = get_model_registry_dir(model_name)
    if not parent_dir.exists():
        return True
    subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    return len(subdirs) == 0
