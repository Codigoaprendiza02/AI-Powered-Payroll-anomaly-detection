import uuid
from datetime import datetime
import os

# Random Seed
RANDOM_SEED = 42

# Store Directories
BASE_STORE_DIR = "store"
FEATURES_STORE_DIR = os.path.join(BASE_STORE_DIR, "features")
DRIFT_STORE_DIR = os.path.join(BASE_STORE_DIR, "drift")
RISK_REGISTER_DIR = os.path.join(BASE_STORE_DIR, "risk_register")
FORECASTS_DIR = os.path.join(BASE_STORE_DIR, "forecasts")
AUDIT_LOG_DIR = os.path.join(BASE_STORE_DIR, "audit_log")

# Models Registry
MODELS_REGISTRY_DIR = os.path.join("models", "registry")

# API Configuration
API_KEY = "super_secret_payroll_key_123"

def generate_run_id() -> str:
    # Use datetime.utcnow() to generate run ID timestamp
    return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# Ensure directories exist
for directory in [FEATURES_STORE_DIR, DRIFT_STORE_DIR, RISK_REGISTER_DIR, FORECASTS_DIR, AUDIT_LOG_DIR, MODELS_REGISTRY_DIR]:
    os.makedirs(directory, exist_ok=True)
