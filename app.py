import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body
from typing import Dict, List

app = FastAPI(title="AI Powered Payroll Intelligence System")

# ======================================================
# LOAD MODELS
# ======================================================

absenteeism_model = joblib.load("saved_models/absenteeism_risk_model.pkl")
salary_risk_model = joblib.load("saved_models/salary_risk_model.pkl")

payroll_forecast_model = joblib.load("saved_models/payroll_forecast_model.pkl")
dept_forecast_model = joblib.load("saved_models/department_forecast_model.pkl")
overtime_forecast_model = joblib.load("saved_models/overtime_forecast_model.pkl")

# ======================================================
# LOAD FEATURE LISTS
# ======================================================

employee_features = joblib.load("saved_models/employee_features.pkl")
risk_features = joblib.load("saved_models/risk_features.pkl")
forecast_features = joblib.load("saved_models/forecast_features.pkl")


# ======================================================
# HELPER FUNCTION
# ======================================================

def prepare_features(data: Dict, features: List[str]) -> pd.DataFrame:
    """
    Ensures the dataframe has correct columns
    and correct feature order for sklearn models
    """

    df = pd.DataFrame([data])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    return df

def generate_system_alerts(payroll_data):

    alerts = []

    salary_diff = payroll_data['salary_diff']

    # 1️⃣ Mean shift detection (concept drift)
    mean_shift = abs(salary_diff.mean())
    std_dev = salary_diff.std()

    if std_dev > 0 and mean_shift > 0.5 * std_dev:
        alerts.append({
            "type": "DATA_DRIFT",
            "severity": "HIGH",
            "message": "Possible payroll logic drift detected.",
            "metric": float(mean_shift)
        })

    # 2️⃣ High anomaly concentration
    if 'final_risk_level' in payroll_data.columns:

        high_ratio = (payroll_data['final_risk_level'] == 3).mean()

        if high_ratio > 0.15:
            alerts.append({
                "type": "ANOMALY_SPIKE",
                "severity": "CRITICAL",
                "message": "Abnormally high anomaly concentration detected.",
                "ratio": float(high_ratio)
            })

    # 3️⃣ Uniform systematic payroll shift
    mad = np.median(
        np.abs(
            salary_diff - np.median(salary_diff)
        )
    )

    if mad < 0.01:
        alerts.append({
            "type": "SYSTEMATIC_SHIFT",
            "severity": "CRITICAL",
            "message": "Possible systematic uniform shift in payroll."
        })

    # 4️⃣ Payroll spike detection
    if 'updated_net_salary' in payroll_data.columns:

        payroll_std = payroll_data['updated_net_salary'].std()

        if payroll_std > 0 and payroll_data['updated_net_salary'].max() > payroll_data['updated_net_salary'].mean() + 3 * payroll_std:

            alerts.append({
                "type": "PAYROLL_SPIKE",
                "severity": "MEDIUM",
                "message": "Unusual payroll spike detected."
            })

    if len(alerts) == 0:
        alerts.append({
            "type": "SYSTEM_HEALTHY",
            "severity": "INFO",
            "message": "No anomalies detected."
        })

    return alerts

# ======================================================
# EMPLOYEE RISK PREDICTION
# ======================================================

@app.post("/predict-employee")
def predict_employee(data: Dict):

    X_emp = prepare_features(data, employee_features)
    X_risk = prepare_features(data, risk_features)

    absentee_prob = absenteeism_model.predict_proba(X_emp)[0][1]

    salary_pred = salary_risk_model.predict(X_risk)[0]
    salary_risk = 1 if salary_pred == -1 else 0

    return {
        "absenteeism_risk": float(absentee_prob),
        "salary_manipulation_risk": float(salary_pred)
    }


# ======================================================
# PAYROLL FORECASTING
# ======================================================

@app.post("/forecast-payroll")
def forecast_payroll(payload: dict):

    df = pd.DataFrame(payload["data"])

    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month")

    df["month_index"] = range(1, len(df) + 1)

    X = df[forecast_features]

    payroll_pred = payroll_forecast_model.predict(X)
    dept_pred = dept_forecast_model.predict(X)
    overtime_pred = overtime_forecast_model.predict(X)

    return {
        "payroll_prediction": payroll_pred.tolist(),
        "department_prediction": dept_pred.tolist(),
        "overtime_prediction": overtime_pred.tolist()
    }


# ======================================================
# SYSTEM MONITOR
# ======================================================

@app.post("/system-monitor")
def system_monitor(data: list = Body(..., embed=True)):

    df = pd.DataFrame(data)

    alerts = generate_system_alerts(df)

    return {
        "system_alerts": alerts
    }


# ======================================================
# MODEL MONITORING
# ======================================================

@app.post("/model-monitoring")
def model_monitoring(payload: dict):

    df = pd.DataFrame(payload["data"])

    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month")

    df["month_index"] = range(1, len(df) + 1)

    X = df[forecast_features]

    preds = payroll_forecast_model.predict(X)
    actual = df["updated_net_salary"]

    error = abs(actual - preds)
    drift_score = error.mean()

    alerts = []

    if drift_score > 50000:
        alerts.append({
            "type": "Accuracy Decay",
            "severity": "HIGH"
        })

    return {
        "drift_score": float(drift_score),
        "alerts": alerts
    }