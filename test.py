import requests

BASE_URL = "http://127.0.0.1:8000"


# ======================================================
# TEST EMPLOYEE RISK
# ======================================================

employee_data = {
    "LOP_days": 2,
    "paid_leaves": 1,
    "overtime_hours": 6,
    "attendance_ratio": 0.92,

    "salary_diff": 12000,
    "salary_dev_pct": 14,
    "deduction_anomaly_score": 0.3,
    "salary_anomaly_score": 0.6,
    "rule_violation_score": 0.2
}

print("\n========== EMPLOYEE PREDICTION ==========")

response = requests.post(
    f"{BASE_URL}/predict-employee",
    json=employee_data
)

print("Status:", response.status_code)

try:
    print(response.json())
except:
    print("Raw response:", response.text)


# ======================================================
# TEST FORECASTING
# ======================================================

forecast_data = {
    "data": [
        {
            "year_month": "2024-01-01",
            "updated_net_salary": 200000
        },
        {
            "year_month": "2024-02-01",
            "updated_net_salary": 210000
        },
        {
            "year_month": "2024-03-01",
            "updated_net_salary": 215000
        }
    ]
}

print("\n========== PAYROLL FORECAST ==========")

response = requests.post(
    f"{BASE_URL}/forecast-payroll",
    json=forecast_data
)


print("Status:", response.status_code)

try:
    print(response.json())
except:
    print("Raw response:", response.text)


# ======================================================
# TEST SYSTEM ALERTS
# ======================================================

print("\n========== SYSTEM ALERTS ==========")

response = requests.post(
    f"{BASE_URL}/system-monitor",
    json={"data": [employee_data]}
)

print("Status:", response.status_code)

try:
    print(response.json())
except:
    print("Raw response:", response.text)


# ======================================================
# TEST MODEL MONITORING
# ======================================================

print("\n========== MODEL MONITORING ==========")

response = requests.post(
    f"{BASE_URL}/model-monitoring",
    json=forecast_data
)

print("Status:", response.status_code)

try:
    print(response.json())
except:
    print("Raw response:", response.text)