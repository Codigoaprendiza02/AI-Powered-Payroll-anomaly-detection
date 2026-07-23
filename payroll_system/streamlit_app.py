import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Defensive sys.path append for imports when run from different directories
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))



from config.settings import (
    AUDIT_LOG_DIR, FORECASTS_DIR, RISK_REGISTER_DIR, 
    FEATURES_STORE_DIR, MODELS_REGISTRY_DIR, DRIFT_STORE_DIR
)
from pipelines.pipeline_runner import run_full_pipeline

# 1. Page Config
st.set_page_config(
    page_title="Payroll Intelligence & Anomaly Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Theme state setup
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Colors Configuration
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"
radius = "10px"

# Inject Global CSS
st.markdown(f"""
<style>
:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {green};
    --green-muted: {green_muted};
    --red: {red};
    --red-muted: {red_muted};
    --amber: {amber};
    --amber-muted: {amber_muted};
    --shadow: {shadow};
    --radius: {radius};
}}

/* Hide Streamlit default components */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}
.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1360px !important;
}}

/* Custom Tabs layout styling */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px;
}}

/* Styled wrappers and metric cards */
.card-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
}}
.metric-label {{
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
}}
.metric-delta {{
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.4rem;
    padding: 2px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-warn {{ color: var(--amber); background: var(--amber-muted); }}

.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.2rem 0.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.chart-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
}}
.chart-subtitle {{
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}}

/* Custom tables layout */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
}}
.data-table td {{
    padding: 0.65rem 0.8rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}

/* Badges styling */
.badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}

[data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
</style>
""", unsafe_allow_html=True)

# 4. Header Section
head_left, head_right = st.columns([8, 1.5])
with head_left:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 1.5rem;'>
        <span style='font-size: 1.5rem; color: var(--accent); font-weight: 800;'>◆</span>
        <span style='font-weight: 700; font-size: 1.35rem; color: var(--text); letter-spacing: -0.02em;'>Payroll Intelligence System</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# Helper: custom metric card
def metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Helper: load files
def get_last_run_id() -> str:
    last_run_file = Path(AUDIT_LOG_DIR) / "last_run_id.txt"
    if last_run_file.exists():
        with open(last_run_file, "r") as f:
            return f.read().strip()
    return ""

def load_run_summary(run_id: str) -> dict:
    summary_path = Path(AUDIT_LOG_DIR) / f"{run_id}_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# 5. Application Tabs Navigation
tab_upload, tab_profiles, tab_analytics, tab_system = st.tabs([
    "Upload & Ingest", "High-Risk Profiles", "Forecast & Drift", "System Status"
])

# Plotly styling templates
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=40, r=20, t=25, b=40),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# TAB 1: UPLOAD & INGEST
with tab_upload:
    st.markdown("### Payroll Data Ingestion")
    st.write("Upload a new monthly payroll report to automatically trigger the 8-stage anomaly detection and intelligence runner.")
    
    uploaded_file = st.file_uploader("Choose a payroll file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        suffix = os.path.splitext(file_name)[1]
        
        # Save uploaded file locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        btn_trigger = st.button("Trigger Full Pipeline Ingestion", type="primary")
        
        if btn_trigger:
            with st.spinner("Executing pipeline pipelines sequentially..."):
                try:
                    # Run the full pipeline runner
                    run_id = run_full_pipeline(tmp_path)
                    st.success(f"Ingestion pipeline completed successfully! Run ID: {run_id}")
                except Exception as e:
                    st.error(f"Pipeline Execution Failed: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
    # Show summary statistics of latest run if it exists
    last_run_id = get_last_run_id()
    if last_run_id:
        st.markdown(f"#### Latest Ingested Run Analysis: `{last_run_id}`")
        summary = load_run_summary(last_run_id)
        if summary:
            # Row 1: KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            inf_metrics = summary.get("inference", {})
            total_rec = inf_metrics.get("total_records", 0)
            anomaly_rate = inf_metrics.get("anomaly_rate", 0.0) * 100
            risk_tiers = inf_metrics.get("risk_tiers", {})
            high_risk = risk_tiers.get("HIGH", 0) + risk_tiers.get("CRITICAL", 0)
            
            with c1:
                metric_card("Total Employee Records", f"{total_rec:,}")
            with c2:
                metric_card("Statistical Anomaly Rate", f"{anomaly_rate:.2f}%", 
                            delta="Active Anomalies" if anomaly_rate > 5 else "Normal",
                            delta_type="down" if anomaly_rate > 5 else "up")
            with c3:
                metric_card("Critical & High Risk Cases", f"{high_risk}", 
                            delta="Action Required" if high_risk > 0 else "Clear",
                            delta_type="down" if high_risk > 0 else "up")
            with c4:
                # Load latest drift report if exists
                drift_path = Path(AUDIT_LOG_DIR) / f"{last_run_id}_drift_report.json"
                drift_status = "Skipped (Baseline)"
                if drift_path.exists():
                    try:
                        with open(drift_path, "r") as f:
                            drift_data = json.load(f)
                            drift_status = drift_data.get("overall_severity", "No Drift")
                    except Exception:
                        pass
                metric_card("Data Drift Status", drift_status, 
                            delta="Calibrated" if "No Drift" in drift_status else "Drift Warning",
                            delta_type="up" if "No" in drift_status or "Baseline" in drift_status else "warn")
                
            # Row 2: Sub-dimension anomaly indicators
            st.markdown("##### Sub-Dimension Vulnerability Counts")
            c1, c2, c3 = st.columns(3)
            risk_mon = summary.get("risk_monitoring", {})
            with c1:
                metric_card("Absenteeism Risk Flags", f"{risk_mon.get('absenteeism_risk_count', 0)}")
            with c2:
                metric_card("Overtime Abuse Flags", f"{risk_mon.get('overtime_abuse_count', 0)}")
            with c3:
                metric_card("Salary Manipulation Flags", f"{risk_mon.get('salary_manipulation_count', 0)}")
        else:
            st.info("Run details are being compiled. Refresh the page shortly.")
    else:
        st.info("No runs found in directory. Please upload a file to analyze.")

# TAB 2: HIGH RISK EMPLOYEE PROFILES
with tab_profiles:
    st.markdown("### High-Risk Employee Profiles")
    st.write("Inspect details for employees flagged as HIGH or CRITICAL risk during the latest evaluation run.")
    
    last_run_id = get_last_run_id()
    if last_run_id:
        high_risk_csv = Path(AUDIT_LOG_DIR) / f"{last_run_id}_high_risk_employees.csv"
        if high_risk_csv.exists():
            df_hr = pd.read_csv(high_risk_csv)
            if not df_hr.empty:
                # Add search filter
                search_id = st.text_input("Filter by Employee ID", "").strip()
                if search_id:
                    df_disp = df_hr[df_hr["employee_id"].astype(str).str.contains(search_id, case=False)]
                else:
                    df_disp = df_hr
                
                # Render table
                st.markdown(f"**Flagged Employees List ({len(df_disp)} records shown):**")
                
                # Display table of rows and columns using st.dataframe
                df_table = df_disp.copy()
                # Ensure columns are ordered and named nicely
                cols_to_keep = ["employee_id", "department", "designation", "net_salary", "risk_tier", "risk_score"]
                # Map any alternate column names
                if "take_home" in df_table.columns and "net_salary" not in df_table.columns:
                    df_table["net_salary"] = df_table["take_home"]
                if "anomaly_score" in df_table.columns and "risk_score" not in df_table.columns:
                    df_table["risk_score"] = df_table["anomaly_score"]
                
                # Filter to available requested columns
                cols_present = [c for c in cols_to_keep if c in df_table.columns]
                df_table = df_table[cols_present]
                
                # Format headers for display
                df_table.columns = [c.replace("_", " ").title() for c in df_table.columns]
                
                st.dataframe(df_table, use_container_width=True, hide_index=True)
                st.write("")
                
                # Select employee details card
                selected_emp = st.selectbox("Select flagged Employee ID to inspect profile:", df_disp["employee_id"].tolist())
                if selected_emp:
                    # Fetch detailed register profile
                    register_path = Path(RISK_REGISTER_DIR) / "register.json"
                    emp_details = None
                    if register_path.exists():
                        try:
                            with open(register_path, "r") as f:
                                register = json.load(f)
                            emp_details = next((e for e in register if e["employee_id"] == selected_emp), None)
                        except Exception:
                            pass
                    
                    # Fetch explanation from inference parquet
                    inf_parquet = Path(AUDIT_LOG_DIR) / f"{last_run_id}_inference_results.parquet"
                    explanation = "N/A"
                    if inf_parquet.exists():
                        try:
                            df_inf = pd.read_parquet(inf_parquet)
                            emp_col = "employee_id" if "employee_id" in df_inf.columns else df_inf.index.name
                            if not emp_col:
                                emp_col = "index"
                            row = df_inf[df_inf[emp_col].astype(str) == str(selected_emp)]
                            if not row.empty:
                                explanation = row.iloc[0].get("explanation", "N/A")
                        except Exception:
                            pass
                    
                    st.markdown(f"#### Employee Risk Profile: `{selected_emp}`")
                    
                    c1, c2 = st.columns([1.5, 2.5])
                    with c1:
                        st.markdown("<div class='card-wrap'>", unsafe_allow_html=True)
                        st.write(f"**Employee ID:** {selected_emp}")
                        if emp_details:
                            st.write(f"**Risk Score:** {emp_details.get('current_score', 0.0):.3f}")
                            st.write(f"**Trend Indicator:** {emp_details.get('trend', 'N/A')}")
                            st.markdown("**Risk Dimension Metrics:**")
                            for k, v in emp_details.get("risk_dimensions", {}).items():
                                st.write(f"- {k.replace('_', ' ').title()}: {v:.2f}")
                        else:
                            st.write("Register details not found.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown("<div class='card-wrap'>", unsafe_allow_html=True)
                        st.markdown("**Core Anomaly Explanation & Action Item:**")
                        st.write(explanation)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success("Excellent! No high-risk employees flagged in this run.")
        else:
            st.info("High-risk employee records are missing for the latest run ID.")
    else:
        st.info("No run history loaded. Navigate to 'Upload & Ingest' to parse a payroll dataset.")

# TAB 3: FORECASTS & DRIFT
with tab_analytics:
    st.markdown("### Forecasts & Data Drift Analytics")
    
    last_run_id = get_last_run_id()
    if last_run_id:
        # 1. Plotly Forecast Chart
        forecast_path = Path(FORECASTS_DIR) / f"{last_run_id}_forecast.json"
        if forecast_path.exists():
            try:
                with open(forecast_path, "r") as f:
                    fc = json.load(f)
                    
                periods = fc.get("forecast_periods", [])
                values = fc.get("forecast_values", [])
                
                if periods and values:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=periods, y=values, 
                        mode='lines+markers',
                        name='Payroll Forecast',
                        line=dict(color='#2563eb', width=3),
                        marker=dict(size=7, color='#2563eb')
                    ))
                    
                    fig.update_layout(
                        title=f"Total Company Payroll Forecast Projection (Latest Run ID: {last_run_id})",
                        xaxis_title="Future Month Projections",
                        yaxis_title="Total Payroll Pay ($)",
                        **PLOT_LAYOUT
                    )
                    
                    st.markdown("""<div class="chart-wrap">""", unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not load forecast chart: {str(e)}")
        else:
            st.info("Forecast data missing for this run.")
            
        # 2. Data Drift PSI Table
        st.write("")
        st.markdown("#### Feature Data Drift Monitor")
        drift_path = Path(AUDIT_LOG_DIR) / f"{last_run_id}_drift_report.json"
        if drift_path.exists():
            try:
                with open(drift_path, "r") as f:
                    drift_data = json.load(f)
                
                if drift_data.get("cold_start", False):
                    st.info("Cold Start: This is the baseline run. Baseline distribution thresholds successfully saved to registry. Skip drift calculation.")
                else:
                    sev = drift_data.get("overall_severity", "No Drift")
                    sev_class = "badge-green" if "No" in sev else ("badge-amber" if "Moderate" in sev else "badge-red")
                    st.markdown(f"Overall Data Drift Alert Status: <span class=\"badge {sev_class}\">{sev}</span>", unsafe_allow_html=True)
                    st.write(f"**Max Population Stability Index (PSI):** {drift_data.get('max_psi', 0.0):.4f}")
                    
                    feature_psi = drift_data.get("feature_psi", {})
                    if feature_psi:
                        df_drift = pd.DataFrame([
                            {
                                "Evaluated Feature Column": col,
                                "PSI Value Score": round(psi_val, 4),
                                "Alert Indicator": "Normal" if psi_val < 0.1 else ("Moderate" if psi_val <= 0.25 else "Action Required")
                            }
                            for col, psi_val in feature_psi.items()
                        ])
                        st.dataframe(df_drift, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error parsing drift report: {str(e)}")
        else:
            st.info("No drift report compiled yet (indicates baseline cold start or missing file).")
    else:
        st.info("No run logs detected. Upload data first.")

# TAB 4: SYSTEM STATUS & ALERTS LOG
with tab_system:
    st.markdown("### System Health Diagnostician & Alerts Log")
    
    # 1. Health Diagnostics
    st.markdown("#### Directory Store Statuses")
    components = {
        "Feature Parquet Store": Path(FEATURES_STORE_DIR).exists(),
        "Model Registry Manager": Path(MODELS_REGISTRY_DIR).exists(),
        "Drift Reference Store": Path(DRIFT_STORE_DIR).exists(),
        "Active Risk Register": Path(RISK_REGISTER_DIR).exists(),
        "Run Audit Logging Store": Path(AUDIT_LOG_DIR).exists()
    }
    
    # Render component health as a table of rows and columns
    df_health = pd.DataFrame([
        {
            "Component Engine": k,
            "Diagnostic Status": "🟢 HEALTHY" if v else "🔴 ERROR (Missing Directory)"
        }
        for k, v in components.items()
    ])
    st.table(df_health)
    
    # 2. Alerts Log
    st.write("")
    st.markdown("#### System Alerts Log (Latest Run)")
    last_run_id = get_last_run_id()
    if last_run_id:
        alerts_file = Path(AUDIT_LOG_DIR) / f"{last_run_id}_alerts.json"
        if alerts_file.exists():
            try:
                with open(alerts_file, "r") as f:
                    alerts = json.load(f)
                    
                if alerts:
                    df_alerts = pd.DataFrame(alerts)
                    # Select and rename columns for display
                    cols_mapping = {
                        "timestamp": "Timestamp",
                        "severity": "Severity",
                        "alert_type": "Alert Type",
                        "affected_entity": "Affected Component",
                        "trigger_value": "Trigger Value",
                        "recommended_action": "Action Recommendation"
                    }
                    # Filter and reorder columns
                    present_cols = [c for c in cols_mapping.keys() if c in df_alerts.columns]
                    df_alerts = df_alerts[present_cols]
                    df_alerts = df_alerts.rename(columns=cols_mapping)
                    
                    # Format timestamp if present
                    if "Timestamp" in df_alerts.columns:
                        df_alerts["Timestamp"] = df_alerts["Timestamp"].astype(str).str.slice(0, 19)
                        
                    st.table(df_alerts)
                else:
                    st.success("Ideal Status: 0 active system anomalies flagged for this run.")
            except Exception as e:
                st.error(f"Error reading alerts file: {str(e)}")
        else:
            st.info("No active alerts logged for the latest run.")
    else:
        st.info("No run logs found in dashboard cache.")
