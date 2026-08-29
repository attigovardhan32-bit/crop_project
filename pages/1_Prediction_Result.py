"""
Crop Yield Prediction — Dedicated Results Dashboard.
Displays high-precision prediction outputs, input parameter breakdown,
model evaluation benchmarks, and interactive visual charts.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Safe UTF-8 encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import load_dataset
from src.model import load_trained_model, load_model_metrics, predict_yield
from src.utils import generate_farm_advisory, get_country_defaults
from src.evaluation import (
    plot_historical_and_forecast,
    plot_actual_vs_predicted,
    plot_feature_importance_top,
    plot_metrics_summary,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction Results | AgroYield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# LOAD CUSTOM CSS
# -----------------------------------------------------------------------------
css_path = os.path.join(ROOT_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RETRIEVE OR INITIALIZE PREDICTION DATA
# -----------------------------------------------------------------------------
metrics = load_model_metrics()
pipeline = load_trained_model()
df = load_dataset()

# If navigated directly without session state, populate default sample
if "latest_prediction" not in st.session_state:
    sample_input = {
        "Area": "India",
        "Item": "Rice, paddy",
        "Year": 2024,
        "average_rain_fall_mm_per_year": 1080.0,
        "pesticides_tonnes": 550.0,
        "avg_temp": 26.5,
    }
    sample_pred = predict_yield(sample_input)
    sample_adv = generate_farm_advisory(
        crop="Rice, paddy",
        country="India",
        rainfall_mm=1080.0,
        temp_c=26.5,
        soil_type="Alluvial Soil",
    )
    st.session_state["latest_prediction"] = {
        "inputs": sample_input,
        "result": sample_pred,
        "advisory": sample_adv,
        "farm_area": 2.5,
    }

pred_bundle = st.session_state["latest_prediction"]
inp = pred_bundle["inputs"]
res = pred_bundle["result"]
adv = pred_bundle["advisory"]
farm_area = pred_bundle.get("farm_area", 2.5)

hg_yield = res["predicted_yield_hg_ha"]
tonnes_yield = res["predicted_yield_tonnes_ha"]
total_tonnes = tonnes_yield * farm_area

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Result Navigation")
    st.page_link("app.py", label="⬅️ Back to Main Predictor", icon="🏠")
    st.page_link("pages/2_Model_Performance.py", label="🧪 Model Performance", icon="🔬")
    st.page_link("pages/3_Tree_Calculator.py", label="🌳 Tree Planting Calculator", icon="🌲")
    st.page_link("pages/4_Farm_Advisory.py", label="🌾 Farm Advisory", icon="🌦")
    st.markdown("---")
    st.markdown("#### 📋 Active Prediction Query")
    st.markdown(f"**Country:** `{inp['Area']}`")
    st.markdown(f"**Crop:** `{inp['Item']}`")
    st.markdown(f"**Target Year:** `{inp['Year']}`")
    st.markdown(f"**Rainfall:** `{inp['average_rain_fall_mm_per_year']} mm`")
    st.markdown(f"**Pesticides:** `{inp['pesticides_tonnes']} tonnes`")
    st.markdown(f"**Temperature:** `{inp['avg_temp']} °C`")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <span class="hero-badge">🌳 Official Machine Learning Inference</span>
        <h1>🌾 Crop Yield Prediction Result</h1>
        <p style="font-size: 1.1rem; color: #c8e6c9;">
            Comprehensive yield forecast, feature impact analysis, historical trends, and agronomic insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HERO PREDICTION CARD
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="yield-hero-card">
        <h4>🎯 PREDICTED HARVEST YIELD — {inp['Item'].upper()} ({inp['Area'].upper()})</h4>
        <div class="yield-value">{hg_yield:,.2f} <span style="font-size: 1.6rem; color: #a7f3d0;">hg/ha</span></div>
        <div class="yield-subvalue">📦 Equivalent to <b>{tonnes_yield:,.2f} Tonnes per Hectare</b></div>
        <div style="margin-top: 15px; font-size: 1.05rem; color: #e8f5e9; background: rgba(0,0,0,0.25); display: inline-block; padding: 6px 20px; border-radius: 20px;">
            Estimated Total Production for <b>{farm_area:.1f} ha</b> Farm: <b>{total_tonnes:,.2f} Tonnes</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# METRICS ROW
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
r2_val = metrics.get("r2_test", 0.9844) if metrics else 0.9844
mae_val = metrics.get("mae", 4134.34) if metrics else 4134.34
rmse_val = metrics.get("rmse", 10641.67) if metrics else 10641.67
algo_name = res.get("model_type", "RandomForestRegressor")

with m1:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Model Validation R²</div>
            <div class="metric-number">{r2_val:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-number">{mae_val:,.0f} <span style="font-size:0.9rem">hg/ha</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Root Mean Sq. Error</div>
            <div class="metric-number">{rmse_val:,.0f} <span style="font-size:0.9rem">hg/ha</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Model Pipeline</div>
            <div class="metric-number" style="font-size: 1.25rem; margin-top: 8px;">{algo_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INPUT BREAKDOWN & HISTORICAL COMPARISON
# -----------------------------------------------------------------------------
st.markdown("### 📋 Input Summary & Historical Baseline Comparison")

c_sub = df[(df["Area"] == inp["Area"]) & (df["Item"] == inp["Item"])]
hist_avg_yield = c_sub["hg/ha_yield"].mean() if not c_sub.empty else hg_yield
hist_avg_rain = c_sub["average_rain_fall_mm_per_year"].mean() if not c_sub.empty else inp["average_rain_fall_mm_per_year"]
hist_avg_temp = c_sub["avg_temp"].mean() if not c_sub.empty else inp["avg_temp"]

yield_diff_pct = ((hg_yield - hist_avg_yield) / hist_avg_yield) * 100 if hist_avg_yield > 0 else 0

col_summary_1, col_summary_2 = st.columns([1, 1])

with col_summary_1:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #a7f3d0; margin-bottom: 12px;">🌾 Active Input Vector</h4>
            <table style="width: 100%; color: #e8f5e9; font-size: 0.95rem;">
                <tr><td style="padding: 6px 0;"><b>Country / Area:</b></td><td style="text-align: right;">{inp['Area']}</td></tr>
                <tr><td style="padding: 6px 0;"><b>Crop Variety:</b></td><td style="text-align: right;">{inp['Item']}</td></tr>
                <tr><td style="padding: 6px 0;"><b>Harvest Year:</b></td><td style="text-align: right;">{inp['Year']}</td></tr>
                <tr><td style="padding: 6px 0;"><b>Rainfall (mm/yr):</b></td><td style="text-align: right;">{inp['average_rain_fall_mm_per_year']:,.1f} mm</td></tr>
                <tr><td style="padding: 6px 0;"><b>Pesticides (tonnes):</b></td><td style="text-align: right;">{inp['pesticides_tonnes']:,.1f} tonnes</td></tr>
                <tr><td style="padding: 6px 0;"><b>Temperature (°C):</b></td><td style="text-align: right;">{inp['avg_temp']:.1f} °C</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_summary_2:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #a7f3d0; margin-bottom: 12px;">📊 Benchmark Comparison ({inp['Area']})</h4>
            <table style="width: 100%; color: #e8f5e9; font-size: 0.95rem;">
                <tr><td style="padding: 6px 0;"><b>Historical Avg Yield:</b></td><td style="text-align: right;">{hist_avg_yield:,.0f} hg/ha ({hist_avg_yield/10000:.2f} t/ha)</td></tr>
                <tr><td style="padding: 6px 0;"><b>Predicted vs Historical:</b></td><td style="text-align: right; color: {'#00e676' if yield_diff_pct >= 0 else '#ff7043'};"><b>{yield_diff_pct:+.1f}%</b></td></tr>
                <tr><td style="padding: 6px 0;"><b>Historical Avg Rainfall:</b></td><td style="text-align: right;">{hist_avg_rain:,.0f} mm/year</td></tr>
                <tr><td style="padding: 6px 0;"><b>Historical Avg Temp:</b></td><td style="text-align: right;">{hist_avg_temp:.1f} °C</td></tr>
                <tr><td style="padding: 6px 0;"><b>Target Seasonality:</b></td><td style="text-align: right;">{'Future Forecast' if inp['Year'] > 2013 else 'Historical Calibration'}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GRAPH VISUALIZATIONS SECTION
# -----------------------------------------------------------------------------
st.markdown("### 📈 Comprehensive Visual Analytics")

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_historical = plot_historical_and_forecast(
        pipeline=pipeline,
        country=inp["Area"],
        crop=inp["Item"],
        selected_year=inp["Year"],
        current_prediction_hg_ha=hg_yield,
        rainfall=inp["average_rain_fall_mm_per_year"],
        pesticides=inp["pesticides_tonnes"],
        temperature=inp["avg_temp"],
    )
    st.plotly_chart(fig_historical, use_container_width=True)

with col_g2:
    fig_fi = plot_feature_importance_top(pipeline=pipeline, top_n=12)
    st.plotly_chart(fig_fi, use_container_width=True)

col_g3, col_g4 = st.columns(2)

with col_g3:
    fig_scatter = plot_actual_vs_predicted(pipeline=pipeline, sample_size=1000)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_g4:
    fig_metrics = plot_metrics_summary(metrics)
    st.plotly_chart(fig_metrics, use_container_width=True)

# -----------------------------------------------------------------------------
# AGRONOMIC INTERPRETATION & RECOMMENDATIONS
# -----------------------------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### 💡 Agronomic Interpretation & Field Guidance")

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #81c784;">🌧️ Climate & Moisture Dynamics</h4>
            <p><b>Rainfall Assessment:</b> {adv['rain_status']}</p>
            <p><b>Thermal Balance:</b> {adv['temp_status']}</p>
            <p><b>Management Note:</b> Yield is heavily dependent on soil moisture retention during the flowering and grain-filling / tuberization phases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_rec2:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #81c784;">🧪 Soil & Nutrient Management</h4>
            <p><b>Soil Compatibility:</b> Selected `{adv['soil_type']}` (Optimal: `{adv['ideal_soil']}`).</p>
            <p><b>N-P-K Guideline:</b> {adv['npk_advice']}</p>
            <p><b>Target Recommendation:</b> <code>{adv['recommended_npk']}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# NAVIGATION FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.page_link("app.py", label="🔄 New Yield Prediction", icon="🌱", use_container_width=True)
with nav_col2:
    st.page_link("pages/3_Tree_Calculator.py", label="🌳 Agroforestry Planning", icon="🌲", use_container_width=True)
with nav_col3:
    st.page_link("pages/2_Model_Performance.py", label="🔬 Model Architecture & R²", icon="🧪", use_container_width=True)
