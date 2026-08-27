"""
Farm & Soil Advisory Intelligence Dashboard.
Provides agronomic guidelines, crop-specific climate thresholds,
N-P-K nutrient management, and soil health recommendations.
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go

# Safe UTF-8 encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import load_dataset
from src.utils import generate_farm_advisory, get_country_and_crop_lists

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Farm & Soil Advisory | AgroYield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(ROOT_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD METADATA
# -----------------------------------------------------------------------------
df = load_dataset()
countries, all_crops, _ = get_country_and_crop_lists(df)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 AgroYield AI")
    st.page_link("app.py", label="🏠 Yield Prediction Home", icon="🌱")
    st.page_link("pages/1_Prediction_Result.py", label="📊 Results Dashboard", icon="📈")
    st.page_link("pages/2_Model_Performance.py", label="🧪 Model Performance", icon="🔬")
    st.page_link("pages/3_Tree_Calculator.py", label="🌳 Tree Planting Calculator", icon="🌲")
    st.markdown("---")
    st.markdown("#### 🌾 Advisory Scope")
    st.caption("Advisory recommendations combine global agronomic agronomy standards with input environmental parameters.")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <span class="hero-badge">🌦 Agronomic Decision Support</span>
        <h1>🌾 Weather, Soil & Farm Advisory Intelligence</h1>
        <p style="font-size: 1.15rem; color: #c8e6c9;">
            Data-backed crop management guidelines, optimal soil conditions, 
            nutrient rebalancing, and climate adaptation strategies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# ADVISORY INPUTS
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Select Farm Conditions")

c_in1, c_in2, c_in3 = st.columns(3)

with c_in1:
    adv_crop = st.selectbox("🌱 Crop Variety", options=all_crops, index=all_crops.index("Rice, paddy") if "Rice, paddy" in all_crops else 0)
    adv_country = st.selectbox("🌍 Region / Country", options=countries, index=countries.index("India") if "India" in countries else 0)

with c_in2:
    adv_soil = st.selectbox("🌱 Current Soil Type", ["Alluvial Soil", "Black Soil", "Clay Loam", "Loamy Soil", "Sandy Loam", "Red Soil"], index=0)
    adv_rain = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, max_value=6000.0, value=1100.0, step=50.0)

with c_in3:
    adv_temp = st.number_input("🌡️ Average Temperature (°C)", min_value=-15.0, max_value=50.0, value=26.0, step=0.5)
    adv_n = st.number_input("🧪 Nitrogen N (kg/ha)", min_value=0.0, max_value=400.0, value=90.0, step=5.0)

# Optional Nutrients
with st.expander("🧪 Phosphorus (P) & Potassium (K) Baseline Levels"):
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        adv_p = st.number_input("Phosphorus P (kg/ha)", min_value=0.0, max_value=300.0, value=45.0, step=5.0)
    with p_col2:
        adv_k = st.number_input("Potassium K (kg/ha)", min_value=0.0, max_value=300.0, value=40.0, step=5.0)

# Generate Advisory
adv_output = generate_farm_advisory(
    crop=adv_crop,
    country=adv_country,
    rainfall_mm=adv_rain,
    temp_c=adv_temp,
    soil_type=adv_soil,
    nitrogen_n=adv_n,
    phosphorus_p=adv_p,
    potassium_k=adv_k,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ADVISORY OUTPUT CARDS
# -----------------------------------------------------------------------------
st.markdown(f"### 📋 Agronomic Report: **{adv_crop}** ({adv_country})")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #a7f3d0;">🌧️ Hydrological & Thermal Assessment</h4>
            <p><b>Precipitation Status:</b> {adv_output['rain_status']}</p>
            <p><b>Thermal Status:</b> {adv_output['temp_status']}</p>
            <p style="font-size: 0.9rem; color: #a5d6a7;">
                Tip: Maintain consistent soil moisture levels during reproductive crop stages to optimize yield formation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_a2:
    st.markdown(
        f"""
        <div class="agri-card">
            <h4 style="color: #a7f3d0;">🧪 Soil Health & Nutrient Balance</h4>
            <p><b>Soil Match:</b> Selected <code>{adv_output['soil_type']}</code> (Standard: <code>{adv_output['ideal_soil']}</code>).</p>
            <p><b>N-P-K Evaluation:</b> {adv_output['npk_advice']}</p>
            <p><b>Optimal Target:</b> <code>{adv_output['recommended_npk']}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PEST, DISEASE & SUSTAINABILITY PRACTICES
# -----------------------------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("### 🛡️ Sustainable Crop Protection & Best Practices")

bp_c1, bp_c2, bp_c3 = st.columns(3)

with bp_c1:
    st.markdown(
        """
        <div class="agri-card">
            <h4 style="color: #81c784;">🌱 Integrated Pest Mgmt. (IPM)</h4>
            <p style="font-size: 0.9rem; color: #d8eadb;">
                • Adopt biological pest controls and pheromone traps.<br>
                • Minimize synthetic pesticide application to avoid soil biome degradation.<br>
                • Rotate crops every season to break insect pest lifecycles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with bp_c2:
    st.markdown(
        """
        <div class="agri-card">
            <h4 style="color: #81c784;">💧 Precision Water Management</h4>
            <p style="font-size: 0.9rem; color: #d8eadb;">
                • Use drip or furrow irrigation to conserve up to 40% water.<br>
                • Apply organic mulch to reduce surface evaporation.<br>
                • Schedule irrigation early in the morning to avoid thermal shock.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with bp_c3:
    st.markdown(
        """
        <div class="agri-card">
            <h4 style="color: #81c784;">🌾 Soil Organic Carbon (SOC)</h4>
            <p style="font-size: 0.9rem; color: #d8eadb;">
                • Incorporate crop residues and farmyard manure.<br>
                • Plant cover crops (legumes) during fallow periods to fix nitrogen.<br>
                • Perform periodic soil testing every 2 cropping cycles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🌿 AgroYield AI Smart Farm Advisory System | Precision Agriculture Decision Support")
