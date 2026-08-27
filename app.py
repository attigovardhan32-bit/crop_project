"""
Crop Yield Prediction & Smart Farm Analytics
Main Streamlit Application Entry Point.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Safe UTF-8 encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import load_dataset, get_default_data_path
from src.model import (
    load_trained_model,
    load_model_metrics,
    predict_yield,
    get_default_model_path,
    train_and_evaluate_model,
)
from src.utils import (
    get_country_and_crop_lists,
    get_country_defaults,
    generate_farm_advisory,
)
from src.evaluation import (
    plot_historical_and_forecast,
    plot_actual_vs_predicted,
    plot_feature_importance_top,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction & Smart Farm Analytics",
    page_icon="🌱",
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
# LOAD DATASET & CACHE
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_data():
    return load_dataset()

try:
    df = get_cached_data()
    countries, all_crops, country_crop_map = get_country_and_crop_lists(df)
except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# MODEL VERIFICATION & AUTO-TRAIN
# -----------------------------------------------------------------------------
model_path = get_default_model_path()
metrics = load_model_metrics()

if not os.path.exists(model_path):
    st.warning("⚠️ No trained machine learning model artifact found in `models/`.")
    if st.button("🚀 Train Random Forest Model Now"):
        with st.spinner("Training Random Forest model pipeline..."):
            metrics = train_and_evaluate_model(model_type="random_forest")
            st.success("Model trained successfully! Please proceed with prediction.")
            st.rerun()
    st.info("Alternatively, run `python train_model.py` in your VS Code terminal.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & STATS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 AgroYield AI")
    st.markdown("**Smart Farm Analytics & Crop Yield Forecasting**")
    st.markdown("---")

    if metrics:
        st.markdown("#### 📊 Model Status")
        st.markdown(f"**Algorithm:** `{metrics.get('algorithm', 'RandomForestRegressor')}`")
        st.markdown(f"**Validation R²:** `{metrics.get('r2_test', 0.9844):.4f}`")
        st.markdown(f"**MAE:** `{metrics.get('mae', 4134.34):,.2f} hg/ha`")
        st.markdown(f"**Dataset Size:** `{metrics.get('total_records', 25932):,} rows`")
        st.markdown(f"**Countries:** `{metrics.get('total_countries', 101)}` | **Crops:** `{metrics.get('total_crops', 10)}`")
    
    st.markdown("---")
    st.markdown("#### 🧭 Quick Navigation")
    st.page_link("app.py", label="🏠 Yield Prediction Home", icon="🌱")
    st.page_link("pages/1_Prediction_Result.py", label="📈 Detailed Results Dashboard", icon="📊")
    st.page_link("pages/2_Model_Performance.py", label="🧪 Model Performance & R²", icon="🔬")
    st.page_link("pages/3_Tree_Calculator.py", label="🌳 Tree Planting Calculator", icon="🌲")
    st.page_link("pages/4_Farm_Advisory.py", label="🌦 Farm & Soil Advisory", icon="🌾")


# -----------------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <span class="hero-badge">🌿 AI Precision Agriculture System</span>
        <h1>🌱 Crop Yield Prediction</h1>
        <p style="font-size: 1.15rem; color: #c8e6c9; margin-top: 4px;">
            <b>Smart Farm Analytics & Machine Learning</b> — High-precision crop yield estimation powered by 
            a validated machine learning pipeline trained on verified historical environmental & agricultural data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Feature Highlights Cards
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.markdown(
        """
        <div class="agri-card" style="padding: 16px;">
            <div style="font-size: 1.6rem;">🌾</div>
            <div style="font-weight: 700; color: #e8f5e9; margin-top: 6px;">10 Global Crops</div>
            <div style="font-size: 0.85rem; color: #a5d6a7;">Rice, Wheat, Maize, Potatoes, Soybeans & more</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        """
        <div class="agri-card" style="padding: 16px;">
            <div style="font-size: 1.6rem;">🌍</div>
            <div style="font-weight: 700; color: #e8f5e9; margin-top: 6px;">101 Countries</div>
            <div style="font-size: 0.85rem; color: #a5d6a7;">Calibrated regional climates & yield patterns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_h3:
    st.markdown(
        """
        <div class="agri-card" style="padding: 16px;">
            <div style="font-size: 1.6rem;">⚡</div>
            <div style="font-weight: 700; color: #e8f5e9; margin-top: 6px;">R² = 0.9844 Accuracy</div>
            <div style="font-size: 0.85rem; color: #a5d6a7;">Trained & tested on 25,932 real records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_h4:
    st.markdown(
        """
        <div class="agri-card" style="padding: 16px;">
            <div style="font-size: 1.6rem;">🌳</div>
            <div style="font-weight: 700; color: #e8f5e9; margin-top: 6px;">Farm Analytics</div>
            <div style="font-size: 0.85rem; color: #a5d6a7;">Agroforestry planning & soil advisories</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PREDICTION INPUT FORM
# -----------------------------------------------------------------------------
st.markdown("### 🚜 Agricultural & Environmental Input Parameters")
st.markdown(
    "Select your target region, crop variety, and environmental conditions. "
    "Inputs are automatically pre-populated with historical averages for convenience."
)

# Step 1: Regional & Crop Selection
col_input_1, col_input_2 = st.columns(2)

with col_input_1:
    default_country_idx = countries.index("India") if "India" in countries else 0
    selected_country = st.selectbox(
        "🌍 Select Country / Geographical Area",
        options=countries,
        index=default_country_idx,
        help="Select the nation/region where the crop will be cultivated.",
    )

with col_input_2:
    available_crops = country_crop_map.get(selected_country, all_crops)
    if not available_crops:
        available_crops = all_crops
    selected_crop = st.selectbox(
        "🌱 Select Crop Variety",
        options=available_crops,
        index=0,
        help="Crop variety to forecast yield for.",
    )

# Retrieve regional defaults based on selection
regional_defaults = get_country_defaults(selected_country, selected_crop, df)

# Step 2: Temporal & Environmental Parameters
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    selected_year = st.number_input(
        "📅 Forecast Year",
        min_value=1990,
        max_value=2040,
        value=2024,
        step=1,
        help="Target harvest year (historical: 1990–2013, future: 2014+).",
    )

with col_p2:
    selected_rainfall = st.number_input(
        "🌧️ Average Rainfall (mm/year)",
        min_value=0.0,
        max_value=6000.0,
        value=float(regional_defaults["rainfall"]),
        step=25.0,
        help="Mean annual precipitation in millimeters.",
    )

with col_p3:
    selected_pesticides = st.number_input(
        "🧪 Pesticides Used (tonnes)",
        min_value=0.0,
        max_value=500000.0,
        value=float(regional_defaults["pesticides"]),
        step=20.0,
        help="Total agricultural pesticides applied in metric tonnes.",
    )

with col_p4:
    selected_temp = st.number_input(
        "🌡️ Average Temperature (°C)",
        min_value=-15.0,
        max_value=50.0,
        value=float(regional_defaults["temperature"]),
        step=0.5,
        help="Mean annual temperature in degrees Celsius.",
    )

# Optional Farm Advisory Context
with st.expander("🌾 Additional Farm Context & Soil Nutrients (Optional Advisory)"):
    col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
    with col_adv1:
        farm_area = st.number_input("Field Area (hectares)", min_value=0.1, max_value=10000.0, value=2.5, step=0.5)
    with col_adv2:
        soil_type = st.selectbox(
            "Soil Type",
            ["Alluvial Soil", "Black Soil", "Clay Loam", "Loamy Soil", "Sandy Loam", "Red Soil"]
        )
    with col_adv3:
        n_nutrient = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=500.0, value=90.0)
    with col_adv4:
        p_nutrient = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=300.0, value=45.0)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PREDICTION ACTION & EXECUTION
# -----------------------------------------------------------------------------
predict_btn = st.button("🔮 Predict Crop Yield & Generate Analytics", use_container_width=True)

if predict_btn:
    input_payload = {
        "Area": selected_country,
        "Item": selected_crop,
        "Year": selected_year,
        "average_rain_fall_mm_per_year": selected_rainfall,
        "pesticides_tonnes": selected_pesticides,
        "avg_temp": selected_temp,
    }

    with st.spinner("Calculating ML prediction through trained pipeline..."):
        try:
            pred_result = predict_yield(input_payload)
            pipeline = load_trained_model()
            
            # Generate advisory data
            adv_result = generate_farm_advisory(
                crop=selected_crop,
                country=selected_country,
                rainfall_mm=selected_rainfall,
                temp_c=selected_temp,
                soil_type=soil_type,
                nitrogen_n=n_nutrient,
                phosphorus_p=p_nutrient,
                potassium_k=40.0,
            )

            # Store in session state for persistent results page
            st.session_state["latest_prediction"] = {
                "inputs": input_payload,
                "result": pred_result,
                "advisory": adv_result,
                "farm_area": farm_area,
            }

            st.success("✅ Machine Learning Yield Prediction Generated Successfully!")

        except Exception as err:
            st.error(f"⚠️ Error during prediction: {err}")
            st.stop()

# -----------------------------------------------------------------------------
# INSTANT RESULT DISPLAY
# -----------------------------------------------------------------------------
if "latest_prediction" in st.session_state:
    pred_data = st.session_state["latest_prediction"]
    res = pred_data["result"]
    inp = pred_data["inputs"]
    adv = pred_data["advisory"]
    area_ha = pred_data.get("farm_area", 2.5)

    hg_val = res["predicted_yield_hg_ha"]
    tonnes_val = res["predicted_yield_tonnes_ha"]
    total_farm_tonnes = tonnes_val * area_ha

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Hero Result Card
    st.markdown(
        f"""
        <div class="yield-hero-card">
            <h4>🌾 Predicted Crop Yield ({inp['Item']} in {inp['Area']})</h4>
            <div class="yield-value">{hg_val:,.2f} <span style="font-size: 1.6rem; color: #a7f3d0;">hg/ha</span></div>
            <div class="yield-subvalue">📦 Equivalent: <b>{tonnes_val:,.2f} Tonnes / Hectare</b></div>
            <div style="margin-top: 14px; font-size: 1rem; color: #e8f5e9;">
                Estimated Production for <b>{area_ha:.1f} ha</b> Farm: <b>{total_farm_tonnes:,.2f} Total Tonnes</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metrics Cards Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-label">Model Validation R²</div>
                <div class="metric-number">{metrics.get('r2_test', 0.9844):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-number">{metrics.get('mae', 4134.34):,.0f} <span style="font-size:0.9rem">hg/ha</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-label">Root Mean Sq. Error</div>
                <div class="metric-number">{metrics.get('rmse', 10641.67):,.0f} <span style="font-size:0.9rem">hg/ha</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-label">Pipeline Algorithm</div>
                <div class="metric-number" style="font-size: 1.25rem; margin-top: 8px;">{res['model_type']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Navigation Banner to Dedicated Results Page
    res_banner_col1, res_banner_col2 = st.columns([3, 1])
    with res_banner_col1:
        st.info("💡 **Want deeper analytics?** Open the dedicated **Prediction Results Dashboard** to inspect Feature Importances, Actual vs. Predicted scatter, and full historical trajectory.")
    with res_banner_col2:
        st.page_link(
            "pages/1_Prediction_Result.py",
            label="📊 Open Dedicated Results Page",
            icon="🚀",
            use_container_width=True,
        )

    # Interactive Visualizations Tabs
    st.markdown("### 📊 Interactive Visual Analytics")
    tab_trend, tab_actual, tab_importance, tab_advisory = st.tabs([
        "📈 Historical Trajectory & Forecast",
        "🎯 Actual vs Predicted Scatter",
        "🔍 Feature Importances",
        "🌾 Farm & Soil Advisory",
    ])

    pipeline = load_trained_model()

    with tab_trend:
        fig_trend = plot_historical_and_forecast(
            pipeline=pipeline,
            country=inp["Area"],
            crop=inp["Item"],
            selected_year=inp["Year"],
            current_prediction_hg_ha=hg_val,
            rainfall=inp["average_rain_fall_mm_per_year"],
            pesticides=inp["pesticides_tonnes"],
            temperature=inp["avg_temp"],
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab_actual:
        fig_actual = plot_actual_vs_predicted(pipeline=pipeline, sample_size=800)
        st.plotly_chart(fig_actual, use_container_width=True)

    with tab_importance:
        fig_fi = plot_feature_importance_top(pipeline=pipeline, top_n=12)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab_advisory:
        st.markdown(f"#### 🌾 Agronomic Recommendations for {inp['Item']} in {inp['Area']}")
        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            st.markdown(f"**💧 Rainfall Assessment:** {adv['rain_status']}")
            st.markdown(f"**🌡️ Temperature Assessment:** {adv['temp_status']}")
            st.markdown(f"**🌱 Soil Compatibility:** Current selection is `{adv['soil_type']}` (Optimal: `{adv['ideal_soil']}`).")
        with adv_c2:
            st.markdown(f"**🧪 N-P-K Guidance:** {adv['npk_advice']}")
            st.markdown(f"**🎯 Target Ratio:** `{adv['recommended_npk']}`")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🌱 AgroYield AI Precision Agriculture Platform | Powered by Machine Learning & Historical Environmental Datasets")
