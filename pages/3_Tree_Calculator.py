"""
Agroforestry & Tree Planting Land Capacity Calculator.
Estimates field tree planting capacity, spacing density, boundary tree buffers,
and annual carbon sequestration benefits.
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

from src.utils import calculate_tree_planting_capacity

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tree Planting Calculator | AgroYield AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(ROOT_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 AgroYield AI")
    st.page_link("app.py", label="🏠 Yield Prediction Home", icon="🌱")
    st.page_link("pages/1_Prediction_Result.py", label="🌾 Results Dashboard", icon="📈")
    st.page_link("pages/2_Model_Performance.py", label="🧪 Model Performance", icon="🔬")
    st.page_link("pages/4_Farm_Advisory.py", label="🌾 Farm Advisory", icon="🌦")
    st.markdown("---")
    st.markdown("#### 🌳 Agroforestry Tips")
    st.info("Intercropping timber/fruit trees with field crops creates microclimate buffers, reduces evaporation, and sequesters atmospheric carbon.")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <span class="hero-badge">🌲 Agroforestry Planning & Carbon Modeling</span>
        <h1>🌳 Tree Planting & Land Capacity Calculator</h1>
        <p style="font-size: 1.15rem; color: #c8e6c9;">
            Calculate optimal tree capacity, row grid spacing, boundary windbreak buffers, 
            and annual carbon sequestration based on your exact land dimensions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INPUT CONTROLS
# -----------------------------------------------------------------------------
st.markdown("### 📐 Land & Spacing Configuration")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    field_area_val = st.number_input(
        "🌾 Total Field Area",
        min_value=0.1,
        max_value=100000.0,
        value=2.0,
        step=0.5,
        help="Total land area available for planting or agroforestry.",
    )
    area_unit_sel = st.selectbox(
        "📏 Land Area Unit",
        options=["Hectares", "Acres", "Square Meters (m²)"],
        index=0,
    )

with col_in2:
    row_spacing = st.number_input(
        "↔️ Row Spacing (meters)",
        min_value=0.5,
        max_value=30.0,
        value=4.0,
        step=0.5,
        help="Distance between consecutive tree rows.",
    )
    tree_spacing = st.number_input(
        "↕️ Tree Spacing within Row (meters)",
        min_value=0.5,
        max_value=30.0,
        value=3.0,
        step=0.5,
        help="Distance between trees within the same row.",
    )

with col_in3:
    usable_pct = st.slider(
        "🚜 Usable Net Land Percentage (%)",
        min_value=50,
        max_value=100,
        value=90,
        step=1,
        help="Portion of land allocated for planting after subtracting access roads, irrigation paths, and homesteads.",
    )
    tree_type = st.selectbox(
        "🌱 Agroforestry System Type",
        options=[
            "Fruit Orchard (Mango, Citrus, Apple, Guava)",
            "Timber & Agroforestry (Teak, Eucalyptus, Mahogany)",
            "Agro-silvopastoral (Fodder & Shade Trees)",
            "Boundary Windbreak & Soil Conservation",
        ],
        index=0,
    )

# -----------------------------------------------------------------------------
# REAL COMPUTATION
# -----------------------------------------------------------------------------
calc_result = calculate_tree_planting_capacity(
    area_value=field_area_val,
    area_unit=area_unit_sel,
    row_spacing_m=row_spacing,
    tree_spacing_m=tree_spacing,
    usable_percentage=float(usable_pct),
    tree_species=tree_type,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESULT DISPLAY CARDS
# -----------------------------------------------------------------------------
st.markdown("### 🌾 Calculated Planting Capacity & Eco-Metrics")

res_c1, res_c2, res_c3, res_c4 = st.columns(4)

with res_c1:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Estimated Tree Capacity</div>
            <div class="metric-number" style="color: #00e676;">{calc_result['estimated_tree_count']:,}</div>
            <div style="font-size: 0.75rem; color: #a5d6a7; margin-top: 4px;">Trees on {calc_result['input_area']} {calc_result['area_unit']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_c2:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Tree Density / Hectare</div>
            <div class="metric-number">{calc_result['tree_density_per_ha']:,.0f}</div>
            <div style="font-size: 0.75rem; color: #a5d6a7; margin-top: 4px;">{calc_result['area_per_tree_m2']:.1f} m² per tree</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_c3:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Boundary Buffer Trees</div>
            <div class="metric-number">{calc_result['boundary_tree_count']:,}</div>
            <div style="font-size: 0.75rem; color: #a5d6a7; margin-top: 4px;">Perimeter Windbreaks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_c4:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="metric-label">Annual Carbon Offset</div>
            <div class="metric-number">{calc_result['annual_co2_tonnes']:.2f} <span style="font-size: 0.85rem">t CO₂/yr</span></div>
            <div style="font-size: 0.75rem; color: #a5d6a7; margin-top: 4px;">~{calc_result['annual_co2_kg']:,.0f} kg CO₂/yr</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VISUAL BREAKDOWN & GAUGES
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    fig_land = go.Figure(
        go.Pie(
            labels=["Net Usable Planting Area (m²)", "Non-Planting Buffers / Roads (m²)"],
            values=[calc_result["usable_area_m2"], calc_result["total_area_m2"] - calc_result["usable_area_m2"]],
            hole=0.55,
            marker_colors=["#2e7d32", "#78909c"],
        )
    )
    fig_land.update_layout(
        title="<b>Land Area Allocation Breakdown</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(13, 59, 37, 0.4)",
        plot_bgcolor="rgba(8, 43, 27, 0.6)",
        font=dict(color="#d8eadb"),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_land, use_container_width=True)

with col_chart2:
    st.markdown(
        f"""
        <div class="agri-card" style="height: 350px;">
            <h4 style="color: #a7f3d0;">🌳 Mathematical Calculation Breakdown</h4>
            <table style="width: 100%; color: #e8f5e9; font-size: 0.95rem;">
                <tr><td style="padding: 6px 0;"><b>Total Converted Land:</b></td><td style="text-align: right;">{calc_result['total_area_m2']:,.1f} m²</td></tr>
                <tr><td style="padding: 6px 0;"><b>Usable Allocated Area ({calc_result['usable_percentage']}%):</b></td><td style="text-align: right;">{calc_result['usable_area_m2']:,.1f} m²</td></tr>
                <tr><td style="padding: 6px 0;"><b>Grid Allocation per Tree:</b></td><td style="text-align: right;">{calc_result['row_spacing_m']}m × {calc_result['tree_spacing_m']}m = {calc_result['area_per_tree_m2']} m²</td></tr>
                <tr><td style="padding: 6px 0;"><b>Tree Formula:</b></td><td style="text-align: right;"><code>Usable Area ÷ Grid Area</code></td></tr>
                <tr><td style="padding: 6px 0;"><b>Calculated Tree Count:</b></td><td style="text-align: right; color: #00e676;"><b>{calc_result['estimated_tree_count']:,} Trees</b></td></tr>
                <tr><td style="padding: 6px 0;"><b>CO₂ Sequestration Rate:</b></td><td style="text-align: right;">~21.8 kg CO₂/tree/year</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### 💡 Field Implementation Recommendations")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown(
        """
        <div class="agri-card">
            <h4 style="color: #81c784;">🌱 Spacing & Canopy Management</h4>
            <p>• <b>Intercropping Space:</b> For intercropping with cash crops (e.g. Soybeans, Maize, Pulses), maintain at least 4.0m row spacing to allow tractor or manual equipment movement.</p>
            <p>• <b>Sunlight Penetration:</b> Orient tree rows North-to-South to equalize sunlight distribution across seasonal crop rows throughout the day.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f_col2:
    st.markdown(
        """
        <div class="agri-card">
            <h4 style="color: #81c784;">💧 Irrigation & Root Architecture</h4>
            <p>• <b>Drip Line Planning:</b> Pair young tree saplings with dedicated sub-surface drip irrigation to encourage deep taproot development and prevent root competition with shallow-rooted crops.</p>
            <p>• <b>Perimeter Windbreaks:</b> Plant denser trees along prevailing wind boundaries to reduce wind erosion and crop lodging.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
