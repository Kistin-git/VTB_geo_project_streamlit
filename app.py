from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS, SCENARIO_WEIGHTS


st.set_page_config(page_title="VTB Geo Analytics", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
      .stApp { background:
        radial-gradient(circle at top left, rgba(255,140,66,0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(17,48,92,0.18), transparent 28%),
        linear-gradient(180deg, #f7f3ed 0%, #f0ebe2 100%);
      }
      html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
      .mono { font-family: 'IBM Plex Mono', monospace; }
      .hero { padding: 1.2rem 1.4rem; border-radius: 20px; background: #0f2747; color: #f7f3ed; }
      .hero h1 { margin: 0; font-size: 2.2rem; }
      .hero p { margin-top: 0.4rem; color: #d7e1ef; }
      .metric-card { padding: 1rem; border-radius: 18px; background: rgba(255,255,255,0.72); border: 1px solid rgba(15,39,71,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = pd.read_parquet(ARTIFACTS.candidates)
    recommendations = pd.read_parquet(ARTIFACTS.recommendations)
    summary = json.loads(ARTIFACTS.summary.read_text(encoding="utf-8"))
    return features, recommendations, summary


features, recommendations, summary = load_data()
scenario = st.sidebar.selectbox("Scenario", list(SCENARIO_WEIGHTS))
top_n = st.sidebar.slider("Show top N recommendations", min_value=5, max_value=40, value=15)

scenario_df = recommendations[recommendations["scenario"] == scenario].sort_values("selected_rank").head(top_n)
optimal_n = summary[scenario]["optimal_n"]

st.markdown(
    """
    <div class="hero">
      <h1>VTB Geo Analytics</h1>
      <p>Баланс между плотностью сети, profitability proxy, покрытием и социальной доступностью на H3-сетке Москвы.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scenario", scenario)
col2.metric("Optimal New ATM", optimal_n)
col3.metric("Top Candidate", summary[scenario]["top_cell"])
col4.metric("Best Objective", f"{summary[scenario]['best_cumulative_objective']:.3f}")

layer = pdk.Layer(
    "H3HexagonLayer",
    data=features,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="h3_index",
    get_fill_color="[255 * composite_model_score, 80, 255 - 180 * composite_model_score, 180]",
    get_line_color=[15, 39, 71, 70],
    line_width_min_pixels=0.5,
)

recommendation_layer = pdk.Layer(
    "ScatterplotLayer",
    data=scenario_df,
    get_position="[lon, lat]",
    get_fill_color="[15, 39, 71, 220]",
    get_radius=180,
    pickable=True,
)

view_state = pdk.ViewState(latitude=55.75, longitude=37.62, zoom=9.4, pitch=0)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer, recommendation_layer],
        initial_view_state=view_state,
        map_provider="carto",
        tooltip={
            "html": "<b>Cell:</b> {h3_index}<br/><b>Composite score:</b> {composite_model_score}<br/><b>Recommended ATM:</b> {recommended_atm_type}<br/><b>Marginal objective:</b> {marginal_objective}",
            "style": {"backgroundColor": "#0f2747", "color": "white"},
        },
        map_style=pdk.map_styles.LIGHT,
    )
)

left, right = st.columns([1.2, 1.0])
with left:
    st.subheader("Top Recommendations")
    st.dataframe(
        scenario_df[
            [
                "selected_rank",
                "h3_index",
                "recommended_atm_type",
                "marginal_objective",
                "cumulative_objective",
                "profit_core_score",
                "coverage_core_score",
                "social_core_score",
            ]
        ].round(3),
        use_container_width=True,
    )

with right:
    st.subheader("Objective Curve")
    curve = recommendations[recommendations["scenario"] == scenario].sort_values("selected_rank")
    fig = px.line(
        curve,
        x="selected_rank",
        y="cumulative_objective",
        markers=True,
        color_discrete_sequence=["#0f2747"],
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Подробные страницы со слоями данных, моделями и метриками находятся в разделе pages/ .")
