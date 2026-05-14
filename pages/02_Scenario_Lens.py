from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS, SCENARIO_WEIGHTS


st.set_page_config(page_title="Scenario Lens", layout="wide")
recommendations = pd.read_parquet(ARTIFACTS.recommendations)

st.title("Scenario Lens")
scenario = st.selectbox("Scenario", list(SCENARIO_WEIGHTS))
frame = recommendations[recommendations["scenario"] == scenario].copy()

st.dataframe(
    frame[
        [
            "selected_rank",
            "h3_index",
            "recommended_atm_type",
            "marginal_objective",
            "cumulative_objective",
            "objective_breakdown",
        ]
    ].round(3),
    use_container_width=True,
)

fig = px.scatter(
    frame,
    x="coverage_core_score",
    y="profit_core_score",
    size="social_core_score",
    color="recommended_atm_type",
    hover_name="h3_index",
)
fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
