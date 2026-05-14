from __future__ import annotations

import json
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
summary = json.loads(ARTIFACTS.summary.read_text(encoding="utf-8"))

st.title("Scenario Lens")
st.write(
    "Здесь видно, как меняется решение при смене бизнес-цели. В этой задаче нет одной универсальной точки: "
    "режим `profit` и режим `social` по определению толкают optimizer в разные части города."
)

scenario = st.selectbox("Scenario", list(SCENARIO_WEIGHTS))
frame = recommendations[recommendations["scenario"] == scenario].copy().sort_values("selected_rank")
weights = pd.DataFrame(
    [{"component": key, "weight": value} for key, value in SCENARIO_WEIGHTS[scenario].items()]
).sort_values("weight", ascending=False)

hero_cols = st.columns(3)
hero_cols[0].metric("Optimal N", summary[scenario]["optimal_n"])
hero_cols[1].metric("Best Objective", f"{summary[scenario]['best_cumulative_objective']:.3f}")
hero_cols[2].metric("Top Cell", summary[scenario]["top_cell"])

with st.expander("Как scenario engine решает задачу баланса", expanded=True):
    st.markdown(
        """
        Для каждой candidate-cell optimizer считает:

        - profit contribution;
        - coverage gain;
        - social value;
        - competitor effect;
        - business relevance;
        - cannibalization penalty;
        - risk penalty.

        Затем веса этих компонентов меняются по сценарию. Поэтому у одной и той же клетки итоговая полезность может быть высокой в `profit`,
        но средней в `social`, и наоборот.
        """
    )

cols = st.columns([1.05, 1.0])
with cols[0]:
    st.subheader("Scenario Weights")
    st.dataframe(weights.round(3), use_container_width=True)

with cols[1]:
    st.subheader("Coverage vs Profit Trade-off")
    fig = px.scatter(
        frame,
        x="coverage_core_score",
        y="profit_core_score",
        size="social_core_score",
        color="recommended_atm_type",
        hover_name="h3_index",
        hover_data={"selected_rank": True, "marginal_objective": ":.3f"},
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Scenario Shortlist")
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

curve = px.line(
    frame,
    x="selected_rank",
    y="cumulative_objective",
    markers=True,
    color_discrete_sequence=["#0f2747"],
)
curve.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(curve, use_container_width=True)
