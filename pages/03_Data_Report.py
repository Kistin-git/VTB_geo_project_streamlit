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

from vtb_geo_project.config import ARTIFACTS


def feature_group(feature: str) -> str:
    if feature.startswith("mcc_") or feature.startswith("time_"):
        return "transaction profile"
    if "_ring" in feature:
        return "neighbor / spatial lag"
    if feature.startswith("dist_"):
        return "distance / accessibility"
    if "core_score" in feature or "intensity" in feature or "pressure" in feature:
        return "engineered score"
    if feature in {"tx_sum", "tx_count", "unique_customers", "avg_ticket"}:
        return "core transactions"
    return "other"


st.set_page_config(page_title="Data Report", layout="wide")

features = pd.read_parquet(ARTIFACTS.features)
external = pd.read_csv(ARTIFACTS.external_summary)

st.title("Data Report")
st.write(
    "Страница про то, как собирался датасет, из каких блоков он состоит и какие внешние слои усиливают кейсовые данные."
)

hero_cols = st.columns(4)
hero_cols[0].metric("H3 Cells", int(features["h3_index"].nunique()))
hero_cols[1].metric("ATM-positive Cells", int(features["atm_presence"].sum()))
hero_cols[2].metric("Feature Count", features.shape[1])
hero_cols[3].metric("External Layers", external["source"].nunique())

with st.expander("Как строился data pipeline", expanded=True):
    st.markdown(
        """
        1. Из кейсового `data.parquet` строятся cell-level агрегаты по H3.
        2. Из `target.parquet` формируется ATM-сигнал по ячейкам.
        3. Все внутренние признаки агрегируются в одну таблицу.
        4. BBBike / OSM слой Москвы добавляет внешний геоконтекст: метро, школы, вузы, больницы, офисы, торговые точки и присутствие банков.
        5. Для соседних H3-колец считаются spatial lag признаки.
        6. На последнем шаге собираются engineered scores для optimizer.
        """
    )

cols = st.columns([1.05, 1.0])
with cols[0]:
    st.subheader("External Layers")
    st.dataframe(external, use_container_width=True)

with cols[1]:
    st.subheader("External Layer Sizes")
    fig_ext = px.bar(
        external.sort_values("records", ascending=False),
        x="source",
        y="records",
        color="source",
        color_discrete_sequence=["#0f2747", "#ff8c42", "#6a8ba8", "#8ea7bf", "#d9c1ab"],
    )
    fig_ext.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig_ext, use_container_width=True)

feature_groups = pd.Series([feature_group(col) for col in features.columns]).value_counts().rename_axis("group").reset_index(name="count")
st.subheader("Feature Families")
fig_groups = px.bar(feature_groups, x="group", y="count", color="group", color_discrete_sequence=["#0f2747", "#ff8c42", "#6a8ba8", "#8ea7bf", "#d9c1ab"])
fig_groups.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
st.plotly_chart(fig_groups, use_container_width=True)

st.subheader("Feature Sample")
st.dataframe(features.head(25), use_container_width=True)
