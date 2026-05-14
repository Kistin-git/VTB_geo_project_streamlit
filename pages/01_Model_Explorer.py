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

from vtb_geo_project.config import ARTIFACTS


st.set_page_config(page_title="Model Explorer", layout="wide")

metrics = json.loads(ARTIFACTS.metrics.read_text(encoding="utf-8"))
importance = pd.read_csv(ARTIFACTS.feature_importance)

st.title("Model Explorer")
st.write("Spatial CV и интерпретация основной модели.")

col1, col2, col3 = st.columns(3)
col1.metric("Spatial CV AUC", f"{metrics['mean']['catboost_auc']:.4f}")
col2.metric("Spatial CV AP", f"{metrics['mean']['catboost_ap']:.4f}")
col3.metric("Spatial CV NDCG@K", f"{metrics['mean']['catboost_reg_ndcg']:.4f}")

folds = pd.DataFrame(metrics["folds"])
st.subheader("Metrics by Fold")
st.dataframe(folds.round(4), use_container_width=True)

st.subheader("Top Features")
fig = px.bar(
    importance.head(15).sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale=["#ff8c42", "#0f2747"],
)
fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
