from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS


st.set_page_config(page_title="Data Report", layout="wide")

features = pd.read_parquet(ARTIFACTS.features)
external = pd.read_csv(ARTIFACTS.external_summary)

st.title("Data Report")
st.write("Сводка по внутренним и внешним данным, использованным в пайплайне.")

col1, col2, col3 = st.columns(3)
col1.metric("H3 Cells", int(features["h3_index"].nunique()))
col2.metric("ATM-positive Cells", int(features["atm_presence"].sum()))
col3.metric("Feature Count", features.shape[1])

st.subheader("External Layers")
st.dataframe(external, use_container_width=True)

st.subheader("Feature Sample")
st.dataframe(features.head(25), use_container_width=True)
