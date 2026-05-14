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


st.set_page_config(page_title="Balance Logic", layout="wide")

recommendations = pd.read_parquet(ARTIFACTS.recommendations)
summary = json.loads(ARTIFACTS.summary.read_text(encoding="utf-8"))

st.title("Balance Logic")
st.write(
    "Это главная методологическая страница про задачу баланса: почему нельзя просто брать `top-K` клеток, "
    "как появляется оптимальное число новых ATM и зачем нужен штраф за каннибализацию."
)

with st.expander("Формулировка задачи баланса", expanded=True):
    st.markdown(
        """
        Если максимизировать только score ячейки, решение быстро становится нереалистичным:

        - модель начнет ставить слишком много новых точек;
        - сильные клетки будут скапливаться рядом;
        - часть новых ATM будет отбирать трафик у уже существующих банкоматов ВТБ.

        Поэтому проект решает задачу в два уровня:

        1. `Cell scoring` — насколько сильна конкретная H3-ячейка.
        2. `Portfolio optimization` — какие ячейки выбрать вместе, чтобы сеть оставалась прибыльной и не перегретой.
        """
    )

with st.expander("Как устроена целевая функция", expanded=True):
    st.latex(
        r"""
        Objective = w_p Profit + w_c Coverage + w_s Social + w_k Competitor + w_b Business
        - w_{can} Cannibalization - w_r Risk - Penalty(N)
        """
    )
    st.markdown(
        """
        Где:

        - `Profit` — profitability proxy;
        - `Coverage` — закрытие пробелов сети;
        - `Social` — полезность для соцсценариев;
        - `Competitor` — эффект конкурентного давления;
        - `Business` — релевантность для cash-in / малого бизнеса;
        - `Cannibalization` — штраф за близость к существующей сети и уже выбранным новым точкам;
        - `Penalty(N)` — штраф за разрастание сети, создающий внутренний максимум по числу ATM.
        """
    )

summary_frame = pd.DataFrame(
    [
        {
            "scenario": scenario,
            "optimal_n": payload["optimal_n"],
            "best_cumulative_objective": payload["best_cumulative_objective"],
            "top_cell": payload["top_cell"],
        }
        for scenario, payload in summary.items()
    ]
).sort_values("best_cumulative_objective", ascending=False)

cols = st.columns([1.0, 1.1])
with cols[0]:
    st.subheader("Оптимум по сценариям")
    st.dataframe(summary_frame.round(3), use_container_width=True)

with cols[1]:
    st.subheader("Сравнение optimal N")
    fig = px.bar(
        summary_frame.sort_values("optimal_n", ascending=False),
        x="scenario",
        y="optimal_n",
        color="scenario",
        color_discrete_sequence=["#0f2747", "#ff8c42", "#6a8ba8", "#8ea7bf", "#d9c1ab", "#b3c7d8"],
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

scenario = st.selectbox("Разобрать кривую сценария", summary_frame["scenario"].tolist())
curve = recommendations[recommendations["scenario"] == scenario].sort_values("selected_rank")

fig_curve = px.line(
    curve,
    x="selected_rank",
    y="cumulative_objective",
    markers=True,
    color_discrete_sequence=["#0f2747"],
)
fig_curve.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_curve, use_container_width=True)

st.markdown(
    """
    Итог: задача баланса решается не на уровне отдельной точки, а на уровне всей последовательности добавления ATM в сеть.
    Именно поэтому в проекте есть и модель, и optimizer, а не просто рейтинг ячеек.
    """
)
