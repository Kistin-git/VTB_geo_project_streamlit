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


def feature_family(feature: str) -> str:
    if feature.startswith("mcc_") or feature.startswith("time_"):
        return "transaction profile"
    if "_ring" in feature:
        return "spatial lag"
    if feature.startswith("dist_"):
        return "distance"
    if "core_score" in feature or "intensity" in feature or "pressure" in feature:
        return "engineered score"
    if feature in {"tx_sum", "tx_count", "unique_customers", "avg_ticket"}:
        return "base transactions"
    return "other"


st.set_page_config(page_title="Model Explorer", layout="wide")

metrics = json.loads(ARTIFACTS.metrics.read_text(encoding="utf-8"))
importance = pd.read_csv(ARTIFACTS.feature_importance)
importance["family"] = importance["feature"].map(feature_family)

st.title("Model Explorer")
st.write(
    "Эта страница объясняет, как обучалась модель, почему выбран именно такой стек "
    "и как мы проверяем, что решение обобщается не только на случайный train/test split, но и на новые зоны города."
)

hero_cols = st.columns(4)
hero_cols[0].metric("Spatial CV AUC", f"{metrics['mean']['catboost_auc']:.4f}")
hero_cols[1].metric("Spatial CV AP", f"{metrics['mean']['catboost_ap']:.4f}")
hero_cols[2].metric("Spatial CV NDCG@K", f"{metrics['mean']['catboost_reg_ndcg']:.4f}")
hero_cols[3].metric("Baseline AUC", f"{metrics['mean']['baseline_auc']:.4f}")

with st.expander("Как устроен модельный слой", expanded=True):
    st.markdown(
        """
        1. `Baseline classification`: логистическая регрессия проверяет, что задача действительно не сводится к линейным зависимостям.
        2. `Baseline demand`: простая регрессия по `log1p(atm_users_count)` дает минимальную планку качества для demand-предсказания.
        3. `CatBoostClassifier`: оценивает вероятность ATM-присутствия в ячейке.
        4. `CatBoostRegressor`: оценивает интенсивность ATM-спроса в ячейке.
        5. `Composite score`: объединяет классификационную вероятность и регрессионную demand-оценку.

        Такая схема выбрана, потому что бизнесу нужно не только понять, есть ли у зоны шанс, но и ранжировать зоны по силе спроса.
        """
    )

with st.expander("Почему не только random split", expanded=True):
    st.markdown(
        """
        Для геоаналитики обычный случайный split опасен: соседние ячейки часто похожи, и модель может переоценить себя из-за spatial leakage.

        Поэтому мы используем `spatial cross-validation`:

        - разбиваем Москву на несколько пространственных блоков;
        - учим модель на части блоков;
        - проверяем ее на остальных;
        - усредняем результат по fold-ам.

        Это гораздо ближе к реальному сценарию внедрения, где банк переносит решение на новую территорию, а не на почти соседнюю клетку.
        """
    )

folds = pd.DataFrame(metrics["folds"])
st.subheader("Метрики по spatial fold-ам")
st.dataframe(folds.round(4), use_container_width=True)

chart_cols = st.columns([1.1, 1.0])
with chart_cols[0]:
    st.subheader("Top Features")
    fig = px.bar(
        importance.head(18).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#ff8c42", "#0f2747"],
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with chart_cols[1]:
    st.subheader("Feature Families")
    grouped = importance.groupby("family", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    fig_grouped = px.pie(grouped, names="family", values="importance", hole=0.45, color_discrete_sequence=["#0f2747", "#ff8c42", "#6a8ba8", "#a6bacd", "#d9c1ab"])
    fig_grouped.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_grouped, use_container_width=True)

st.subheader("Как читать метрики")
st.markdown(
    """
    - `AUC` показывает, насколько хорошо модель различает ATM-positive и ATM-negative ячейки.
    - `AP` особенно важен из-за дисбаланса: положительных клеток существенно меньше, чем отрицательных.
    - `NDCG@K` проверяет качество ранжирования сильнейших зон в топе списка.
    - `RMSE/MAE` по demand показывают, насколько стабильно модель оценивает интенсивность ATM-спроса.

    Важная идея: победитель выбирается не по одному числу, а по связке `spatial generalization + ranking quality + interpretability`.
    """
)
