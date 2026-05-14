from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import folium
import h3
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS, SCENARIO_WEIGHTS
from vtb_geo_project.utils import haversine_km


SCENARIO_DESCRIPTIONS = {
    "balanced": "Компромисс между profitability proxy, покрытием, социальным эффектом и штрафом за каннибализацию.",
    "profit": "Режим для максимизации бизнес-эффекта: выше вес у денежного потока и ожидаемого ATM-спроса.",
    "coverage": "Режим для закрытия сервисных пустот: приоритет получают зоны далеко от текущего присутствия ВТБ.",
    "social": "Режим социальной миссии: усиливает районы рядом со школами, вузами и медицинскими объектами.",
    "competitor": "Режим конкурентного ответа: ищет клетки, где есть клиентская активность и насыщенность чужими ATM.",
    "business": "Режим для cash-in и малого бизнеса: сильнее оценивает торговые и офисные кластеры.",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(255,140,66,0.16), transparent 34%),
              radial-gradient(circle at bottom right, rgba(17,48,92,0.12), transparent 30%),
              linear-gradient(180deg, #faf6ef 0%, #f0ebe2 100%);
          }
          html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
          .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f2747 0%, #183a63 100%);
            color: #f8f2e8;
            box-shadow: 0 18px 42px rgba(15,39,71,0.18);
          }
          .hero h1 { margin: 0; font-size: 2.25rem; }
          .hero p { margin: 0.45rem 0 0 0; color: #d4dfef; max-width: 980px; }
          .soft-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(15,39,71,0.08);
          }
          .metric-note {
            font-size: 0.88rem;
            color: #4a6079;
            margin-top: -0.3rem;
          }
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fffaf2 0%, #f2ebdf 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, list[str]]]:
    features = pd.read_parquet(ARTIFACTS.candidates)
    recommendations = pd.read_parquet(ARTIFACTS.recommendations)
    summary = json.loads(ARTIFACTS.summary.read_text(encoding="utf-8"))
    explanations_raw = json.loads(ARTIFACTS.explanations.read_text(encoding="utf-8"))
    explanations = {row["h3_index"]: row["reasons"] for row in explanations_raw}
    return features, recommendations, summary, explanations


def cell_boundary(cell: str) -> list[list[float]]:
    return [[float(lat), float(lon)] for lat, lon in h3.cell_to_boundary(cell)]


def focus_subset(frame: pd.DataFrame, center_lat: float, center_lon: float, radius_km: float) -> pd.DataFrame:
    subset = frame.copy()
    subset["focus_distance_km"] = subset.apply(
        lambda row: haversine_km(center_lat, center_lon, row["lat"], row["lon"]),
        axis=1,
    )
    return subset[subset["focus_distance_km"] <= radius_km].copy()


def build_map(
    display_cells: pd.DataFrame,
    scenario_df: pd.DataFrame,
    best_row: pd.Series,
    show_city_overview: bool,
    show_sector_box: bool,
    radius_km: float,
) -> folium.Map:
    zoom_start = 10 if show_city_overview else 13
    map_center = [55.7558, 37.6176] if show_city_overview else [best_row["lat"], best_row["lon"]]
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="CartoDB Voyager",
        control_scale=True,
        prefer_canvas=True,
        max_zoom=17,
    )

    if not show_city_overview:
        lat_pad = radius_km / 110.574
        lon_pad = radius_km / (111.320 * max(0.2, abs(math.cos(math.radians(best_row["lat"])))))
        bounds = [
            [best_row["lat"] - lat_pad, best_row["lon"] - lon_pad],
            [best_row["lat"] + lat_pad, best_row["lon"] + lon_pad],
        ]
        m.fit_bounds(bounds)
        if show_sector_box:
            folium.Rectangle(
                bounds=bounds,
                color="#0f2747",
                weight=2,
                fill=False,
                dash_array="6 6",
                tooltip="Фокусный сектор для разбора сценария",
            ).add_to(m)

    if display_cells.empty:
        return m

    color_scale = LinearColormap(
        colors=["#f1d5bf", "#ff8c42", "#0f2747"],
        vmin=float(display_cells["composite_model_score"].min()),
        vmax=float(display_cells["composite_model_score"].max()),
    )

    for row in display_cells.itertuples():
        folium.Polygon(
            locations=cell_boundary(row.h3_index),
            color="#f5f2eb",
            weight=0.6,
            fill=True,
            fill_opacity=0.58,
            fill_color=color_scale(float(row.composite_model_score)),
            tooltip=(
                f"H3: {row.h3_index}<br>"
                f"Composite score: {row.composite_model_score:.3f}<br>"
                f"Demand score: {row.demand_score:.3f}<br>"
                f"Coverage score: {row.coverage_core_score:.3f}"
            ),
        ).add_to(m)

    for row in scenario_df.itertuples():
        marker_color = "#0f2747" if int(row.selected_rank) <= 3 else "#ff8c42"
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=8 if int(row.selected_rank) <= 3 else 6,
            color=marker_color,
            weight=2,
            fill=True,
            fill_opacity=0.95,
            fill_color=marker_color,
            tooltip=(
                f"Rank {int(row.selected_rank)}<br>"
                f"{row.recommended_atm_type}<br>"
                f"Marginal objective: {row.marginal_objective:.3f}"
            ),
        ).add_to(m)

    return m


def metric_block(label: str, value: str, note: str) -> None:
    st.metric(label, value)
    st.markdown(f"<div class='metric-note'>{note}</div>", unsafe_allow_html=True)


st.set_page_config(page_title="VTB Geo Analytics", page_icon="🏦", layout="wide")
apply_style()

features, recommendations, summary, explanations = load_data()

st.sidebar.title("Управление")
scenario = st.sidebar.selectbox("Режим оптимизации", list(SCENARIO_WEIGHTS))
top_n = st.sidebar.slider("Сколько рекомендованных точек показать", min_value=5, max_value=30, value=12)
map_scope = st.sidebar.radio("Область карты", ["Фокусный сектор", "Вся Москва"], index=0)
focus_radius_km = st.sidebar.slider("Радиус фокусного сектора, км", min_value=2.0, max_value=6.0, value=3.4, step=0.2)

st.sidebar.markdown("### Что делает режим")
st.sidebar.info(SCENARIO_DESCRIPTIONS[scenario])

with st.sidebar.expander("Как пользоваться демо", expanded=True):
    st.markdown(
        """
        1. Выберите режим слева.
        2. Начните с `Фокусного сектора`: там лучше виден один сильный кусок Москвы.
        3. На карте цвет гексагона показывает интегральный потенциал ячейки.
        4. Темные маркеры — точки, которые optimizer рекомендует открыть первыми.
        5. Ниже смотрите, почему именно эти точки попали в shortlist и где находится оптимум по числу новых ATM.
        """
    )

with st.sidebar.expander("Что значат метрики"):
    st.markdown(
        """
        - `Optimal New ATM`: число новых банкоматов, на котором cumulative objective достигает максимума.
        - `Best Objective`: значение целевой функции с учетом прибыли, покрытия, соцэффекта и штрафов.
        - `Coverage score`: насколько зона закрывает пробел в текущем присутствии ВТБ.
        - `Marginal objective`: прирост целевой функции от добавления еще одной точки в сеть.
        """
    )

scenario_all = recommendations[recommendations["scenario"] == scenario].sort_values("selected_rank").copy()
scenario_df = scenario_all.head(top_n).copy()
best_row = scenario_all.iloc[0]
optimal_n = summary[scenario]["optimal_n"]

if map_scope == "Фокусный сектор":
    display_cells = focus_subset(features, best_row["lat"], best_row["lon"], focus_radius_km)
    if len(display_cells) < 25:
        display_cells = features.nlargest(120, "composite_model_score").copy()
    display_cells = display_cells.nlargest(140, "composite_model_score")
    scenario_map_df = focus_subset(scenario_df, best_row["lat"], best_row["lon"], focus_radius_km)
    if scenario_map_df.empty:
        scenario_map_df = scenario_df.copy()
    show_city = False
else:
    display_cells = features.nlargest(320, "composite_model_score").copy()
    scenario_map_df = scenario_df.copy()
    show_city = True

scenario_df["why_recommended"] = scenario_df["h3_index"].map(lambda value: "; ".join(explanations.get(value, [])[:3]))

st.markdown(
    """
    <div class="hero">
      <h1>VTB Geo Analytics</h1>
      <p>Интерактивный инструмент для выбора новых зон под банкоматы ВТБ в Москве. Здесь объединены прогноз спроса, баланс плотности сети и штраф за каннибализацию, чтобы решение было понятным и для бизнеса, и для жюри кейса.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stats_cols = st.columns(4)
with stats_cols[0]:
    metric_block("Режим", scenario, "Текущая бизнес-цель, которая меняет веса в optimizer.")
with stats_cols[1]:
    metric_block("Оптимальное число новых ATM", str(optimal_n), "Точка максимума cumulative objective в выбранном режиме.")
with stats_cols[2]:
    metric_block("Лучшая H3-ячейка", summary[scenario]["top_cell"], "Первая рекомендация оптимизатора в этом сценарии.")
with stats_cols[3]:
    metric_block("Best objective", f"{summary[scenario]['best_cumulative_objective']:.3f}", "Максимум функции полезности сети после учета штрафов.")

st.markdown("### Карта сценария")
st.caption(
    "Нейтральная картография: слой улиц и зданий построен на CartoDB Voyager. "
    "В фокусном режиме показывается сектор Москвы вокруг лучших рекомендаций, а не весь город целиком."
)

scenario_map = build_map(
    display_cells=display_cells,
    scenario_df=scenario_map_df,
    best_row=best_row,
    show_city_overview=show_city,
    show_sector_box=not show_city,
    radius_km=focus_radius_km,
)
st_folium(scenario_map, width=None, height=700, returned_objects=[])

how_cols = st.columns(3)
with how_cols[0]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Как читать цвета</b><br>
          Светлый гексагон — умеренный потенциал.<br>
          Насыщенный темный — сильная зона по композитному model score.
        </div>
        """,
        unsafe_allow_html=True,
    )
with how_cols[1]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Как читать маркеры</b><br>
          Оранжевые точки — shortlist optimizer.<br>
          Темно-синие точки — первые рекомендации с наибольшим приростом objective.
        </div>
        """,
        unsafe_allow_html=True,
    )
with how_cols[2]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Что такое фокусный сектор</b><br>
          Это локальный участок Москвы, где модель видит один из сильных кластеров открытия ATM и где удобно разбирать решение на уровне улиц.
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Shortlist", "Кривая оптимума", "Пояснение статистик"])

with tabs[0]:
    st.subheader("Лучшие точки по сценарию")
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
                "why_recommended",
            ]
        ].rename(
            columns={
                "selected_rank": "Rank",
                "h3_index": "H3",
                "recommended_atm_type": "Тип ATM",
                "marginal_objective": "Marginal objective",
                "cumulative_objective": "Cumulative objective",
                "profit_core_score": "Profit score",
                "coverage_core_score": "Coverage score",
                "social_core_score": "Social score",
                "why_recommended": "Почему попала в shortlist",
            }
        ).round(3),
        use_container_width=True,
    )

with tabs[1]:
    st.subheader("Как выбирается оптимальное число ATM")
    curve = scenario_all.copy()
    fig = px.line(
        curve,
        x="selected_rank",
        y="cumulative_objective",
        markers=True,
        color_discrete_sequence=["#0f2747"],
    )
    fig.add_vline(
        x=optimal_n,
        line_dash="dash",
        line_color="#ff8c42",
        annotation_text=f"optimal_n={optimal_n}",
        annotation_position="top",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(fig, use_container_width=True)

    type_share = scenario_all.head(optimal_n)["recommended_atm_type"].value_counts().rename_axis("Тип ATM").reset_index(name="count")
    fig_types = px.bar(type_share, x="Тип ATM", y="count", color="Тип ATM", color_discrete_sequence=["#0f2747", "#ff8c42", "#5386b5", "#8cb0cb"])
    fig_types.update_layout(height=300, margin=dict(l=0, r=0, t=8, b=0), showlegend=False)
    st.plotly_chart(fig_types, use_container_width=True)

with tabs[2]:
    st.subheader("Зачем эти статистики нужны бизнесу")
    st.markdown(
        """
        - `Profit score` нужен, чтобы не ставить новые ATM в зоны с низкой денежной отдачей.
        - `Coverage score` нужен, чтобы сеть не была слишком плотной в центре и слишком пустой на периферии.
        - `Social score` показывает, насколько выбор помогает сценариям рядом со школами, вузами и медучреждениями.
        - `Marginal objective` важен для ответа на вопрос "сколько банкоматов ставить", а не только "где ставить".
        - `Тип ATM` нужен, потому что одна и та же точка не всегда требует одинакового устройства: high-flow и business cash-in логично ставить в разных кластерах.
        """
    )

st.caption(
    "Дополнительные страницы `Model Explorer`, `Scenario Lens` и `Data Report` оставлены как аналитическое приложение второго уровня; "
    "основной пользовательский сценарий теперь собран на этой странице."
)
