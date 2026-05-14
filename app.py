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
from shapely.geometry import Point, Polygon
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS, SCENARIO_WEIGHTS
from vtb_geo_project.optimization import optimize_scenario
from vtb_geo_project.utils import haversine_km


SCENARIO_DESCRIPTIONS = {
    "balanced": "Компромисс между profitability proxy, покрытием, социальным эффектом и штрафом за каннибализацию.",
    "profit": "Режим для максимизации бизнес-эффекта: выше вес у денежного потока и ожидаемого ATM-спроса.",
    "coverage": "Режим для закрытия сервисных пустот: приоритет получают зоны далеко от текущего присутствия ВТБ.",
    "social": "Режим социальной миссии: усиливает районы рядом со школами, вузами и медицинскими объектами.",
    "competitor": "Режим конкурентного ответа: ищет клетки, где есть клиентская активность и насыщенность чужими ATM.",
    "business": "Режим для cash-in и малого бизнеса: сильнее оценивает торговые и офисные кластеры.",
}

AREA_MODE_DESCRIPTIONS = {
    "Фокусный сектор": "Автоматически показывает один из сильных локальных кластеров рекомендаций.",
    "Ручной прямоугольник": "Кликните по двум противоположным углам прямоугольника и посмотрите локальный shortlist внутри него.",
    "Одна H3 ячейка": "Кликните по карте и проверьте, как модель оценивает конкретный гексагон.",
    "Вся Москва": "Показывает глобальную картину по всей столице и общегородской shortlist.",
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
          html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            color: #10253f;
          }
          p, div, span, label, li, .stMarkdown, .stCaption {
            color: #10253f;
          }
          .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f2747 0%, #183a63 100%);
            color: #f8f2e8;
            box-shadow: 0 18px 42px rgba(15,39,71,0.18);
          }
          .hero, .hero * { color: #f8f2e8 !important; }
          .hero h1 { margin: 0; font-size: 2.25rem; }
          .hero p { margin: 0.45rem 0 0 0; color: #d4dfef !important; max-width: 980px; }
          .soft-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(15,39,71,0.08);
            box-shadow: 0 12px 30px rgba(15,39,71,0.06);
          }
          .metric-note {
            font-size: 0.88rem;
            color: #4a6079;
            margin-top: -0.3rem;
          }
          .map-help {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15,39,71,0.08);
            box-shadow: 0 10px 24px rgba(15,39,71,0.05);
            margin-bottom: 0.8rem;
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


def polygon_from_drawings(drawings: list[dict] | None) -> Polygon | None:
    if not drawings:
        return None
    geometry = drawings[-1].get("geometry", {})
    if geometry.get("type") != "Polygon":
        return None
    coords = geometry.get("coordinates", [])
    if not coords:
        return None
    return Polygon(coords[0])


def rectangle_from_corners(corner_a: tuple[float, float], corner_b: tuple[float, float]) -> Polygon:
    lat1, lon1 = corner_a
    lat2, lon2 = corner_b
    min_lat, max_lat = sorted([lat1, lat2])
    min_lon, max_lon = sorted([lon1, lon2])
    return Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )


def nearest_cell(lat: float, lon: float, cells: pd.DataFrame) -> pd.Series:
    query_cell = h3.latlng_to_cell(lat, lon, 9)
    match = cells[cells["h3_index"] == query_cell]
    if not match.empty:
        return match.iloc[0]
    idx = ((cells["lat"] - lat) ** 2 + (cells["lon"] - lon) ** 2).idxmin()
    return cells.loc[idx]


def filter_by_polygon(frame: pd.DataFrame, polygon: Polygon) -> pd.DataFrame:
    return frame[
        frame.apply(
            lambda row: polygon.contains(Point(float(row["lon"]), float(row["lat"])))
            or polygon.touches(Point(float(row["lon"]), float(row["lat"]))),
            axis=1,
        )
    ].copy()


def build_map(
    display_cells: pd.DataFrame,
    scenario_df: pd.DataFrame,
    best_row: pd.Series,
    area_mode: str,
    radius_km: float,
    selected_polygon: Polygon | None = None,
    selected_cell: str | None = None,
    pending_corner: tuple[float, float] | None = None,
) -> folium.Map:
    show_city = area_mode in {"Вся Москва", "Ручной прямоугольник", "Одна H3 ячейка"}
    zoom_start = 10 if show_city else 13
    map_center = [55.7558, 37.6176] if show_city else [best_row["lat"], best_row["lon"]]
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
        max_zoom=17,
    )

    if not show_city:
        lat_pad = radius_km / 110.574
        lon_pad = radius_km / (111.320 * max(0.2, abs(math.cos(math.radians(best_row["lat"])))))
        bounds = [
            [best_row["lat"] - lat_pad, best_row["lon"] - lon_pad],
            [best_row["lat"] + lat_pad, best_row["lon"] + lon_pad],
        ]
        m.fit_bounds(bounds)
        folium.Rectangle(
            bounds=bounds,
            color="#0f2747",
            weight=2,
            fill=False,
            dash_array="6 6",
            tooltip="Фокусный сектор",
        ).add_to(m)

    if not display_cells.empty:
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

    if selected_polygon is not None:
        folium.GeoJson(
            data=selected_polygon.__geo_interface__,
            style_function=lambda _: {
                "color": "#0f2747",
                "weight": 3,
                "fillColor": "#ff8c42",
                "fillOpacity": 0.08,
            },
            tooltip="Выбранная прямоугольная область",
        ).add_to(m)

    if selected_cell is not None:
        folium.Polygon(
            locations=cell_boundary(selected_cell),
            color="#0f2747",
            weight=3,
            fill=True,
            fill_color="#ff8c42",
            fill_opacity=0.18,
            tooltip=f"Выбранная H3-ячейка: {selected_cell}",
        ).add_to(m)

    if pending_corner is not None:
        folium.CircleMarker(
            location=[pending_corner[0], pending_corner[1]],
            radius=8,
            color="#ff8c42",
            weight=3,
            fill=True,
            fill_color="#ff8c42",
            fill_opacity=0.96,
            tooltip="Первая вершина прямоугольника зафиксирована",
        ).add_to(m)

    return m


def metric_block(label: str, value: str, note: str) -> None:
    st.metric(label, value)
    st.markdown(f"<div class='metric-note'>{note}</div>", unsafe_allow_html=True)


def compute_area_analysis(area_features: pd.DataFrame, scenario: str, top_n: int) -> tuple[pd.DataFrame, int, float, str]:
    if area_features.empty:
        return area_features.head(0).copy(), 0, 0.0, "n/a"

    if len(area_features) == 1:
        single = area_features.copy()
        single["selected_rank"] = 1
        single["recommended_atm_type"] = "full_service_cash_in_qr"
        single["marginal_objective"] = single["profit_core_score"]
        single["cumulative_objective"] = single["marginal_objective"]
        single["optimal_n_for_scenario"] = 1
        single["is_before_optimal_n"] = True
        return single, 1, float(single["cumulative_objective"].iloc[0]), str(single["h3_index"].iloc[0])

    local_recs = optimize_scenario(area_features.copy(), scenario, max_new_atm=min(max(16, top_n * 3), len(area_features)))
    if local_recs.empty:
        fallback = area_features.sort_values("composite_model_score", ascending=False).head(top_n).copy()
        fallback["selected_rank"] = range(1, len(fallback) + 1)
        fallback["marginal_objective"] = fallback["profit_core_score"]
        fallback["cumulative_objective"] = fallback["marginal_objective"].cumsum()
        fallback["recommended_atm_type"] = "full_service_cash_in_qr"
        fallback["optimal_n_for_scenario"] = len(fallback)
        fallback["is_before_optimal_n"] = True
        best_obj = float(fallback["cumulative_objective"].max())
        top_cell = str(fallback.iloc[0]["h3_index"])
        return fallback, len(fallback), best_obj, top_cell

    optimal_n = int(local_recs["optimal_n_for_scenario"].max())
    best_obj = float(local_recs["cumulative_objective"].max())
    top_cell = str(local_recs.sort_values("selected_rank").iloc[0]["h3_index"])
    return local_recs, optimal_n, best_obj, top_cell


st.set_page_config(page_title="VTB Geo Analytics", page_icon="🏦", layout="wide")
apply_style()

features, recommendations, summary, explanations = load_data()

if "project_area_mode" not in st.session_state:
    st.session_state["project_area_mode"] = "Фокусный сектор"
if "selected_h3" not in st.session_state:
    st.session_state["selected_h3"] = None
if "selected_polygon" not in st.session_state:
    st.session_state["selected_polygon"] = None
if "rectangle_corner_a" not in st.session_state:
    st.session_state["rectangle_corner_a"] = None
if "project_last_click_key" not in st.session_state:
    st.session_state["project_last_click_key"] = None

st.sidebar.title("Управление")
scenario = st.sidebar.selectbox("Режим оптимизации", list(SCENARIO_WEIGHTS))
top_n = st.sidebar.slider("Сколько рекомендованных точек показать", min_value=5, max_value=30, value=12)
focus_radius_km = st.sidebar.slider("Радиус фокусного сектора, км", min_value=2.0, max_value=6.0, value=3.4, step=0.2)
area_mode = st.sidebar.selectbox(
    "Режим области анализа",
    ["Фокусный сектор", "Ручной прямоугольник", "Одна H3 ячейка", "Вся Москва"],
    key="project_area_mode",
)
if st.sidebar.button("Вся Москва", use_container_width=True):
    st.session_state["project_area_mode"] = "Вся Москва"
    st.rerun()
if st.sidebar.button("Сбросить выделение", use_container_width=True):
    st.session_state["selected_h3"] = None
    st.session_state["selected_polygon"] = None
    st.session_state["rectangle_corner_a"] = None
    st.session_state["project_last_click_key"] = None
    st.rerun()

st.sidebar.markdown("### Что делает режим")
st.sidebar.info(SCENARIO_DESCRIPTIONS[scenario])
st.sidebar.markdown("### Какой тип области сейчас выбран")
st.sidebar.caption(AREA_MODE_DESCRIPTIONS[st.session_state["project_area_mode"]])

with st.sidebar.expander("Как пользоваться демо", expanded=True):
    st.markdown(
        """
        1. Выберите бизнес-режим слева.
        2. Выберите область анализа:
           - `Фокусный сектор` для готового локального примера;
           - `Ручной прямоугольник`: первый клик ставит первую вершину, второй клик замыкает прямоугольник;
           - `Одна H3 ячейка` для анализа конкретного гексагона;
           - `Вся Москва` для общегородской картины.
        3. В режиме прямоугольника третий клик начинает новое выделение.
        4. В режиме H3 один клик выбирает ближайший гексагон.
        4. Ниже приложение локально пересчитает shortlist и оптимум уже только внутри выбранной зоны.
        """
    )

with st.sidebar.expander("Что значат метрики"):
    st.markdown(
        """
        - `Optimal New ATM` — число новых банкоматов, при котором внутри выбранной области достигается максимум cumulative objective.
        - `Best Objective` — интегральная функция полезности: прибыльность, покрытие, соцэффект и штраф за каннибализацию.
        - `Coverage score` — насколько область помогает закрыть пробелы текущей сети.
        - `Marginal objective` — предельная польза от следующей точки.
        """
    )

scenario_global = recommendations[recommendations["scenario"] == scenario].sort_values("selected_rank").copy()
global_best = scenario_global.iloc[0]

selected_polygon = st.session_state.get("selected_polygon")
selected_h3 = st.session_state.get("selected_h3")
pending_corner = st.session_state.get("rectangle_corner_a")

if area_mode == "Фокусный сектор":
    analysis_features = focus_subset(features, global_best["lat"], global_best["lon"], focus_radius_km)
    if len(analysis_features) < 30:
        analysis_features = features.nlargest(160, "composite_model_score").copy()
    analysis_features = analysis_features.nlargest(180, "composite_model_score")
    area_label = "Локальный фокусный сектор"
elif area_mode == "Вся Москва":
    analysis_features = features.copy()
    area_label = "Вся Москва"
elif area_mode == "Ручной прямоугольник":
    analysis_features = features.copy()
    if selected_polygon is not None:
        analysis_features = filter_by_polygon(features, selected_polygon)
    area_label = "Ручной прямоугольник"
else:
    analysis_features = features.copy()
    if selected_h3 is not None:
        analysis_features = features[features["h3_index"] == selected_h3].copy()
    area_label = "Одна H3 ячейка"

analysis_recs, optimal_n, best_obj, top_cell = compute_area_analysis(analysis_features, scenario, top_n)
analysis_recs_full = analysis_recs.sort_values("selected_rank").copy()
analysis_shortlist = analysis_recs_full.head(top_n).copy()
analysis_shortlist["why_recommended"] = analysis_shortlist["h3_index"].map(lambda value: "; ".join(explanations.get(value, [])[:3]))

pre_map_recs = (
    analysis_shortlist.copy()
    if area_mode in {"Ручной прямоугольник", "Одна H3 ячейка"} and not analysis_shortlist.empty
    else scenario_global.head(top_n).copy()
)
pre_map_cells = (
    analysis_features.nlargest(650, "composite_model_score").copy()
    if area_mode in {"Вся Москва", "Ручной прямоугольник", "Одна H3 ячейка"}
    else analysis_features.copy()
)

st.markdown(
    """
    <div class="hero">
      <h1>VTB Geo Analytics</h1>
      <p>Интерактивный инструмент для выбора новых зон под банкоматы ВТБ в Москве. Здесь объединены прогноз спроса, баланс плотности сети и штраф за каннибализацию, а ручные режимы карты имитируют работу менеджера сети, который анализирует конкретный участок города.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Карта сценария")
st.caption("Нейтральная картография: слой улиц и зданий построен на стандартном OpenStreetMap и открывается без VPN.")

if area_mode == "Ручной прямоугольник":
    if pending_corner is None and selected_polygon is None:
        st.markdown(
            """
            <div class="map-help">
              <b>Как выделить прямоугольник</b><br>
              Первый клик ставит первую вершину. Второй клик ставит противоположную вершину и сразу замыкает область.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif pending_corner is not None and selected_polygon is None:
        st.markdown(
            """
            <div class="map-help">
              <b>Первая вершина сохранена</b><br>
              Сделайте второй клик по карте: будет построен прямоугольник и локальный shortlist пересчитается автоматически.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="map-help">
              <b>Прямоугольник выбран</b><br>
              Локальные метрики и рекомендации ниже уже пересчитаны только внутри этой области. Следующий клик начнет новое выделение.
            </div>
            """,
            unsafe_allow_html=True,
        )
elif area_mode == "Одна H3 ячейка":
    st.markdown(
        """
        <div class="map-help">
          <b>Режим H3</b><br>
          Кликните по карте один раз. Приложение выберет ближайший гексагон и покажет его локальную оценку как отдельного кандидата.
        </div>
        """,
        unsafe_allow_html=True,
    )

scenario_map = build_map(
    display_cells=pre_map_cells,
    scenario_df=pre_map_recs,
    best_row=global_best,
    area_mode=area_mode,
    radius_km=focus_radius_km,
    selected_polygon=selected_polygon,
    selected_cell=selected_h3,
    pending_corner=pending_corner,
)
returned = st_folium(
    scenario_map,
    width=None,
    height=720,
    returned_objects=["last_clicked"],
    key="project_main_map",
)

if returned and returned.get("last_clicked"):
    clicked = returned["last_clicked"]
    click_key = f"{area_mode}:{clicked['lat']:.5f}:{clicked['lng']:.5f}"
    if st.session_state.get("project_last_click_key") != click_key:
        st.session_state["project_last_click_key"] = click_key
        if area_mode == "Ручной прямоугольник":
            if st.session_state.get("selected_polygon") is not None:
                st.session_state["selected_polygon"] = None
                st.session_state["rectangle_corner_a"] = (clicked["lat"], clicked["lng"])
            elif st.session_state.get("rectangle_corner_a") is None:
                st.session_state["rectangle_corner_a"] = (clicked["lat"], clicked["lng"])
            else:
                corner_a = st.session_state["rectangle_corner_a"]
                st.session_state["selected_polygon"] = rectangle_from_corners(
                    corner_a,
                    (clicked["lat"], clicked["lng"]),
                )
                st.session_state["rectangle_corner_a"] = None
            st.rerun()
        elif area_mode == "Одна H3 ячейка":
            selected_h3 = nearest_cell(clicked["lat"], clicked["lng"], features)["h3_index"]
            if st.session_state.get("selected_h3") != selected_h3:
                st.session_state["selected_h3"] = selected_h3
                st.rerun()

if area_mode == "Одна H3 ячейка" and st.session_state.get("selected_h3") is not None:
    selected_h3 = st.session_state["selected_h3"]
    analysis_features = features[features["h3_index"] == selected_h3].copy()
    area_label = f"Выбранная H3-ячейка {selected_h3}"
elif area_mode == "Ручной прямоугольник" and st.session_state.get("selected_polygon") is not None:
    selected_polygon = st.session_state["selected_polygon"]
    analysis_features = filter_by_polygon(features, selected_polygon)
    area_label = "Пользовательский прямоугольник"

analysis_recs, optimal_n, best_obj, top_cell = compute_area_analysis(analysis_features, scenario, top_n)
analysis_recs_full = analysis_recs.sort_values("selected_rank").copy()
analysis_shortlist = analysis_recs_full.head(top_n).copy()
analysis_shortlist["why_recommended"] = analysis_shortlist["h3_index"].map(lambda value: "; ".join(explanations.get(value, [])[:3]))

stats_cols = st.columns(4)
with stats_cols[0]:
    metric_block("Режим", scenario, "Текущая бизнес-цель, которая меняет веса в optimizer.")
with stats_cols[1]:
    metric_block("Область анализа", area_label, "Зона, внутри которой сейчас локально считается shortlist.")
with stats_cols[2]:
    metric_block("Optimal New ATM", str(optimal_n), "Локальный максимум cumulative objective внутри текущей области.")
with stats_cols[3]:
    metric_block("Best Objective", f"{best_obj:.3f}", "Максимум целевой функции в выбранной области.")

selection_cols = st.columns(3)
with selection_cols[0]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Ручной режим на карте</b><br>
          В `Ручном прямоугольнике` можно выделить любой кусок Москвы и получить локальный shortlist без перехода на весь город.
        </div>
        """,
        unsafe_allow_html=True,
    )
with selection_cols[1]:
    st.markdown(
        """
        <div class="soft-card">
          <b>H3 по клику</b><br>
          В режиме `Одна H3 ячейка` менеджер может проверить конкретный гексагон как отдельного кандидата на установку.
        </div>
        """,
        unsafe_allow_html=True,
    )
with selection_cols[2]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Зачем оставлена вся Москва</b><br>
          Этот режим нужен, чтобы быстро вернуться к общегородской картине и глобальному оптимуму без локального фильтра.
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Shortlist", "Кривая оптимума", "Пояснение статистик"])

with tabs[0]:
    st.subheader("Лучшие точки внутри текущей области")
    if analysis_shortlist.empty:
        st.warning("Внутри выбранной области не найдено достаточно сильных клеток для локального shortlist.")
    else:
        st.dataframe(
            analysis_shortlist[
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
    if analysis_recs_full.empty:
        st.info("Сначала выделите область с достаточным числом клеток.")
    else:
        curve = analysis_recs_full.copy()
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

        type_share = (
            analysis_recs_full.head(optimal_n)["recommended_atm_type"].value_counts().rename_axis("Тип ATM").reset_index(name="count")
        )
        fig_types = px.bar(
            type_share,
            x="Тип ATM",
            y="count",
            color="Тип ATM",
            color_discrete_sequence=["#0f2747", "#ff8c42", "#5386b5", "#8cb0cb"],
        )
        fig_types.update_layout(height=300, margin=dict(l=0, r=0, t=8, b=0), showlegend=False)
        st.plotly_chart(fig_types, use_container_width=True)

with tabs[2]:
    st.subheader("Зачем эти статистики нужны бизнесу")
    st.markdown(
        """
        - `Profit score` нужен, чтобы не ставить новые ATM в зоны с низкой денежной отдачей.
        - `Coverage score` нужен, чтобы сеть не была слишком плотной в центре и слишком пустой на периферии.
        - `Social score` показывает, насколько выбор помогает сценариям рядом со школами, вузами и медучреждениями.
        - `Marginal objective` важен для ответа на вопрос `сколько банкоматов ставить`, а не только `где ставить`.
        - `Тип ATM` нужен, потому что одна и та же точка не всегда требует одинакового устройства: high-flow и business cash-in логично ставить в разных кластерах.
        - Ручное выделение области позволяет проверять решение не на абстрактной карте Москвы, а на конкретной зоне интереса менеджера сети.
        """
    )

st.caption(
    "Дополнительные страницы `Model Explorer`, `Scenario Lens`, `Data Report` и новые страницы методологии теперь подробно объясняют, "
    "как решалась задача баланса, как устроены признаки и почему выбраны именно эти алгоритмы."
)
