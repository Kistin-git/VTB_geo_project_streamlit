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


st.set_page_config(page_title="Modeling Details", layout="wide")
features = pd.read_parquet(ARTIFACTS.features)

feature_examples = {
    "Транзакционные": [
        "tx_count",
        "tx_sum",
        "avg_ticket",
        "unique_customers",
        "mcc_entropy",
        "peak_time_share",
    ],
    "Пространственные": [
        "tx_count_ring1",
        "tx_sum_ring2",
        "atm_users_count_ring1",
        "coverage_gap_score",
    ],
    "Сетевые": [
        "dist_vtb_service_km",
        "vtb_network_density",
        "network_gap_score",
    ],
    "Конкуренты и инфраструктура": [
        "competitor_pressure",
        "dist_metro_km",
        "school_ring2",
        "office_ring2",
        "social_intensity",
        "business_intensity",
    ],
}

st.title("Modeling Details")
st.write(
    "Подробный рассказ о фичах и алгоритмическом устройстве проекта. "
    "Это страница для тех, кто хочет понять не только интерфейс, но и внутреннюю механику pipeline."
)

with st.expander("Почему именно такой стек", expanded=True):
    st.markdown(
        """
        - Табличные данные + большое число engineered features => лучший прагматичный выбор — boosting по табличным данным.
        - Географическая структура не игнорируется: она попадает в признаки через H3-соседей, расстояния, spatial lag и ручные score-блоки.
        - Для бизнеса критична интерпретируемость, поэтому сложный end-to-end географический black box здесь был бы слабее на защите.
        """
    )

for title, cols in feature_examples.items():
    st.subheader(title)
    present = [col for col in cols if col in features.columns]
    if present:
        st.dataframe(features[present].head(10), use_container_width=True)
    else:
        st.info("В этом демо нужные колонки не попали в итоговый sample.")

with st.expander("Последовательность работы алгоритма", expanded=True):
    st.markdown(
        """
        1. Собираем внутренние транзакционные агрегаты по H3.
        2. Добавляем внешний геослой и близость к POI.
        3. Считаем пространственные признаки соседства.
        4. Обучаем классификатор ATM-presence и регрессор ATM-demand.
        5. Собираем composite score по ячейке.
        6. Для каждого сценария рассчитываем contribution разных типов ATM.
        7. Greedy optimizer последовательно набирает сеть, пересчитывая предельную пользу и штрафы.
        """
    )

st.markdown(
    """
    Ограничение проекта честно зафиксировано: `profit_core_score` — это не бухгалтерская прибыль, а прозрачный profitability proxy.
    Для production-версии его нужно калибровать на реальных данных по CAPEX, OPEX, инкассации и доступности площадки.
    """
)
