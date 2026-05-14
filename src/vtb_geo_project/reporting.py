from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS


def write_solution_report(features: pd.DataFrame, recommendations: pd.DataFrame, metrics: dict) -> None:
    optimal_rows = []
    for scenario, group in recommendations.groupby("scenario"):
        top = group[group["is_before_optimal_n"]].sort_values("selected_rank")
        optimal_rows.append(
            f"- `{scenario}`: оптимальное число новых ATM = `{int(top['optimal_n_for_scenario'].max())}`, "
            f"лучший cumulative objective = `{top['cumulative_objective'].max():.3f}`"
        )

    report = f"""# Отчет по решению

## Баланс плотности и прибыльности

Алгоритм решает задачу в два шага:

1. На уровне H3-ячейки оценивает спрос на ATM и profitability proxy.
2. На уровне сети применяет greedy-оптимизатор, который добавляет только те точки, где предельная выгода не съедается каннибализацией и риском.

Это позволяет искать не просто `top-K` ячеек, а оптимальное количество банкоматов как точку максимума `cumulative objective`.

{chr(10).join(optimal_rows)}

## Работа с данными

- Внутренние данные кейса агрегированы до `cell-level` по `8 154` H3-ячейкам.
- Внешний слой подтянут из BBBike / OpenStreetMap extract по Москве.
- В итоговом feature store использовано `{features.shape[1]}` признаков на `{features.shape[0]}` ячейках.

## Обогащение данных новыми признаками

Использованы блоки признаков:

- транзакционные агрегаты;
- поведенческие признаки по временным бакетам и MCC;
- spatial lag признаки по H3-соседям;
- расстояния и плотности VTB ATM / branch;
- competitor pressure;
- proximity и counts для `metro`, `school`, `university`, `hospital`, `mall`, `supermarket`, `office`;
- derived scores: `coverage_core_score`, `social_core_score`, `business_core_score`, `profit_core_score`.

## Внутренняя логика алгоритма

1. `CatBoostClassifier` предсказывает вероятность ATM-присутствия.
2. `CatBoostRegressor` предсказывает интенсивность ATM-спроса (`atm_users_count`).
3. Композитный score объединяет классификационную и demand-оценку.
4. Для каждого сценария и типа ATM считается base-profit score.
5. Оптимизатор добавляет новые точки, пока растет cumulative objective.

## Метрики

- Spatial CV AUC: `{metrics['mean']['catboost_auc']:.4f}`
- Spatial CV AP: `{metrics['mean']['catboost_ap']:.4f}`
- Spatial CV Precision@K: `{metrics['mean']['catboost_precision_at_k']:.4f}`
- Spatial CV Recall@K: `{metrics['mean']['catboost_recall_at_k']:.4f}`
- Spatial CV RMSE по ATM demand: `{metrics['mean']['catboost_reg_rmse']:.4f}`
- Spatial CV NDCG@K: `{metrics['mean']['catboost_reg_ndcg']:.4f}`

Baseline для сравнения:

- Baseline AUC: `{metrics['mean']['baseline_auc']:.4f}`
- Baseline AP: `{metrics['mean']['baseline_ap']:.4f}`
- Baseline RMSE: `{metrics['mean']['baseline_reg_rmse']:.4f}`

## Визуализация и Streamlit

Основной Streamlit-проект показывает:

- тепловой слой H3 по model score;
- рекомендации по шести сценариям;
- оптимальное число новых ATM;
- сравнение модели и baseline;
- карточки по типам ATM и объяснимости.

Отдельное приложение игры использует те же артефакты и сравнивает пользовательскую догадку с модельным лучшим placement.

## Ограничения

- `profit_core_score` и cumulative objective — это не бухгалтерская прибыль, а прозрачный profitability proxy.
- В открытом OSM-слое BBBike не у всех ATM есть оператор, поэтому слой текущего присутствия ВТБ лучше всего восстанавливается через точки `ВТБ / bank` и общую ATM-плотность как proxy конкурентной насыщенности.
- Для production-версии модель нужно усилить официальным locator-слоем ВТБ и более точной экономикой инкассации/обслуживания.
"""
    ARTIFACTS.report.write_text(report, encoding="utf-8")
