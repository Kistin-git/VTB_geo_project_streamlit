# Отчет по решению

## Баланс плотности и прибыльности

Алгоритм решает задачу в два шага:

1. На уровне H3-ячейки оценивает спрос на ATM и profitability proxy.
2. На уровне сети применяет greedy-оптимизатор, который добавляет только те точки, где предельная выгода не съедается каннибализацией и риском.

Это позволяет искать не просто `top-K` ячеек, а оптимальное количество банкоматов как точку максимума `cumulative objective`.

- `balanced`: оптимальное число новых ATM = `65`, лучший cumulative objective = `21.609`
- `business`: оптимальное число новых ATM = `56`, лучший cumulative objective = `14.737`
- `competitor`: оптимальное число новых ATM = `56`, лучший cumulative objective = `14.013`
- `coverage`: оптимальное число новых ATM = `50`, лучший cumulative objective = `10.627`
- `profit`: оптимальное число новых ATM = `76`, лучший cumulative objective = `36.276`
- `social`: оптимальное число новых ATM = `40`, лучший cumulative objective = `5.837`

## Работа с данными

- Внутренние данные кейса агрегированы до `cell-level` по `8 154` H3-ячейкам.
- Внешний слой подтянут из BBBike / OpenStreetMap extract по Москве.
- В итоговом feature store использовано `131` признаков на `8154` ячейках.

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

- Spatial CV AUC: `0.9101`
- Spatial CV AP: `0.7294`
- Spatial CV Precision@K: `0.7000`
- Spatial CV Recall@K: `0.5197`
- Spatial CV RMSE по ATM demand: `50.8795`
- Spatial CV NDCG@K: `0.7823`

Baseline для сравнения:

- Baseline AUC: `0.8590`
- Baseline AP: `0.6882`
- Baseline RMSE: `156.8893`

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
