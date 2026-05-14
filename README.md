# VTB Geo Project Streamlit

Геоаналитический проект по выбору оптимальных зон размещения банкоматов ВТБ в Москве.

## Что внутри

- воспроизводимый pipeline подготовки данных;
- обогащение H3-ячеек внутренними и внешними геопризнаками;
- baseline + CatBoost-модели с пространственной валидацией;
- оптимизатор баланса прибыльности, покрытия и каннибализации;
- Streamlit-интерфейс для анализа сценариев;
- экспорт компактных артефактов для отдельной игры.

## Быстрый старт

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_all.py --game-repo ../VTB_geo_game_streamlit
streamlit run app.py
```

## Откуда берутся данные

### Локальные данные кейса

Скрипт автоматически ищет архив кейса в одном из мест:

- `data/raw/case_bundle.zip`
- `../Данные для продвинутого анализа-20260514T092848Z-3-001.zip`
- `../../Данные для продвинутого анализа-20260514T092848Z-3-001.zip`

### Внешние открытые данные

- BBBike / OpenStreetMap extract for Moscow:
  `https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.shp.zip`

Фичи получают `source_confidence` по классам:

- `official_case_data`
- `community_open_data`

## Что генерируется

- `data/processed/cell_features.parquet` — итоговый cell-level слой.
- `data/processed/recommendations.parquet` — рекомендации по сценариям.
- `data/processed/model_metrics.json` — метрики.
- `data/processed/dashboard_summary.json` — KPI для приложения.
- `reports/solution_report.md` — отчет по данным, модели и визуализации.

## Сценарии в приложении

- `balanced`
- `profit`
- `coverage`
- `social`
- `competitor`
- `business`
