from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import h3
import numpy as np
import pandas as pd


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def minmax(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if values.empty:
        return values
    vmin = values.min()
    vmax = values.max()
    if math.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if values.empty:
        return values
    std = values.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def cell_lat_lon(cell: str) -> tuple[float, float]:
    lat, lon = h3.cell_to_latlng(cell)
    return float(lat), float(lon)


def cell_polygon(cell: str) -> list[list[float]]:
    boundary = h3.cell_to_boundary(cell)
    return [[float(lat), float(lon)] for lat, lon in boundary]


def json_dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(np.clip(series.astype(float), a_min=0, a_max=None))


def rank_normalize(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    return ranked.fillna(0.0)


def top_feature_reasons(row: pd.Series, importance: pd.DataFrame, top_n: int = 3) -> list[str]:
    reasons: list[str] = []
    feature_weights = importance.set_index("feature")["importance_norm"].to_dict()
    for feature, weight in sorted(feature_weights.items(), key=lambda item: item[1], reverse=True):
        if feature not in row.index:
            continue
        value = row[feature]
        if pd.isna(value):
            continue
        if abs(float(value)) < 0.1:
            continue
        reasons.append(f"{feature}={value:.2f}")
        if len(reasons) >= top_n:
            break
    return reasons

