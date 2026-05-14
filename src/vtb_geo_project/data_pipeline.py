from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyogrio
import requests
from sklearn.neighbors import BallTree

from .config import (
    ARTIFACTS,
    CASE_ARCHIVE_CANDIDATES,
    CASE_DATA_FILE,
    CASE_EXTRACT_DIR,
    CASE_TARGET_FILE,
    COMPETITOR_TEXT_TOKENS,
    EXTERNAL_DIR,
    H3_RESOLUTION,
    MOSCOW_CENTER_LAT,
    MOSCOW_CENTER_LON,
    OSM_SHAPE_DIR,
    OSM_SHAPE_URL,
    OSM_SHAPE_ZIP,
    POI_CATEGORY_RULES,
    PROCESSED_DIR,
    RAW_DIR,
    VTB_TEXT_TOKENS,
)
from .utils import cell_lat_lon, ensure_dirs, haversine_km, minmax, safe_log1p


def ensure_case_data() -> tuple[Path, Path]:
    ensure_dirs([RAW_DIR, CASE_EXTRACT_DIR])
    if CASE_DATA_FILE.exists() and CASE_TARGET_FILE.exists():
        return CASE_DATA_FILE, CASE_TARGET_FILE

    archive_path = next((path for path in CASE_ARCHIVE_CANDIDATES if path.exists()), None)
    if archive_path is None:
        tried = "\n".join(str(path) for path in CASE_ARCHIVE_CANDIDATES)
        raise FileNotFoundError(f"Не найден архив кейса. Проверены пути:\n{tried}")

    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            filename = Path(member.filename).name
            if filename not in {"data.parquet", "target.parquet"}:
                continue
            output_path = CASE_EXTRACT_DIR / filename
            with zf.open(member) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return CASE_DATA_FILE, CASE_TARGET_FILE


def ensure_osm_extract() -> Path:
    ensure_dirs([EXTERNAL_DIR])
    if OSM_SHAPE_DIR.exists():
        return OSM_SHAPE_DIR

    if not OSM_SHAPE_ZIP.exists():
        response = requests.get(OSM_SHAPE_URL, stream=True, timeout=300)
        response.raise_for_status()
        with open(OSM_SHAPE_ZIP, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    OSM_SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OSM_SHAPE_ZIP) as zf:
        zf.extractall(OSM_SHAPE_DIR)
    return OSM_SHAPE_DIR


def load_case_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path, target_path = ensure_case_data()
    data = pd.read_parquet(data_path)
    target = pd.read_parquet(target_path)
    return data, target


def _entropy_from_counts(values: pd.Series) -> float:
    counts = values[values > 0].astype(float)
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _build_internal_features(data: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("h3_index").agg(
        row_count=("h3_index", "size"),
        tx_count=("count", "sum"),
        tx_sum=("sum", "sum"),
        tx_avg_row=("avg", "mean"),
        tx_min=("min", "min"),
        tx_max=("max", "max"),
        tx_std_row=("std", "mean"),
        tx_unique_tx=("count_distinct", "sum"),
        unique_customers=("customer_id", "nunique"),
        unique_mcc=("mcc_code", "nunique"),
        unique_time_buckets=("datetime_id", "nunique"),
    )
    grouped["avg_ticket"] = grouped["tx_sum"] / grouped["tx_count"].replace(0, np.nan)
    grouped["tx_per_customer"] = grouped["tx_count"] / grouped["unique_customers"].replace(0, np.nan)
    grouped["sum_per_customer"] = grouped["tx_sum"] / grouped["unique_customers"].replace(0, np.nan)

    mcc_pivot = (
        data.groupby(["h3_index", "mcc_code"])["count"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("mcc_count_")
    )
    time_pivot = (
        data.groupby(["h3_index", "datetime_id"])["count"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("time_count_")
    )

    features = grouped.join(mcc_pivot, how="left").join(time_pivot, how="left").fillna(0)

    mcc_cols = [col for col in features.columns if col.startswith("mcc_count_")]
    time_cols = [col for col in features.columns if col.startswith("time_count_")]

    features["mcc_entropy"] = features[mcc_cols].apply(_entropy_from_counts, axis=1)
    features["time_entropy"] = features[time_cols].apply(_entropy_from_counts, axis=1)
    features["peak_time_share"] = features[time_cols].max(axis=1) / features[time_cols].sum(axis=1).replace(0, np.nan)
    features["peak_mcc_share"] = features[mcc_cols].max(axis=1) / features[mcc_cols].sum(axis=1).replace(0, np.nan)

    atm_users = target.groupby("h3_index")["customer_id"].nunique().rename("atm_users_count")
    features = features.join(atm_users, how="left").fillna({"atm_users_count": 0})
    features["atm_presence"] = (features["atm_users_count"] > 0).astype(int)
    features = features.reset_index()

    latitudes = []
    longitudes = []
    distances = []
    for cell in features["h3_index"]:
        lat, lon = cell_lat_lon(cell)
        latitudes.append(lat)
        longitudes.append(lon)
        distances.append(haversine_km(lat, lon, MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON))

    features["lat"] = latitudes
    features["lon"] = longitudes
    features["dist_to_city_center_km"] = distances
    features["log_tx_sum"] = safe_log1p(features["tx_sum"])
    features["log_tx_count"] = safe_log1p(features["tx_count"])
    features["log_atm_users_count"] = safe_log1p(features["atm_users_count"])
    return features


def _neighbor_aggregate_lookup(features: pd.DataFrame, column: str, ring: int) -> pd.Series:
    value_map = features.set_index("h3_index")[column].to_dict()
    results = []
    for cell in features["h3_index"]:
        neighbors = h3.grid_ring(cell, ring)
        values = [value_map.get(neighbor, 0.0) for neighbor in neighbors]
        results.append(float(np.sum(values)))
    return pd.Series(results, index=features.index)


def add_spatial_features(features: pd.DataFrame) -> pd.DataFrame:
    for base_col in ("tx_count", "tx_sum", "unique_customers", "atm_users_count"):
        for ring in (1, 2):
            features[f"{base_col}_ring{ring}"] = _neighbor_aggregate_lookup(features, base_col, ring)
    features["coverage_gap_score"] = minmax(features["dist_to_city_center_km"]) * (1 - minmax(features["atm_users_count"]))
    return features


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _read_vector_layers(base_dir: Path) -> gpd.GeoDataFrame:
    vector_paths = sorted(base_dir.rglob("*.shp"))
    frames: list[gpd.GeoDataFrame] = []
    for shp_path in vector_paths:
        try:
            gdf = gpd.read_file(shp_path)
        except Exception:
            continue
        if gdf.empty or "geometry" not in gdf.columns:
            continue
        gdf = gdf.to_crs(4326)
        gdf["source_layer"] = shp_path.stem.lower()
        frames.append(gdf)

    gpkg_paths = sorted(base_dir.rglob("*.gpkg"))
    for gpkg_path in gpkg_paths:
        try:
            layers = pyogrio.list_layers(gpkg_path)
        except Exception:
            continue
        for layer_name, _ in layers:
            try:
                gdf = gpd.read_file(gpkg_path, layer=layer_name)
            except Exception:
                continue
            if gdf.empty or "geometry" not in gdf.columns:
                continue
            gdf = gdf.to_crs(4326)
            gdf["source_layer"] = layer_name.lower()
            frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=4326)
    if not gdf.empty:
        gdf = gdf[gdf.geometry.notna()].copy()
        metric = gdf.to_crs(3857)
        gdf["geometry"] = metric.geometry.centroid.to_crs(4326)
    return gdf


def _extract_text_blob(row: pd.Series) -> str:
    parts: list[str] = []
    for column, value in row.items():
        if column == "geometry":
            continue
        if pd.isna(value):
            continue
        parts.append(f"{column}={_normalize_text(value)}")
    blob = " | ".join(parts)
    return f" {blob} "


def _classify_poi(blob: str) -> dict[str, int]:
    result = {name: 0 for name in POI_CATEGORY_RULES}
    for name, tokens in POI_CATEGORY_RULES.items():
        if any(token in blob for token in tokens):
            result[name] = 1
    result["vtb_atm"] = int(result["atm_any"] and any(token in blob for token in VTB_TEXT_TOKENS))
    result["competitor_atm"] = int(result["atm_any"] and not result["vtb_atm"])
    result["vtb_branch"] = int(any(token in blob for token in VTB_TEXT_TOKENS) and ("bank" in blob or "branch" in blob or "office" in blob))
    result["competitor_brand"] = int(any(token in blob for token in COMPETITOR_TEXT_TOKENS))
    return result


def build_external_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    osm_dir = ensure_osm_extract()
    osm = _read_vector_layers(osm_dir)

    if osm.empty:
        summary = pd.DataFrame(
            [{"source": "bbbike_osm", "records": 0, "status": "empty", "confidence": "community_open_data"}]
        )
        return features, summary

    osm = osm.copy()
    osm["lat"] = osm.geometry.y.astype(float)
    osm["lon"] = osm.geometry.x.astype(float)
    osm["text_blob"] = osm.apply(_extract_text_blob, axis=1)

    categories = osm["text_blob"].apply(_classify_poi).apply(pd.Series)
    osm = pd.concat([osm.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    osm["h3_index"] = osm.apply(lambda row: h3.latlng_to_cell(row["lat"], row["lon"], H3_RESOLUTION), axis=1)

    category_names = [
        "vtb_atm",
        "competitor_atm",
        "vtb_branch",
        "school",
        "university",
        "hospital",
        "mall",
        "supermarket",
        "office",
        "metro",
    ]

    summary_rows = []
    for category in category_names:
        subset = osm[osm[category] == 1]
        summary_rows.append(
            {
                "source": category,
                "records": int(len(subset)),
                "status": "ok",
                "confidence": "community_open_data",
            }
        )
        if subset.empty:
            features[f"{category}_in_cell"] = 0
            features[f"{category}_ring1"] = 0
            features[f"{category}_ring2"] = 0
            features[f"dist_{category}_km"] = np.nan
            continue

        cell_counts = subset.groupby("h3_index").size().to_dict()
        in_cell = []
        ring1 = []
        ring2 = []
        for cell in features["h3_index"]:
            in_cell.append(int(cell_counts.get(cell, 0)))
            ring1.append(int(sum(cell_counts.get(c, 0) for c in h3.grid_ring(cell, 1))))
            ring2.append(int(sum(cell_counts.get(c, 0) for c in h3.grid_ring(cell, 2))))
        features[f"{category}_in_cell"] = in_cell
        features[f"{category}_ring1"] = ring1
        features[f"{category}_ring2"] = ring2

        coords = np.radians(subset[["lat", "lon"]].to_numpy())
        tree = BallTree(coords, metric="haversine")
        distances, _ = tree.query(np.radians(features[["lat", "lon"]].to_numpy()), k=1)
        features[f"dist_{category}_km"] = distances[:, 0] * 6371.0088

    features["social_intensity"] = (
        1.3 * features["school_ring2"]
        + 1.1 * features["hospital_ring2"]
        + 1.0 * features["university_ring2"]
    )
    features["business_intensity"] = (
        1.2 * features["office_ring2"]
        + 1.1 * features["mall_ring2"]
        + 1.0 * features["supermarket_ring2"]
    )
    features["high_flow_intensity"] = (
        1.4 * features["metro_ring2"]
        + 0.8 * features["competitor_atm_ring2"]
    )
    if features["dist_vtb_atm_km"].isna().all():
        features["dist_vtb_atm_km"] = features["dist_vtb_branch_km"]
    if features["dist_supermarket_km"].isna().all():
        features["dist_supermarket_km"] = features["dist_to_city_center_km"]
    features["competitor_pressure"] = minmax(features["competitor_atm_ring2"].fillna(0))
    features["vtb_network_density"] = minmax(
        features["vtb_atm_ring2"].fillna(0) + features["vtb_branch_ring2"].fillna(0)
    )
    features["dist_vtb_service_km"] = features[["dist_vtb_atm_km", "dist_vtb_branch_km"]].min(axis=1)
    features["network_gap_score"] = minmax(features["dist_vtb_service_km"].fillna(features["dist_to_city_center_km"]))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ARTIFACTS.external_summary, index=False)
    return features, summary


def build_feature_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs([PROCESSED_DIR])
    data, target = load_case_tables()
    features = _build_internal_features(data, target)
    features = add_spatial_features(features)
    features, external_summary = build_external_features(features)
    features["source_confidence"] = "official_case_data"
    features["source_confidence_external"] = "community_open_data"
    features.to_parquet(ARTIFACTS.features, index=False)
    return features, external_summary
