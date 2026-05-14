from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CASE_ARCHIVE_CANDIDATES = [
    RAW_DIR / "case_bundle.zip",
    ROOT.parent / "Данные для продвинутого анализа-20260514T092848Z-3-001.zip",
    ROOT.parent.parent / "Данные для продвинутого анализа-20260514T092848Z-3-001.zip",
]

CASE_EXTRACT_DIR = RAW_DIR / "case"
CASE_DATA_FILE = CASE_EXTRACT_DIR / "data.parquet"
CASE_TARGET_FILE = CASE_EXTRACT_DIR / "target.parquet"

OSM_SHAPE_URL = "https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.shp.zip"
OSM_SHAPE_ZIP = EXTERNAL_DIR / "Moscow.osm.shp.zip"
OSM_SHAPE_DIR = EXTERNAL_DIR / "Moscow.osm.shp"

MOSCOW_CENTER_LAT = 55.7558
MOSCOW_CENTER_LON = 37.6176
H3_RESOLUTION = 9
TOP_K = 150
SPATIAL_FOLDS = 5

VTB_TEXT_TOKENS = ("втб", "vtb")
COMPETITOR_TEXT_TOKENS = (
    "sber",
    "сбер",
    "alfa",
    "альфа",
    "tinkoff",
    "тинько",
    "gazprom",
    "газпром",
    "raiffeisen",
    "райфф",
    "rosselkhoz",
    "россельхоз",
    "pochta",
    "почта",
    "otkritie",
    "открытие",
)


ATM_TYPE_PARAMS = {
    "standard_cash_out": {
        "cost": 0.80,
        "revenue_mult": 0.95,
        "coverage_mult": 1.00,
        "social_mult": 0.90,
        "business_mult": 0.60,
        "high_flow_mult": 0.85,
    },
    "full_service_cash_in_qr": {
        "cost": 1.00,
        "revenue_mult": 1.00,
        "coverage_mult": 1.08,
        "social_mult": 1.00,
        "business_mult": 0.90,
        "high_flow_mult": 1.00,
    },
    "high_flow_24_7": {
        "cost": 1.22,
        "revenue_mult": 1.20,
        "coverage_mult": 0.96,
        "social_mult": 0.72,
        "business_mult": 0.75,
        "high_flow_mult": 1.30,
    },
    "business_cash_in": {
        "cost": 1.12,
        "revenue_mult": 1.05,
        "coverage_mult": 0.88,
        "social_mult": 0.66,
        "business_mult": 1.35,
        "high_flow_mult": 0.92,
    },
}


SCENARIO_WEIGHTS = {
    "balanced": {"profit": 0.42, "coverage": 0.24, "social": 0.10, "competitor": 0.10, "business": 0.06, "cannibalization": 0.18, "risk": 0.10},
    "profit": {"profit": 0.60, "coverage": 0.12, "social": 0.04, "competitor": 0.12, "business": 0.08, "cannibalization": 0.18, "risk": 0.12},
    "coverage": {"profit": 0.24, "coverage": 0.44, "social": 0.08, "competitor": 0.06, "business": 0.04, "cannibalization": 0.16, "risk": 0.12},
    "social": {"profit": 0.16, "coverage": 0.20, "social": 0.48, "competitor": 0.04, "business": 0.02, "cannibalization": 0.12, "risk": 0.10},
    "competitor": {"profit": 0.32, "coverage": 0.14, "social": 0.04, "competitor": 0.36, "business": 0.04, "cannibalization": 0.16, "risk": 0.10},
    "business": {"profit": 0.34, "coverage": 0.08, "social": 0.04, "competitor": 0.08, "business": 0.38, "cannibalization": 0.14, "risk": 0.10},
}


POI_CATEGORY_RULES = {
    "atm_any": (" atm ", "amenity=atm", "fclass=atm", "atm"),
    "school": ("amenity=school", " school ", "школ"),
    "university": ("amenity=university", " college ", "универс", "вуз", "institute"),
    "hospital": ("amenity=hospital", "amenity=clinic", " clinic ", "hospital", "больниц", "поликлиник", "медицин"),
    "mall": ("shop=mall", " mall ", "торгов"),
    "supermarket": ("shop=supermarket", " hypermarket ", "супермар"),
    "office": ("office=", "business_centre", "building=office", "офис"),
    "metro": ("subway", "station", "railway=station", "public_transport=station", "metro"),
}


@dataclass(frozen=True)
class ArtifactPaths:
    features: Path = PROCESSED_DIR / "cell_features.parquet"
    candidates: Path = PROCESSED_DIR / "candidate_scores.parquet"
    recommendations: Path = PROCESSED_DIR / "recommendations.parquet"
    metrics: Path = PROCESSED_DIR / "model_metrics.json"
    feature_importance: Path = PROCESSED_DIR / "feature_importance.csv"
    explanations: Path = PROCESSED_DIR / "local_explanations.json"
    summary: Path = PROCESSED_DIR / "dashboard_summary.json"
    external_summary: Path = PROCESSED_DIR / "external_source_summary.csv"
    report: Path = REPORTS_DIR / "solution_report.md"


ARTIFACTS = ArtifactPaths()

