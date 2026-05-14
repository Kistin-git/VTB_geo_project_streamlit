from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vtb_geo_project.config import ARTIFACTS, PROCESSED_DIR
from vtb_geo_project.data_pipeline import build_feature_table
from vtb_geo_project.modeling import build_local_explanations, train_models
from vtb_geo_project.optimization import build_recommendations, score_candidates
from vtb_geo_project.reporting import write_solution_report
from vtb_geo_project.utils import ensure_dirs, json_dump


def export_game_artifacts(features: pd.DataFrame, recommendations: pd.DataFrame, game_repo: Path) -> None:
    game_data = game_repo / "data"
    ensure_dirs([game_data])
    scenario_best = (
        recommendations[recommendations["is_before_optimal_n"]]
        .sort_values(["scenario", "selected_rank"])
        .groupby("scenario")
        .head(1)
        .reset_index(drop=True)
    )
    feature_cols = [
        "h3_index",
        "lat",
        "lon",
        "composite_model_score",
        "profit_core_score",
        "coverage_core_score",
        "social_core_score",
        "competitor_core_score",
        "business_core_score",
        "dist_vtb_service_km",
    ]
    features[feature_cols].to_parquet(game_data / "game_cells.parquet", index=False)
    recommendations.to_parquet(game_data / "game_recommendations.parquet", index=False)
    scenario_best.to_parquet(game_data / "game_best_by_scenario.parquet", index=False)
    scenario_frames = []
    for scenario in recommendations["scenario"].unique():
        scenario_type_map = (
            recommendations[recommendations["scenario"] == scenario][["h3_index", "recommended_atm_type"]]
            .drop_duplicates("h3_index")
            .set_index("h3_index")["recommended_atm_type"]
        )
        scenario_frame = features[feature_cols].copy()
        scenario_frame["scenario"] = scenario
        scenario_frame["recommended_atm_type"] = scenario_frame["h3_index"].map(scenario_type_map).fillna(
            "full_service_cash_in_qr"
        )
        scenario_frames.append(scenario_frame)
    pd.concat(scenario_frames, ignore_index=True).to_parquet(game_data / "game_cells_by_scenario.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-repo", type=Path, default=None)
    args = parser.parse_args()

    ensure_dirs([PROCESSED_DIR])
    features, _ = build_feature_table()
    model_bundle, scored_features, metrics = train_models(features)
    build_local_explanations(model_bundle, scored_features)
    candidates = score_candidates(scored_features)
    recommendations = build_recommendations(candidates)
    write_solution_report(scored_features, recommendations, metrics)

    if args.game_repo:
        export_game_artifacts(candidates, recommendations, args.game_repo.resolve())

    print(f"Features: {ARTIFACTS.features}")
    print(f"Recommendations: {ARTIFACTS.recommendations}")
    print(f"Metrics: {ARTIFACTS.metrics}")
    print(f"Report: {ARTIFACTS.report}")


if __name__ == "__main__":
    main()
