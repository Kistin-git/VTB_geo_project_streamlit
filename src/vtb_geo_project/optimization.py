from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ATM_TYPE_PARAMS, ARTIFACTS, SCENARIO_WEIGHTS
from .utils import haversine_km, json_dump, minmax


@dataclass
class ScenarioResult:
    scenario: str
    selected: pd.DataFrame


def score_candidates(features: pd.DataFrame) -> pd.DataFrame:
    candidates = features.copy()
    candidates["service_gap_score"] = minmax(candidates["dist_vtb_service_km"].fillna(candidates["dist_to_city_center_km"]))
    candidates["demand_score"] = (
        0.65 * minmax(candidates["pred_atm_users_count"])
        + 0.35 * minmax(candidates["pred_atm_presence_prob"])
    )
    candidates["profit_core_score"] = (
        0.50 * minmax(candidates["log_tx_sum"])
        + 0.25 * minmax(candidates["demand_score"])
        + 0.15 * minmax(candidates["competitor_pressure"])
        + 0.10 * minmax(candidates["business_intensity"])
    )
    candidates["social_core_score"] = minmax(candidates["social_intensity"])
    candidates["coverage_core_score"] = minmax(candidates["service_gap_score"] + (1 - candidates["vtb_network_density"]))
    candidates["business_core_score"] = minmax(candidates["business_intensity"])
    candidates["competitor_core_score"] = minmax(candidates["competitor_pressure"] + candidates["service_gap_score"])
    candidates["risk_score"] = minmax(candidates["vtb_network_density"] + candidates["dist_to_city_center_km"])

    for atm_type, params in ATM_TYPE_PARAMS.items():
        candidates[f"{atm_type}_base_profit"] = (
            candidates["profit_core_score"] * params["revenue_mult"]
            + candidates["coverage_core_score"] * params["coverage_mult"] * 0.25
            + candidates["social_core_score"] * params["social_mult"] * 0.12
            + candidates["business_core_score"] * params["business_mult"] * 0.18
            + candidates["high_flow_intensity"] * params["high_flow_mult"] * 0.02
            - params["cost"] * 0.45
        )
    candidates.to_parquet(ARTIFACTS.candidates, index=False)
    return candidates


def _pair_distance_km(row_a: pd.Series, row_b: pd.Series) -> float:
    return haversine_km(row_a["lat"], row_a["lon"], row_b["lat"], row_b["lon"])


def _best_atm_type(row: pd.Series, scenario: str) -> str:
    weights = SCENARIO_WEIGHTS[scenario]
    best_type = "full_service_cash_in_qr"
    best_score = -1e9
    for atm_type in ATM_TYPE_PARAMS:
        score = row[f"{atm_type}_base_profit"]
        score += weights["social"] * row["social_core_score"] * ATM_TYPE_PARAMS[atm_type]["social_mult"]
        score += weights["coverage"] * row["coverage_core_score"] * ATM_TYPE_PARAMS[atm_type]["coverage_mult"]
        score += weights["business"] * row["business_core_score"] * ATM_TYPE_PARAMS[atm_type]["business_mult"]
        if score > best_score:
            best_type = atm_type
            best_score = score
    return best_type


def _marginal_objective(row: pd.Series, scenario: str, selected: list[pd.Series]) -> float:
    weights = SCENARIO_WEIGHTS[scenario]
    atm_type = row["recommended_atm_type"]
    cannibalization_existing = np.exp(-row["dist_vtb_service_km"] / 0.75) if pd.notna(row["dist_vtb_service_km"]) else 0.0
    cannibalization_new = 0.0
    for item in selected:
        distance = _pair_distance_km(row, item)
        cannibalization_new = max(cannibalization_new, float(np.exp(-distance / 0.8)))

    # Penalty for network growth creates a realistic interior optimum for N:
    # each next ATM is slightly less attractive than the previous one even if
    # local cell quality stays high.
    selected_count = len(selected) + 1
    portfolio_expansion_penalty = 0.01 * selected_count + 0.00025 * selected_count**2

    return float(
        weights["profit"] * row[f"{atm_type}_base_profit"]
        + weights["coverage"] * row["coverage_core_score"]
        + weights["social"] * row["social_core_score"]
        + weights["competitor"] * row["competitor_core_score"]
        + weights["business"] * row["business_core_score"]
        - weights["cannibalization"] * (cannibalization_existing + cannibalization_new)
        - weights["risk"] * row["risk_score"]
        - portfolio_expansion_penalty
    )


def optimize_scenario(candidates: pd.DataFrame, scenario: str, max_new_atm: int = 80, min_distance_km: float = 0.55) -> pd.DataFrame:
    frame = candidates.copy()
    frame["recommended_atm_type"] = frame.apply(lambda row: _best_atm_type(row, scenario), axis=1)
    frame["scenario"] = scenario
    frame["selected_rank"] = np.nan
    selected_rows: list[pd.Series] = []
    selected_indices: list[int] = []
    cumulative = 0.0

    eligible = frame.sort_values("composite_model_score", ascending=False)
    for idx, row in eligible.iterrows():
        too_close = False
        for selected in selected_rows:
            if _pair_distance_km(row, selected) < min_distance_km:
                too_close = True
                break
        if too_close:
            continue
        marginal = _marginal_objective(row, scenario, selected_rows)
        if marginal <= 0 and len(selected_rows) >= 8:
            continue
        row = row.copy()
        cumulative += marginal
        row["marginal_objective"] = marginal
        row["cumulative_objective"] = cumulative
        row["selected_rank"] = len(selected_rows) + 1
        row["objective_breakdown"] = (
            f"profit={row['profit_core_score']:.2f}, coverage={row['coverage_core_score']:.2f}, "
            f"social={row['social_core_score']:.2f}, competitor={row['competitor_core_score']:.2f}, "
            f"business={row['business_core_score']:.2f}"
        )
        selected_rows.append(row)
        selected_indices.append(idx)
        if len(selected_rows) >= max_new_atm:
            break

    if not selected_rows:
        return frame.head(0).copy()

    selected = pd.DataFrame(selected_rows)
    best_idx = selected["cumulative_objective"].idxmax()
    optimal_rank = int(selected.loc[best_idx, "selected_rank"])
    selected["is_before_optimal_n"] = selected["selected_rank"] <= optimal_rank
    selected["optimal_n_for_scenario"] = optimal_rank
    return selected


def build_recommendations(candidates: pd.DataFrame) -> pd.DataFrame:
    scenario_frames = [optimize_scenario(candidates, scenario) for scenario in SCENARIO_WEIGHTS]
    recommendations = pd.concat(scenario_frames, ignore_index=True, sort=False)
    recommendations.to_parquet(ARTIFACTS.recommendations, index=False)
    summary = {
        scenario: {
            "optimal_n": int(group["optimal_n_for_scenario"].max()),
            "best_cumulative_objective": float(group["cumulative_objective"].max()),
            "top_cell": group.sort_values("selected_rank").iloc[0]["h3_index"],
        }
        for scenario, group in recommendations.groupby("scenario")
    }
    json_dump(ARTIFACTS.summary, summary)
    return recommendations
