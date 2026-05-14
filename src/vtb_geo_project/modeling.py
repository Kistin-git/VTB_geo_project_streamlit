from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS, MODELS_DIR, SPATIAL_FOLDS, TOP_K
from .utils import ensure_dirs, json_dump, minmax, rank_normalize


@dataclass
class ModelBundle:
    classifier: CatBoostClassifier
    regressor: CatBoostRegressor
    baseline_classifier: Pipeline
    baseline_regressor: Pipeline
    feature_columns: list[str]
    feature_importance: pd.DataFrame


def _select_feature_columns(features: pd.DataFrame) -> list[str]:
    exclude = {
        "h3_index",
        "atm_presence",
        "atm_users_count",
        "log_atm_users_count",
        "source_confidence",
        "source_confidence_external",
    }
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    selected = [col for col in numeric_cols if col not in exclude]
    return [col for col in selected if not features[col].isna().all()]


def _spatial_groups(features: pd.DataFrame, n_splits: int = SPATIAL_FOLDS) -> np.ndarray:
    coords = features[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=20)
    return kmeans.fit_predict(coords)


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    preds = np.zeros_like(y_true)
    preds[order] = 1
    return float(precision_score(y_true, preds, zero_division=0))


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    preds = np.zeros_like(y_true)
    preds[order] = 1
    return float(recall_score(y_true, preds, zero_division=0))


def _inverse_log_prediction(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(values, a_min=-8.0, a_max=8.0))


def _evaluate_models(features: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    X = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_clf = features["atm_presence"].astype(int).to_numpy()
    y_reg = features["atm_users_count"].astype(float).to_numpy()
    groups = _spatial_groups(features)
    splitter = GroupKFold(n_splits=SPATIAL_FOLDS)

    clf_pred = np.zeros(len(features))
    reg_pred = np.zeros(len(features))
    base_clf_pred = np.zeros(len(features))
    base_reg_pred = np.zeros(len(features))

    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_clf, groups), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train_clf = y_clf[train_idx]
        y_valid_clf = y_clf[valid_idx]
        y_train_reg = y_reg[train_idx]
        y_valid_reg = y_reg[valid_idx]

        baseline_clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        baseline_reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        cat_clf = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=42,
        )
        cat_reg = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )

        baseline_clf.fit(X_train, y_train_clf)
        baseline_reg.fit(X_train, np.log1p(y_train_reg))
        cat_clf.fit(X_train, y_train_clf)
        cat_reg.fit(X_train, np.log1p(y_train_reg))

        base_clf_pred[valid_idx] = baseline_clf.predict_proba(X_valid)[:, 1]
        base_reg_pred[valid_idx] = _inverse_log_prediction(baseline_reg.predict(X_valid))
        clf_pred[valid_idx] = cat_clf.predict_proba(X_valid)[:, 1]
        reg_pred[valid_idx] = np.expm1(cat_reg.predict(X_valid))

        row = {
            "fold": fold,
            "clf_auc": roc_auc_score(y_valid_clf, clf_pred[valid_idx]),
            "clf_ap": average_precision_score(y_valid_clf, clf_pred[valid_idx]),
            "clf_precision_at_k": _precision_at_k(y_valid_clf, clf_pred[valid_idx], min(TOP_K, len(valid_idx))),
            "clf_recall_at_k": _recall_at_k(y_valid_clf, clf_pred[valid_idx], min(TOP_K, len(valid_idx))),
            "reg_rmse": mean_squared_error(y_valid_reg, reg_pred[valid_idx], squared=False),
            "reg_mae": mean_absolute_error(y_valid_reg, reg_pred[valid_idx]),
            "reg_ndcg": ndcg_score([y_valid_reg], [reg_pred[valid_idx]], k=min(TOP_K, len(valid_idx))),
        }
        fold_rows.append(row)

    metrics = {
        "folds": fold_rows,
        "mean": {
            "catboost_auc": float(np.mean([row["clf_auc"] for row in fold_rows])),
            "catboost_ap": float(np.mean([row["clf_ap"] for row in fold_rows])),
            "catboost_precision_at_k": float(np.mean([row["clf_precision_at_k"] for row in fold_rows])),
            "catboost_recall_at_k": float(np.mean([row["clf_recall_at_k"] for row in fold_rows])),
            "catboost_reg_rmse": float(np.mean([row["reg_rmse"] for row in fold_rows])),
            "catboost_reg_mae": float(np.mean([row["reg_mae"] for row in fold_rows])),
            "catboost_reg_ndcg": float(np.mean([row["reg_ndcg"] for row in fold_rows])),
            "baseline_auc": float(roc_auc_score(y_clf, base_clf_pred)),
            "baseline_ap": float(average_precision_score(y_clf, base_clf_pred)),
            "baseline_reg_rmse": float(mean_squared_error(y_reg, base_reg_pred, squared=False)),
            "baseline_reg_mae": float(mean_absolute_error(y_reg, base_reg_pred)),
        },
    }
    predictions = {
        "clf_pred": clf_pred,
        "reg_pred": np.clip(reg_pred, a_min=0.0, a_max=None),
        "base_clf_pred": base_clf_pred,
        "base_reg_pred": np.clip(base_reg_pred, a_min=0.0, a_max=None),
    }
    return metrics, predictions


def train_models(features: pd.DataFrame) -> tuple[ModelBundle, pd.DataFrame, dict]:
    ensure_dirs([MODELS_DIR])
    feature_columns = _select_feature_columns(features)
    metrics, predictions = _evaluate_models(features, feature_columns)

    X = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_clf = features["atm_presence"].astype(int)
    y_reg = features["atm_users_count"].astype(float)

    baseline_clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    baseline_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    baseline_clf.fit(X, y_clf)
    baseline_reg.fit(X, np.log1p(y_reg))

    classifier = CatBoostClassifier(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
    )
    regressor = CatBoostRegressor(
        iterations=450,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    classifier.fit(X, y_clf)
    regressor.fit(X, np.log1p(y_reg))

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": classifier.get_feature_importance(Pool(X, y_clf)),
        }
    ).sort_values("importance", ascending=False)
    importance["importance_norm"] = importance["importance"] / importance["importance"].sum()
    importance.to_csv(ARTIFACTS.feature_importance, index=False)

    features = features.copy()
    features["pred_atm_presence_prob"] = classifier.predict_proba(X)[:, 1]
    features["pred_atm_users_count"] = np.clip(np.expm1(regressor.predict(X)), a_min=0.0, a_max=None)
    features["baseline_presence_prob"] = baseline_clf.predict_proba(X)[:, 1]
    features["baseline_atm_users_count"] = _inverse_log_prediction(baseline_reg.predict(X))
    features["composite_model_score"] = (
        0.58 * rank_normalize(features["pred_atm_presence_prob"])
        + 0.42 * rank_normalize(features["pred_atm_users_count"])
    )
    features["model_uplift_vs_baseline"] = features["composite_model_score"] - (
        0.58 * rank_normalize(features["baseline_presence_prob"])
        + 0.42 * rank_normalize(features["baseline_atm_users_count"])
    )

    model_bundle = ModelBundle(
        classifier=classifier,
        regressor=regressor,
        baseline_classifier=baseline_clf,
        baseline_regressor=baseline_reg,
        feature_columns=feature_columns,
        feature_importance=importance,
    )
    classifier.save_model(MODELS_DIR / "atm_presence_classifier.cbm")
    regressor.save_model(MODELS_DIR / "atm_demand_regressor.cbm")
    joblib.dump({"features": feature_columns}, MODELS_DIR / "metadata.pkl")
    json_dump(ARTIFACTS.metrics, metrics)
    return model_bundle, features, metrics


def build_local_explanations(model_bundle: ModelBundle, scored_features: pd.DataFrame, top_n: int = 300) -> list[dict]:
    feature_columns = model_bundle.feature_columns
    top_cells = scored_features.sort_values("composite_model_score", ascending=False).head(top_n).copy()
    shap_values = model_bundle.classifier.get_feature_importance(
        Pool(top_cells[feature_columns].replace([np.inf, -np.inf], np.nan)),
        type="ShapValues",
    )
    feature_names = feature_columns
    rows: list[dict] = []
    for idx, (_, row) in enumerate(top_cells.iterrows()):
        shap_row = shap_values[idx][:-1]
        frame = pd.DataFrame({"feature": feature_names, "shap": shap_row})
        frame["abs_shap"] = frame["shap"].abs()
        frame = frame.sort_values("abs_shap", ascending=False).head(3)
        rows.append(
            {
                "h3_index": row["h3_index"],
                "reasons": [
                    f"{feature} ({value:+.3f})" for feature, value in zip(frame["feature"], frame["shap"])
                ],
            }
        )
    json_dump(ARTIFACTS.explanations, rows)
    return rows
