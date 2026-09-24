"""Evaluation, calibration, and capacity-based risk tiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
    total = len(y_true)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return error if total else float("nan")


def top_fraction_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    count = max(1, int(np.ceil(len(y_true) * fraction)))
    base_rate = float(y_true.mean())
    if float(np.ptp(probability)) <= 1e-12:
        return {
            f"precision_at_{int(fraction * 100)}pct": base_rate,
            f"recall_at_{int(fraction * 100)}pct": float(fraction),
            f"lift_at_{int(fraction * 100)}pct": 1.0,
        }
    selected = np.argsort(-probability)[:count]
    selected_rate = float(y_true[selected].mean())
    return {
        f"precision_at_{int(fraction * 100)}pct": selected_rate,
        f"recall_at_{int(fraction * 100)}pct": float(y_true[selected].sum() / max(y_true.sum(), 1)),
        f"lift_at_{int(fraction * 100)}pct": selected_rate / base_rate if base_rate > 0 else float("nan"),
    }


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "rows": len(y),
        "late_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "ece_10_bins": expected_calibration_error(y, probability, bins=10),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    metrics.update(top_fraction_metrics(y, probability, 0.10))
    metrics.update(top_fraction_metrics(y, probability, 0.30))
    return metrics


def select_f2_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return 0.5
    numerator = 5 * precision[:-1] * recall[:-1]
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(thresholds[int(np.nanargmax(f2))])


@dataclass(slots=True)
class PlattCalibrator:
    coefficient: float
    intercept: float

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped))

    @classmethod
    def fit(cls, raw_probability: np.ndarray, y_true: pd.Series | np.ndarray) -> PlattCalibrator:
        scores = cls._logit(raw_probability).reshape(-1, 1)
        model = LogisticRegression(C=1_000_000.0, solver="lbfgs")
        model.fit(scores, np.asarray(y_true, dtype=int))
        return cls(coefficient=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))

    def transform(self, raw_probability: np.ndarray) -> np.ndarray:
        linear = self.coefficient * self._logit(raw_probability) + self.intercept
        calibrated = 1.0 / (1.0 + np.exp(-linear))
        return np.clip(calibrated, 1e-9, 1 - 1e-9)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    high: float
    medium: float
    high_capacity: float
    medium_capacity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_capacity_thresholds(
    calibrated_probability: np.ndarray,
    high_capacity: float = 0.10,
    medium_capacity: float = 0.20,
) -> RiskThresholds:
    if high_capacity <= 0 or medium_capacity <= 0 or high_capacity + medium_capacity >= 1:
        raise ValueError("Risk capacities must be positive and sum to less than one")
    probability = np.asarray(calibrated_probability, dtype=float)
    return RiskThresholds(
        high=float(np.quantile(probability, 1.0 - high_capacity)),
        medium=float(np.quantile(probability, 1.0 - high_capacity - medium_capacity)),
        high_capacity=float(high_capacity),
        medium_capacity=float(medium_capacity),
    )


def assign_risk_tier(probability: np.ndarray, thresholds: RiskThresholds) -> pd.Categorical:
    values = np.select(
        [probability >= thresholds.high, probability >= thresholds.medium],
        ["High", "Medium"],
        default="Low",
    )
    return pd.Categorical(values, categories=["Low", "Medium", "High"], ordered=True)


def assign_capacity_tiers(
    probability: np.ndarray,
    high_capacity: float = 0.10,
    medium_capacity: float = 0.20,
) -> tuple[pd.Categorical, RiskThresholds]:
    """Assign an exact batch-relative review capacity without consulting outcomes."""

    if high_capacity <= 0 or medium_capacity <= 0 or high_capacity + medium_capacity >= 1:
        raise ValueError("Risk capacities must be positive and sum to less than one")
    probability = np.asarray(probability, dtype=float)
    high_count = int(round(len(probability) * high_capacity))
    medium_count = int(round(len(probability) * medium_capacity))
    order = np.argsort(-probability, kind="stable")
    values = np.full(len(probability), "Low", dtype=object)
    values[order[:high_count]] = "High"
    values[order[high_count : high_count + medium_count]] = "Medium"
    thresholds = RiskThresholds(
        high=float(probability[order[high_count - 1]]) if high_count else 1.0,
        medium=float(probability[order[high_count + medium_count - 1]]) if medium_count else 1.0,
        high_capacity=float(high_capacity),
        medium_capacity=float(medium_capacity),
    )
    tiers = pd.Categorical(values, categories=["Low", "Medium", "High"], ordered=True)
    return tiers, thresholds


def risk_tier_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("risk_tier", observed=False)
        .agg(
            orders=("order_id", "size"),
            share=("order_id", lambda values: len(values) / len(predictions)),
            mean_predicted_risk=("late_probability", "mean"),
            observed_late_rate=("is_late", "mean"),
            late_orders=("is_late", "sum"),
        )
        .reset_index()
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
