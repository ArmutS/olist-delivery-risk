"""Out-of-sample segment diagnostics and within-family CRITIC priorities."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_FAMILIES = {
    "product_category": ["dominant_product_category"],
    "customer_state": ["customer_state"],
    "seller_state": ["dominant_seller_state"],
    "route": ["dominant_seller_state", "customer_state"],
}

CRITERIA = [
    "order_volume",
    "expected_late_orders",
    "smoothed_late_rate",
    "mean_positive_delay_days",
    "review_penalty",
]


def _segment_label(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]].fillna("unknown").astype(str)
    return frame[columns].fillna("unknown").astype(str).agg(" -> ".join, axis=1)


def build_segment_metrics(
    predictions: pd.DataFrame,
    family: str,
    columns: list[str],
    minimum_orders: int = 50,
    prior_strength: float = 50.0,
) -> pd.DataFrame:
    local = predictions.copy()
    local["segment"] = _segment_label(local, columns)
    local["positive_delay_days"] = local["delay_days"].where(local["is_late"].eq(1))
    local["late_review"] = local["review_score"].where(local["is_late"].eq(1))
    local["on_time_review"] = local["review_score"].where(local["is_late"].eq(0))
    base_rate = float(local["is_late"].mean())
    global_positive_delay = float(local["positive_delay_days"].mean())
    global_review_penalty = max(
        float(local["on_time_review"].mean() - local["late_review"].mean()),
        0.0,
    )
    metrics = local.groupby("segment", as_index=False).agg(
        order_volume=("order_id", "size"),
        observed_late_orders=("is_late", "sum"),
        observed_late_rate=("is_late", "mean"),
        expected_late_orders=("late_probability", "sum"),
        mean_predicted_risk=("late_probability", "mean"),
        mean_positive_delay_days_raw=("positive_delay_days", "mean"),
        positive_delay_observations=("positive_delay_days", "count"),
        late_review_mean=("late_review", "mean"),
        late_review_observations=("late_review", "count"),
        on_time_review_mean=("on_time_review", "mean"),
    )
    metrics = metrics[metrics["order_volume"] >= minimum_orders].copy()
    metrics["smoothed_late_rate"] = (
        metrics["observed_late_orders"] + prior_strength * base_rate
    ) / (metrics["order_volume"] + prior_strength)
    severity_prior_strength = 10.0
    metrics["mean_positive_delay_days"] = (
        metrics["positive_delay_observations"]
        * metrics["mean_positive_delay_days_raw"].fillna(global_positive_delay)
        + severity_prior_strength * global_positive_delay
    ) / (metrics["positive_delay_observations"] + severity_prior_strength)
    metrics["review_penalty_raw"] = (
        metrics["on_time_review_mean"] - metrics["late_review_mean"]
    ).clip(lower=0)
    review_prior_strength = 10.0
    metrics["review_penalty"] = (
        metrics["late_review_observations"]
        * metrics["review_penalty_raw"].fillna(global_review_penalty)
        + review_prior_strength * global_review_penalty
    ) / (metrics["late_review_observations"] + review_prior_strength)
    metrics.insert(0, "segment_family", family)
    return metrics


def _minmax(series: pd.Series) -> pd.Series:
    lower = series.min()
    upper = series.max()
    if pd.isna(lower) or pd.isna(upper) or np.isclose(lower, upper):
        return pd.Series(0.0, index=series.index)
    return (series - lower) / (upper - lower)


def critic_score(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return metrics.copy(), pd.DataFrame(columns=["criterion", "weight"])
    matrix = metrics[CRITERIA].copy()
    matrix = matrix.fillna(0.0)
    normalized = matrix.apply(_minmax)
    standard_deviation = normalized.std(ddof=0)
    correlation = normalized.corr().fillna(0.0)
    information = standard_deviation * (1.0 - correlation).sum(axis=1)
    if np.isclose(information.sum(), 0.0):
        weights = pd.Series(1.0 / len(CRITERIA), index=CRITERIA)
    else:
        weights = information / information.sum()

    scored = metrics.copy()
    for criterion in CRITERIA:
        scored[f"normalized_{criterion}"] = normalized[criterion]
    scored["priority_score"] = sum(normalized[name] * weights[name] for name in CRITERIA)
    scored = scored.sort_values("priority_score", ascending=False).reset_index(drop=True)
    scored.insert(1, "priority_rank", np.arange(1, len(scored) + 1))
    weight_frame = pd.DataFrame(
        {
            "criterion": CRITERIA,
            "standard_deviation": standard_deviation.reindex(CRITERIA).to_numpy(),
            "information": information.reindex(CRITERIA).to_numpy(),
            "weight": weights.reindex(CRITERIA).to_numpy(),
        }
    )
    return scored, weight_frame


def build_all_priorities(
    predictions: pd.DataFrame,
    minimum_orders: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_frames = []
    weight_frames = []
    for family, columns in SEGMENT_FAMILIES.items():
        metrics = build_segment_metrics(predictions, family, columns, minimum_orders)
        scored, weights = critic_score(metrics)
        if not scored.empty:
            scored_frames.append(scored)
            weights.insert(0, "segment_family", family)
            weight_frames.append(weights)
    return (
        pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame(),
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame(),
    )
