"""Descriptive statistics and post-outcome customer-impact analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .contracts import PREDICTION_TIME, TARGET


def review_impact_analysis(
    frame: pd.DataFrame,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, float | int | str]:
    late = frame.loc[frame[TARGET].eq(1), "review_score"].dropna().to_numpy(float)
    on_time = frame.loc[frame[TARGET].eq(0), "review_score"].dropna().to_numpy(float)
    if len(late) == 0 or len(on_time) == 0:
        raise ValueError("Review impact analysis requires both late and on-time observations")

    result = mannwhitneyu(late, on_time, alternative="two-sided", method="asymptotic")
    rank_biserial = 2.0 * float(result.statistic) / (len(late) * len(on_time)) - 1.0
    observed_mean_difference = float(late.mean() - on_time.mean())

    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(bootstrap_repetitions, dtype=float)
    for index in range(bootstrap_repetitions):
        differences[index] = float(
            rng.choice(late, size=len(late), replace=True).mean()
            - rng.choice(on_time, size=len(on_time), replace=True).mean()
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_value = float(result.pvalue)
    return {
        "late_review_count": len(late),
        "on_time_review_count": len(on_time),
        "late_review_mean": float(late.mean()),
        "on_time_review_mean": float(on_time.mean()),
        "late_review_median": float(np.median(late)),
        "on_time_review_median": float(np.median(on_time)),
        "mann_whitney_u": float(result.statistic),
        "p_value": p_value,
        "p_value_display": "<1e-300" if p_value == 0.0 else f"{p_value:.3e}",
        "rank_biserial_correlation": rank_biserial,
        "mean_difference_late_minus_on_time": observed_mean_difference,
        "mean_difference_ci95_lower": float(lower),
        "mean_difference_ci95_upper": float(upper),
        "interpretation": "Post-outcome association; not a causal effect estimate.",
    }


def monthly_outcome_drift(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = frame.copy()
    monthly["approval_month"] = monthly[PREDICTION_TIME].dt.to_period("M").astype(str)
    return (
        monthly.groupby("approval_month", as_index=False)
        .agg(
            orders=("order_id", "size"),
            late_orders=(TARGET, "sum"),
            late_rate=(TARGET, "mean"),
            mean_delay_days=("delay_days", "mean"),
        )
        .sort_values("approval_month")
    )
