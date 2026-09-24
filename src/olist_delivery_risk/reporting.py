"""Persist compact portfolio artifacts and diagnostic figures."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "olist-risk-matplotlib"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from .contracts import MODEL_FEATURES, TARGET
from .evaluation import json_safe
from .modeling import FinalModelResult, chronological_split, evaluate_feature_ablation
from .prioritization import build_all_priorities
from .statistics import monthly_outcome_drift, review_impact_analysis

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    ordered = comparison.sort_values("average_precision")
    colors = ["#9CA3AF" if name == "dummy_prior" else "#2563EB" for name in ordered["model"]]
    plt.figure(figsize=(8, 4.6))
    plt.barh(ordered["model"], ordered["average_precision"], color=colors)
    plt.xlabel("Average precision on chronological selection period")
    plt.title("Model selection against prevalence baseline")
    for index, value in enumerate(ordered["average_precision"]):
        plt.text(value + 0.004, index, f"{value:.3f}", va="center")
    plt.xlim(0, max(ordered["average_precision"].max() * 1.18, 0.1))
    _save_figure(path)


def _calibration_bins(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    local = predictions[[TARGET, "late_probability"]].copy()
    local["bin"] = pd.qcut(local["late_probability"], q=bins, duplicates="drop")
    return local.groupby("bin", observed=False).agg(
        mean_probability=("late_probability", "mean"),
        observed_rate=(TARGET, "mean"),
        orders=(TARGET, "size"),
    ).reset_index(drop=True)


def plot_calibration(predictions: pd.DataFrame, path: Path) -> None:
    calibration = _calibration_bins(predictions)
    upper = max(calibration["mean_probability"].max(), calibration["observed_rate"].max()) * 1.08
    plt.figure(figsize=(5.6, 5.2))
    plt.plot([0, upper], [0, upper], linestyle="--", color="#6B7280", label="Perfect calibration")
    plt.plot(
        calibration["mean_probability"],
        calibration["observed_rate"],
        marker="o",
        color="#2563EB",
        linewidth=2,
        label="Calibrated model",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed late-order rate")
    plt.title("Calibration on untouched test period")
    plt.legend()
    _save_figure(path)


def plot_precision_recall(predictions: pd.DataFrame, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(predictions[TARGET], predictions["late_probability"])
    baseline = float(predictions[TARGET].mean())
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(recall, precision, color="#2563EB", linewidth=2, label="Calibrated model")
    plt.axhline(baseline, linestyle="--", color="#6B7280", label=f"Prevalence ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curve on test period")
    plt.legend()
    _save_figure(path)


def plot_risk_tiers(summary: pd.DataFrame, path: Path) -> None:
    local = summary.set_index("risk_tier").reindex(["Low", "Medium", "High"])
    x = np.arange(len(local))
    width = 0.36
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width / 2, local["mean_predicted_risk"], width, label="Mean predicted", color="#60A5FA")
    plt.bar(x + width / 2, local["observed_late_rate"], width, label="Observed", color="#F97316")
    plt.xticks(x, local.index)
    plt.ylabel("Rate")
    plt.title("Capacity-based monitoring tiers")
    plt.legend()
    _save_figure(path)


def plot_monthly_drift(monthly: pd.DataFrame, path: Path, minimum_orders: int = 100) -> None:
    local = monthly[monthly["orders"] >= minimum_orders].copy()
    figure, rate_axis = plt.subplots(figsize=(10, 4.8))
    volume_axis = rate_axis.twinx()
    volume_axis.bar(
        local["approval_month"],
        local["orders"],
        color="#CBD5E1",
        alpha=0.55,
        label="Orders",
    )
    rate_axis.plot(
        local["approval_month"],
        local["late_rate"],
        marker="o",
        color="#2563EB",
        linewidth=2,
        label="Late-order rate",
    )
    rate_axis.tick_params(axis="x", rotation=60)
    for label in rate_axis.get_xticklabels():
        label.set_horizontalalignment("right")
    rate_axis.set_ylabel("Observed late-order rate")
    volume_axis.set_ylabel("Monthly orders", color="#64748B")
    volume_axis.tick_params(axis="y", colors="#64748B")
    rate_axis.set_xlabel("Approval month")
    rate_axis.set_title(f"Outcome drift (months with at least {minimum_orders} orders)")
    handles_1, labels_1 = rate_axis.get_legend_handles_labels()
    handles_2, labels_2 = volume_axis.get_legend_handles_labels()
    rate_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
    figure.subplots_adjust(right=0.88)
    _save_figure(path)


def permutation_feature_importance(
    frame: pd.DataFrame,
    result: FinalModelResult,
    proportions: dict[str, float],
    random_state: int,
    sample_size: int = 5000,
) -> pd.DataFrame:
    test = chronological_split(frame, proportions).test
    if len(test) > sample_size:
        test = test.sample(sample_size, random_state=random_state)
    importance = permutation_importance(
        result.model,
        test[MODEL_FEATURES],
        test[TARGET],
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    local = importance.head(top_n).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(local["feature"], local["importance_mean"], xerr=local["importance_std"], color="#2563EB")
    plt.xlabel("Decrease in average precision after permutation")
    plt.title("Out-of-sample permutation importance")
    _save_figure(path)


def write_report_bundle(
    frame: pd.DataFrame,
    result: FinalModelResult,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
    reports_dir: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    reports_dir = Path(reports_dir)
    artifacts_dir = Path(artifacts_dir)
    figures_dir = reports_dir / "figures"
    results_dir = reports_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    review_impact = review_impact_analysis(frame)
    monthly = monthly_outcome_drift(frame)
    priorities, critic_weights = build_all_priorities(
        result.test_predictions,
        minimum_orders=int(config["minimum_segment_orders"]),
    )
    importance = permutation_feature_importance(
        frame,
        result,
        config["split_proportions"],
        int(config["random_state"]),
    )
    ablated_metrics = evaluate_feature_ablation(
        frame,
        result.model_name,
        ["promised_lead_days"],
        config["split_proportions"],
        int(config["random_state"]),
    )
    ablation = pd.DataFrame(
        [
            {"feature_set": "all_approval_time_features", **result.test_metrics},
            {"feature_set": "without_promised_lead_days", **ablated_metrics},
        ]
    )

    _write_json(results_dir / "data_diagnostics.json", diagnostics)
    _write_json(results_dir / "test_metrics.json", result.test_metrics)
    _write_json(results_dir / "calibration.json", {
        "parameters": result.calibrator.to_dict(),
        "raw_calibration_metrics": result.raw_calibration_metrics,
        "calibrated_calibration_metrics": result.calibrated_calibration_metrics,
        "test_batch_tier_thresholds": result.thresholds.to_dict(),
        "decision_threshold": result.decision_threshold,
    })
    _write_json(results_dir / "review_impact.json", review_impact)
    result.split_summary.to_csv(results_dir / "split_summary.csv", index=False)
    result.model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    result.tier_summary.to_csv(results_dir / "risk_tier_summary.csv", index=False)
    monthly.to_csv(results_dir / "monthly_outcome_drift.csv", index=False)
    priorities.to_csv(results_dir / "segment_priorities.csv", index=False)
    critic_weights.to_csv(results_dir / "critic_weights.csv", index=False)
    importance.to_csv(results_dir / "permutation_importance.csv", index=False)
    ablation.to_csv(results_dir / "feature_ablation.csv", index=False)

    result.test_predictions.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "model_name": result.model_name,
            "model": result.model,
            "calibrator": result.calibrator,
            "decision_threshold": result.decision_threshold,
            "risk_capacities": {
                "high": config["high_risk_capacity"],
                "medium": config["medium_risk_capacity"],
            },
            "features": MODEL_FEATURES,
        },
        artifacts_dir / "model_bundle.joblib",
    )

    plot_model_comparison(result.model_comparison, figures_dir / "model_comparison.png")
    plot_calibration(result.test_predictions, figures_dir / "calibration.png")
    plot_precision_recall(result.test_predictions, figures_dir / "precision_recall.png")
    plot_risk_tiers(result.tier_summary, figures_dir / "risk_tiers.png")
    plot_monthly_drift(monthly, figures_dir / "monthly_drift.png")
    plot_feature_importance(importance, figures_dir / "feature_importance.png")

    summary = {
        "selected_model": result.model_name,
        "test_metrics": result.test_metrics,
        "risk_tiers": result.tier_summary.to_dict(orient="records"),
        "review_impact": review_impact,
        "feature_ablation": ablation.to_dict(orient="records"),
    }
    _write_json(reports_dir / "summary.json", summary)
    return summary
