"""Chronological model selection and calibrated final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .contracts import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, PREDICTION_TIME, TARGET
from .evaluation import (
    PlattCalibrator,
    RiskThresholds,
    assign_capacity_tiers,
    binary_metrics,
    risk_tier_summary,
    select_f2_threshold,
)


@dataclass(slots=True)
class TemporalSplits:
    train: pd.DataFrame
    selection: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "selection", "calibration", "test"):
            frame = getattr(self, name)
            rows.append(
                {
                    "split": name,
                    "rows": len(frame),
                    "start": frame[PREDICTION_TIME].min(),
                    "end": frame[PREDICTION_TIME].max(),
                    "late_orders": int(frame[TARGET].sum()),
                    "late_rate": float(frame[TARGET].mean()),
                }
            )
        return pd.DataFrame(rows)


def chronological_split(
    frame: pd.DataFrame,
    proportions: dict[str, float] | None = None,
) -> TemporalSplits:
    proportions = proportions or {"train": 0.60, "selection": 0.10, "calibration": 0.10, "test": 0.20}
    expected_keys = ["train", "selection", "calibration", "test"]
    if list(proportions) != expected_keys:
        raise ValueError(f"Split proportions must be ordered as {expected_keys}")
    if not np.isclose(sum(proportions.values()), 1.0):
        raise ValueError("Split proportions must sum to one")

    ordered = frame.sort_values([PREDICTION_TIME, "order_id"]).reset_index(drop=True)
    cut_1 = int(len(ordered) * proportions["train"])
    cut_2 = cut_1 + int(len(ordered) * proportions["selection"])
    cut_3 = cut_2 + int(len(ordered) * proportions["calibration"])
    splits = TemporalSplits(
        train=ordered.iloc[:cut_1].copy(),
        selection=ordered.iloc[cut_1:cut_2].copy(),
        calibration=ordered.iloc[cut_2:cut_3].copy(),
        test=ordered.iloc[cut_3:].copy(),
    )
    for earlier, later in [(splits.train, splits.selection), (splits.selection, splits.calibration), (splits.calibration, splits.test)]:
        if earlier[PREDICTION_TIME].max() > later[PREDICTION_TIME].min():
            raise AssertionError("Temporal splits overlap out of chronological order")
    return splits


def _one_hot_preprocessor(
    scale_numeric: bool,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), numeric_features), ("categorical", categorical, categorical_features)]
    )


def _ordinal_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)]
    )


def build_candidate_models(
    random_state: int = 42,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    categorical_indices = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )
    return {
        "dummy_prior": Pipeline(
            [
                (
                    "preprocess",
                    _one_hot_preprocessor(False, numeric_features, categorical_features),
                ),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _one_hot_preprocessor(True, numeric_features, categorical_features)),
                (
                    "model",
                    LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _one_hot_preprocessor(False, numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _ordinal_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=250,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        categorical_features=categorical_indices,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _xy(
    frame: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    selected_features = MODEL_FEATURES if features is None else features
    return frame[selected_features], frame[TARGET].astype(int)


def select_model(splits: TemporalSplits, random_state: int = 42) -> tuple[str, pd.DataFrame]:
    x_train, y_train = _xy(splits.train)
    x_selection, y_selection = _xy(splits.selection)
    rows = []
    for name, model in build_candidate_models(random_state).items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_selection)[:, 1]
        rows.append({"model": name, **binary_metrics(y_selection, probability)})
    comparison = pd.DataFrame(rows).sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    ).reset_index(drop=True)
    return str(comparison.loc[0, "model"]), comparison


@dataclass(slots=True)
class FinalModelResult:
    model_name: str
    model: Pipeline
    calibrator: PlattCalibrator
    thresholds: RiskThresholds
    decision_threshold: float
    model_comparison: pd.DataFrame
    split_summary: pd.DataFrame
    raw_calibration_metrics: dict[str, Any]
    calibrated_calibration_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_predictions: pd.DataFrame
    tier_summary: pd.DataFrame


def train_and_evaluate(
    frame: pd.DataFrame,
    proportions: dict[str, float] | None = None,
    random_state: int = 42,
    high_capacity: float = 0.10,
    medium_capacity: float = 0.20,
) -> FinalModelResult:
    splits = chronological_split(frame, proportions)
    selected_name, comparison = select_model(splits, random_state)

    development = pd.concat([splits.train, splits.selection], ignore_index=True)
    x_development, y_development = _xy(development)
    model = build_candidate_models(random_state)[selected_name]
    model.fit(x_development, y_development)

    x_calibration, y_calibration = _xy(splits.calibration)
    raw_calibration_probability = model.predict_proba(x_calibration)[:, 1]
    calibrator = PlattCalibrator.fit(raw_calibration_probability, y_calibration)
    calibrated_probability = calibrator.transform(raw_calibration_probability)
    decision_threshold = select_f2_threshold(y_calibration, calibrated_probability)
    x_test, y_test = _xy(splits.test)
    raw_test_probability = model.predict_proba(x_test)[:, 1]
    test_probability = calibrator.transform(raw_test_probability)
    test_predictions = splits.test[
        [
            "order_id",
            PREDICTION_TIME,
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "delay_days",
            "review_score",
            "dominant_product_category",
            "customer_state",
            "dominant_seller_state",
            TARGET,
        ]
    ].copy()
    test_predictions["raw_probability"] = raw_test_probability
    test_predictions["late_probability"] = test_probability
    test_predictions["risk_tier"], thresholds = assign_capacity_tiers(
        test_probability, high_capacity, medium_capacity
    )

    return FinalModelResult(
        model_name=selected_name,
        model=model,
        calibrator=calibrator,
        thresholds=thresholds,
        decision_threshold=decision_threshold,
        model_comparison=comparison,
        split_summary=splits.summary(),
        raw_calibration_metrics=binary_metrics(y_calibration, raw_calibration_probability),
        calibrated_calibration_metrics=binary_metrics(
            y_calibration, calibrated_probability, threshold=decision_threshold
        ),
        test_metrics=binary_metrics(y_test, test_probability, threshold=decision_threshold),
        test_predictions=test_predictions,
        tier_summary=risk_tier_summary(test_predictions),
    )


def evaluate_feature_ablation(
    frame: pd.DataFrame,
    model_name: str,
    excluded_features: list[str],
    proportions: dict[str, float] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate a preselected model family after removing specified features.

    The model family is fixed before this diagnostic, so the untouched test period
    is used for transparent sensitivity reporting rather than another selection step.
    """
    unknown = sorted(set(excluded_features) - set(MODEL_FEATURES))
    if unknown:
        raise ValueError(f"Unknown ablation features: {unknown}")
    numeric_features = [name for name in NUMERIC_FEATURES if name not in excluded_features]
    categorical_features = [name for name in CATEGORICAL_FEATURES if name not in excluded_features]
    features = numeric_features + categorical_features
    splits = chronological_split(frame, proportions)
    development = pd.concat([splits.train, splits.selection], ignore_index=True)
    x_development, y_development = _xy(development, features)
    model = build_candidate_models(
        random_state,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )[model_name]
    model.fit(x_development, y_development)

    x_calibration, y_calibration = _xy(splits.calibration, features)
    raw_calibration_probability = model.predict_proba(x_calibration)[:, 1]
    calibrator = PlattCalibrator.fit(raw_calibration_probability, y_calibration)
    x_test, y_test = _xy(splits.test, features)
    probability = calibrator.transform(model.predict_proba(x_test)[:, 1])
    return binary_metrics(y_test, probability)
