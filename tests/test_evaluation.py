import numpy as np
import pandas as pd

from olist_delivery_risk.evaluation import (
    PlattCalibrator,
    assign_capacity_tiers,
)


def test_platt_calibration_returns_probabilities() -> None:
    raw = np.array([0.05, 0.15, 0.25, 0.55, 0.75, 0.95])
    target = np.array([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator.fit(raw, target)
    calibrated = calibrator.transform(raw)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) > 0)


def test_capacity_thresholds_create_ordered_tiers() -> None:
    probability = np.linspace(0.01, 0.99, 100)
    tiers_raw, thresholds = assign_capacity_tiers(probability, high_capacity=0.10, medium_capacity=0.20)
    tiers = pd.Series(tiers_raw)
    assert thresholds.high > thresholds.medium
    assert (tiers == "High").sum() == 10
    assert (tiers == "Medium").sum() == 20
