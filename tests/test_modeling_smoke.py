import numpy as np
import pandas as pd

from olist_delivery_risk.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from olist_delivery_risk.modeling import train_and_evaluate


def test_complete_modeling_workflow_runs_on_synthetic_orders() -> None:
    rows = 240
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "order_id": [f"order-{index:03d}" for index in range(rows)],
            "order_approved_at": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "order_estimated_delivery_date": pd.date_range("2020-01-08", periods=rows, freq="h"),
            "order_delivered_customer_date": pd.date_range("2020-01-07", periods=rows, freq="h"),
            "delay_days": np.where(np.arange(rows) % 7 == 0, 2.0, -1.0),
            "review_score": np.where(np.arange(rows) % 7 == 0, 2.0, 5.0),
            "is_late": (np.arange(rows) % 7 == 0).astype(int),
        }
    )
    for feature in NUMERIC_FEATURES:
        frame[feature] = rng.normal(size=rows)
    frame["promised_lead_days"] = np.where(frame["is_late"].eq(1), 3.0, 9.0)
    for index, feature in enumerate(CATEGORICAL_FEATURES):
        frame[feature] = np.where(np.arange(rows) % (index + 2) == 0, "A", "B")

    result = train_and_evaluate(frame, random_state=42)

    assert result.model_name in {
        "dummy_prior",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert len(result.test_predictions) == 48
    assert set(result.test_predictions["risk_tier"]) == {"Low", "Medium", "High"}
    assert 0.0 <= result.test_metrics["average_precision"] <= 1.0
