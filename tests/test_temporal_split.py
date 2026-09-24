import pandas as pd

from olist_delivery_risk.modeling import chronological_split


def test_chronological_split_preserves_time_order() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [f"order-{index}" for index in range(100)],
            "order_approved_at": pd.date_range("2020-01-01", periods=100, freq="h"),
            "is_late": [index % 5 == 0 for index in range(100)],
        }
    )
    splits = chronological_split(frame)
    assert [len(splits.train), len(splits.selection), len(splits.calibration), len(splits.test)] == [60, 10, 10, 20]
    assert splits.train["order_approved_at"].max() < splits.selection["order_approved_at"].min()
    assert splits.selection["order_approved_at"].max() < splits.calibration["order_approved_at"].min()
    assert splits.calibration["order_approved_at"].max() < splits.test["order_approved_at"].min()
