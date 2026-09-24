import pandas as pd

from olist_delivery_risk.data import _seller_history_at_approval


def test_seller_history_uses_only_resolved_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["A", "B", "C"],
            "seller_id": ["seller-1"] * 3,
            "order_approved_at": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-09"]),
            "order_delivered_customer_date": pd.to_datetime(
                ["2020-01-10", "2020-01-08", "2020-01-20"]
            ),
            "is_late": [1, 0, 1],
            "delivery_time_days": [9.0, 3.0, 11.0],
        }
    )

    result = _seller_history_at_approval(frame).set_index("order_id")

    assert result.loc["A", "seller_past_order_count"] == 0
    assert result.loc["B", "seller_past_order_count"] == 0
    assert result.loc["C", "seller_past_order_count"] == 1
    assert result.loc["C", "seller_past_late_sum"] == 0
    assert result.loc["C", "seller_past_delivery_days"] == 3.0
