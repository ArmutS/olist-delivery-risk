import pandas as pd

from olist_delivery_risk.prioritization import build_all_priorities


def test_priorities_are_ranked_within_each_segment_family() -> None:
    rows = []
    for index in range(200):
        rows.append(
            {
                "order_id": f"order-{index}",
                "dominant_product_category": "A" if index < 100 else "B",
                "customer_state": "SP" if index % 2 else "RJ",
                "dominant_seller_state": "SP",
                "is_late": int(index % 7 == 0),
                "late_probability": 0.2 if index < 100 else 0.1,
                "delay_days": 3.0 if index % 7 == 0 else -2.0,
                "review_score": 2.0 if index % 7 == 0 else 5.0,
            }
        )
    priorities, weights = build_all_priorities(pd.DataFrame(rows), minimum_orders=10)
    assert not priorities.empty
    assert not weights.empty
    assert priorities.groupby("segment_family")["priority_rank"].min().eq(1).all()
    assert weights.groupby("segment_family")["weight"].sum().round(10).eq(1.0).all()
