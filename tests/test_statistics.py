import pandas as pd

from olist_delivery_risk.statistics import review_impact_analysis


def test_review_impact_reports_effect_direction() -> None:
    frame = pd.DataFrame(
        {
            "is_late": [1, 1, 1, 0, 0, 0],
            "review_score": [1, 2, 2, 4, 5, 5],
        }
    )
    result = review_impact_analysis(frame, bootstrap_repetitions=100, bootstrap_seed=7)
    assert result["mean_difference_late_minus_on_time"] < 0
    assert result["rank_biserial_correlation"] < 0
