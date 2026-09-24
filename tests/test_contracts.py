import pytest

from olist_delivery_risk.contracts import MODEL_FEATURES, assert_feature_contract


def test_declared_feature_contract_passes() -> None:
    assert_feature_contract(MODEL_FEATURES)


def test_post_outcome_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        assert_feature_contract([*MODEL_FEATURES, "review_score"])


def test_missing_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        assert_feature_contract(MODEL_FEATURES[:-1])
