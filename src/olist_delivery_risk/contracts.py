"""Feature and source-data contracts for leakage-safe modeling."""

from __future__ import annotations

from collections.abc import Iterable

TARGET = "is_late"
PREDICTION_TIME = "order_approved_at"

NUMERIC_FEATURES = [
    "price_total",
    "freight_total",
    "freight_ratio",
    "payment_value_total",
    "payment_installments_max",
    "payment_record_count",
    "item_count",
    "unique_product_count",
    "seller_count",
    "category_count",
    "product_weight_g_total",
    "product_volume_cm3_total",
    "distance_km_mean",
    "distance_km_max",
    "promised_lead_days",
    "shipping_limit_slack_days_min",
    "approval_month_sin",
    "approval_month_cos",
    "approval_dow_sin",
    "approval_dow_cos",
    "seller_history_order_count",
    "seller_history_late_rate",
    "seller_history_delivery_days",
    "new_seller_share",
]

CATEGORICAL_FEATURES = [
    "dominant_product_category",
    "customer_state",
    "dominant_seller_state",
    "dominant_payment_type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROHIBITED_MODEL_COLUMNS = {
    "customer_id",
    "customer_unique_id",
    "order_id",
    "product_id",
    "seller_id",
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivery_time_days",
    "delay_days",
    "review_score",
    "review_comment_message",
    TARGET,
}


def assert_feature_contract(features: Iterable[str]) -> None:
    feature_list = list(features)
    duplicates = sorted({name for name in feature_list if feature_list.count(name) > 1})
    prohibited = sorted(set(feature_list) & PROHIBITED_MODEL_COLUMNS)
    if duplicates:
        raise ValueError(f"Duplicate model features: {duplicates}")
    if prohibited:
        raise ValueError(f"Post-outcome or identifier features are prohibited: {prohibited}")
    if set(feature_list) != set(MODEL_FEATURES):
        missing = sorted(set(MODEL_FEATURES) - set(feature_list))
        extra = sorted(set(feature_list) - set(MODEL_FEATURES))
        raise ValueError(f"Feature contract mismatch; missing={missing}, extra={extra}")
