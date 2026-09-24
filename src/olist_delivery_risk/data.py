"""Build a one-row-per-order, point-in-time-safe modeling table."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import MODEL_FEATURES, PREDICTION_TIME, TARGET, assert_feature_contract

SOURCE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    },
    "products": {
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    },
    "sellers": {"seller_id", "seller_zip_code_prefix", "seller_state"},
    "customers": {
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_state",
    },
    "payments": {
        "order_id",
        "payment_type",
        "payment_installments",
        "payment_value",
    },
    "reviews": {"order_id", "review_score"},
    "geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_state",
    },
    "translation": {"product_category_name", "product_category_name_english"},
}

ORDER_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def normalize_zip_prefix(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)


def mode_or_unknown(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return "unknown"
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def _sum_with_nan(series: pd.Series) -> float:
    return float(series.sum(min_count=1))


def haversine_distance_km(
    lat1: pd.Series,
    lng1: pd.Series,
    lat2: pd.Series,
    lng2: pd.Series,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lng1_rad = np.radians(pd.to_numeric(lng1, errors="coerce"))
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lng2_rad = np.radians(pd.to_numeric(lng2, errors="coerce"))
    delta_lat = lat2_rad - lat1_rad
    delta_lng = lng2_rad - lng1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def load_source_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    missing_files = [filename for filename in SOURCE_FILES.values() if not (data_dir / filename).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing Olist source files in {data_dir}: {missing_files}")

    tables = {name: pd.read_csv(data_dir / filename) for name, filename in SOURCE_FILES.items()}
    for name, required in REQUIRED_COLUMNS.items():
        missing_columns = sorted(required - set(tables[name].columns))
        if missing_columns:
            raise ValueError(f"{SOURCE_FILES[name]} is missing required columns: {missing_columns}")
    return tables


def build_zip_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:
    geo = geolocation.copy()
    geo["zip_prefix"] = normalize_zip_prefix(geo["geolocation_zip_code_prefix"])
    geo["geolocation_lat"] = pd.to_numeric(geo["geolocation_lat"], errors="coerce")
    geo["geolocation_lng"] = pd.to_numeric(geo["geolocation_lng"], errors="coerce")
    geo = geo[
        geo["geolocation_lat"].between(-90, 90)
        & geo["geolocation_lng"].between(-180, 180)
    ].dropna(subset=["zip_prefix", "geolocation_lat", "geolocation_lng"])
    return (
        geo.groupby("zip_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "median"),
            geolocation_lng=("geolocation_lng", "median"),
            geolocation_state=("geolocation_state", mode_or_unknown),
        )
        .sort_values("zip_prefix")
    )


def _prepare_order_cohort(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = orders.copy()
    for column in ORDER_DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    diagnostics = {"source_orders": len(frame)}
    frame = frame[frame["order_status"].eq("delivered")].copy()
    diagnostics["delivered_orders"] = len(frame)
    frame = frame.dropna(subset=ORDER_DATE_COLUMNS)
    diagnostics["complete_timestamp_orders"] = len(frame)
    valid_time_order = (
        (frame["order_purchase_timestamp"] <= frame[PREDICTION_TIME])
        & (frame[PREDICTION_TIME] < frame["order_delivered_customer_date"])
        & (frame["order_purchase_timestamp"] < frame["order_estimated_delivery_date"])
    )
    diagnostics["invalid_timeline_orders_removed"] = int((~valid_time_order).sum())
    frame = frame[valid_time_order].copy()
    frame[TARGET] = (
        frame["order_delivered_customer_date"] > frame["order_estimated_delivery_date"]
    ).astype("int8")
    frame["delay_days"] = (
        frame["order_delivered_customer_date"] - frame["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    frame["delivery_time_days"] = (
        frame["order_delivered_customer_date"] - frame["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    frame["promised_lead_days"] = (
        frame["order_estimated_delivery_date"] - frame[PREDICTION_TIME]
    ).dt.total_seconds() / 86400.0
    return frame, diagnostics


def _seller_history_at_approval(
    seller_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Attach history resolved before each seller-order's approval timestamp."""

    result_frames: list[pd.DataFrame] = []
    columns = [
        "order_id",
        "seller_id",
        "seller_past_order_count",
        "seller_past_late_sum",
        "seller_past_delivery_days_sum",
        "seller_past_late_rate",
        "seller_past_delivery_days",
    ]
    for seller_id, group in seller_orders.groupby("seller_id", sort=False):
        queries = group[["order_id", PREDICTION_TIME]].copy()
        outcomes = group.sort_values("order_delivered_customer_date")
        delivery_times = outcomes["order_delivered_customer_date"].to_numpy(dtype="datetime64[ns]")
        cumulative_late = np.concatenate([[0.0], outcomes[TARGET].to_numpy(float).cumsum()])
        cumulative_delivery = np.concatenate(
            [[0.0], outcomes["delivery_time_days"].to_numpy(float).cumsum()]
        )
        approval_times = queries[PREDICTION_TIME].to_numpy(dtype="datetime64[ns]")
        resolved_count = np.searchsorted(delivery_times, approval_times, side="left")
        late_sum = cumulative_late[resolved_count]
        delivery_sum = cumulative_delivery[resolved_count]
        denominator = np.where(resolved_count > 0, resolved_count, np.nan)
        result_frames.append(
            pd.DataFrame(
                {
                    "order_id": queries["order_id"].to_numpy(),
                    "seller_id": seller_id,
                    "seller_past_order_count": resolved_count,
                    "seller_past_late_sum": late_sum,
                    "seller_past_delivery_days_sum": delivery_sum,
                    "seller_past_late_rate": late_sum / denominator,
                    "seller_past_delivery_days": delivery_sum / denominator,
                }
            )
        )
    if not result_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(result_frames, ignore_index=True)[columns]


def _aggregate_seller_history(history: pd.DataFrame) -> pd.DataFrame:
    grouped = history.groupby("order_id", as_index=False).agg(
        seller_history_order_count=("seller_past_order_count", "sum"),
        seller_history_late_sum=("seller_past_late_sum", "sum"),
        seller_history_delivery_sum=("seller_past_delivery_days_sum", "sum"),
        new_seller_share=("seller_past_order_count", lambda values: float((values == 0).mean())),
    )
    count = grouped["seller_history_order_count"].replace(0, np.nan)
    grouped["seller_history_late_rate"] = grouped["seller_history_late_sum"] / count
    grouped["seller_history_delivery_days"] = grouped["seller_history_delivery_sum"] / count
    return grouped.drop(columns=["seller_history_late_sum", "seller_history_delivery_sum"])


def build_order_level_dataset(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    tables = load_source_tables(data_dir)
    orders, diagnostics = _prepare_order_cohort(tables["orders"])

    geo = build_zip_geolocation(tables["geolocation"])
    customers = tables["customers"].copy()
    customers["customer_zip_code_prefix"] = normalize_zip_prefix(customers["customer_zip_code_prefix"])
    customer_geo = geo.rename(
        columns={
            "zip_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
            "geolocation_state": "customer_geo_state",
        }
    )
    customers = customers.merge(customer_geo, on="customer_zip_code_prefix", how="left", validate="m:1")
    order_base = orders.merge(customers, on="customer_id", how="left", validate="m:1")

    translation = tables["translation"].drop_duplicates("product_category_name")
    sellers = tables["sellers"].copy()
    sellers["seller_zip_code_prefix"] = normalize_zip_prefix(sellers["seller_zip_code_prefix"])
    seller_geo = geo.rename(
        columns={
            "zip_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
            "geolocation_state": "seller_geo_state",
        }
    )
    sellers = sellers.merge(seller_geo, on="seller_zip_code_prefix", how="left", validate="m:1")

    items = tables["items"].copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
    items = items.merge(tables["products"], on="product_id", how="left", validate="m:1")
    items = items.merge(translation, on="product_category_name", how="left", validate="m:1")
    items = items.merge(sellers, on="seller_id", how="left", validate="m:1")
    items["dominant_product_category"] = items["product_category_name_english"].fillna(
        items["product_category_name"]
    )
    items["product_volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )

    item_orders = items.merge(
        order_base[
            [
                "order_id",
                PREDICTION_TIME,
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "customer_lat",
                "customer_lng",
                TARGET,
                "delivery_time_days",
            ]
        ],
        on="order_id",
        how="inner",
        validate="m:1",
    )
    diagnostics["cohort_orders_with_items"] = int(item_orders["order_id"].nunique())
    diagnostics["cohort_orders_without_items_removed"] = len(order_base) - diagnostics["cohort_orders_with_items"]
    item_orders["distance_km"] = haversine_distance_km(
        item_orders["seller_lat"],
        item_orders["seller_lng"],
        item_orders["customer_lat"],
        item_orders["customer_lng"],
    )
    item_orders["shipping_limit_slack_days"] = (
        item_orders["shipping_limit_date"] - item_orders[PREDICTION_TIME]
    ).dt.total_seconds() / 86400.0

    item_agg = item_orders.groupby("order_id", as_index=False).agg(
        price_total=("price", _sum_with_nan),
        freight_total=("freight_value", _sum_with_nan),
        item_count=("order_item_id", "count"),
        unique_product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        category_count=("dominant_product_category", "nunique"),
        product_weight_g_total=("product_weight_g", _sum_with_nan),
        product_volume_cm3_total=("product_volume_cm3", _sum_with_nan),
        dominant_product_category=("dominant_product_category", mode_or_unknown),
        dominant_seller_state=("seller_state", mode_or_unknown),
        distance_km_mean=("distance_km", "mean"),
        distance_km_max=("distance_km", "max"),
        shipping_limit_slack_days_min=("shipping_limit_slack_days", "min"),
    )

    seller_orders = item_orders[
        [
            "order_id",
            "seller_id",
            PREDICTION_TIME,
            "order_delivered_customer_date",
            TARGET,
            "delivery_time_days",
        ]
    ].drop_duplicates(["order_id", "seller_id"])
    seller_history = _aggregate_seller_history(_seller_history_at_approval(seller_orders))

    payments = tables["payments"].groupby("order_id", as_index=False).agg(
        dominant_payment_type=("payment_type", mode_or_unknown),
        payment_installments_max=("payment_installments", "max"),
        payment_value_total=("payment_value", _sum_with_nan),
        payment_record_count=("payment_type", "size"),
    )
    reviews = tables["reviews"].groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )

    frame = order_base.merge(item_agg, on="order_id", how="inner", validate="1:1")
    frame = frame.merge(seller_history, on="order_id", how="left", validate="1:1")
    frame = frame.merge(payments, on="order_id", how="left", validate="1:1")
    frame = frame.merge(reviews, on="order_id", how="left", validate="1:1")
    frame["freight_ratio"] = frame["freight_total"] / frame["price_total"].replace(0, np.nan)

    month_angle = 2 * math.pi * (frame[PREDICTION_TIME].dt.month - 1) / 12
    dow_angle = 2 * math.pi * frame[PREDICTION_TIME].dt.dayofweek / 7
    frame["approval_month_sin"] = np.sin(month_angle)
    frame["approval_month_cos"] = np.cos(month_angle)
    frame["approval_dow_sin"] = np.sin(dow_angle)
    frame["approval_dow_cos"] = np.cos(dow_angle)

    if frame["order_id"].duplicated().any():
        raise AssertionError("Order-level dataset contains duplicate order_id values")
    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise AssertionError("Target must be binary")
    assert_feature_contract(MODEL_FEATURES)

    diagnostics.update(
        {
            "final_orders": len(frame),
            "late_orders": int(frame[TARGET].sum()),
            "late_rate": float(frame[TARGET].mean()),
            "multi_seller_orders_retained": int((frame["seller_count"] > 1).sum()),
            "prediction_start": frame[PREDICTION_TIME].min().isoformat(),
            "prediction_end": frame[PREDICTION_TIME].max().isoformat(),
        }
    )
    frame = frame.sort_values([PREDICTION_TIME, "order_id"]).reset_index(drop=True)
    return frame, diagnostics
