# Data card

## Source

The project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), distributed by its publisher under CC BY-NC-SA 4.0. Raw files are downloaded separately and excluded from this repository.

## Source tables

The pipeline validates and joins nine CSV files:

- orders
- order items
- products
- sellers
- customers
- order payments
- order reviews
- geolocation
- product-category translation

## Analysis unit and cohort

The unit is one order. Item, payment, review, seller, category, and location information is aggregated before modeling, preserving multi-item and multi-seller orders.

Starting from 99,441 source orders:

- 96,478 have delivered status;
- 96,456 have the required purchase, approval, promised-delivery, and actual-delivery timestamps;
- 61 invalid timestamp sequences are removed;
- 96,395 complete order-level rows remain;
- 1,273 multi-seller orders are retained;
- 7,824 final-cohort orders are late (8.12%).

The approval window runs from 15 September 2016 to 29 August 2018.

## Target

`is_late` equals one when actual customer delivery is after the estimated delivery date. `delay_days` is retained only for evaluation and descriptive analysis.

## Feature timing

Model features are restricted to fields available by approval. Seller-history features are calculated point in time: only deliveries completed before the current order's approval contribute to its history. The complete contract is documented in [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Processing decisions

- Duplicate ZIP prefixes are collapsed to median latitude and longitude.
- Customer-to-seller distance uses the haversine formula on ZIP-prefix centroids.
- Product categories are translated when a mapping is available.
- Multiple items, sellers, payments, categories, and product dimensions are aggregated at order level.
- Missing numeric and categorical values are imputed inside each model pipeline, after splitting.
- Post-outcome reviews are excluded from predictors and used only for descriptive association analysis.

## Quality and privacy

The public data contain pseudonymous identifiers and approximate locations. Raw files, processed order-level tables, predictions, and fitted artifacts are not committed. Tracked reports contain only aggregate metrics and figures.

This repository does not claim that public release eliminates all privacy or re-identification concerns. Any downstream use should follow the dataset terms and applicable governance requirements.

## Limitations

- The cohort conditions on recorded delivery and excludes canceled or unavailable outcomes.
- Dates and marketplace processes are historical.
- ZIP-prefix centroids introduce geographic measurement error.
- Reviews are missing for some orders and occur after delivery.
- Marketplace-specific promise setting is a dominant source of predictive signal.
