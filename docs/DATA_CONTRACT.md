# Feature availability contract

| Feature family | Available at approval? | Modeling use |
|---|---:|---|
| Price, freight, payment, product attributes | Yes | Included |
| Customer/seller state and approximate distance | Yes | Included |
| Promised lead time and seller shipping-limit slack | Yes | Included |
| Calendar attributes at approval | Yes | Included |
| Seller outcomes completed before approval | Yes | Included |
| Raw customer, product, order, or seller identifiers | Yes | Excluded |
| Carrier handoff date | No | Excluded |
| Actual customer delivery date | No | Target construction only |
| Delay duration / late flag | No | Target/evaluation only |
| Review score and review text | No | Post-outcome impact analysis only |

All source tables must preserve one row per order after aggregation. The
pipeline fails when order identifiers are duplicated, required columns are
missing, or a prohibited post-outcome column enters the model feature list.

## Dataset license

The source dataset is distributed separately by its publisher under
CC BY-NC-SA 4.0. Raw and row-level processed data are not committed here. The
MIT license in this repository applies to the original software only and does
not relicense the source dataset. Aggregate data-derived outputs follow the
source dataset's terms; see [DATA_LICENSE.md](DATA_LICENSE.md).
