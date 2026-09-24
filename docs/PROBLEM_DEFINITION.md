# Problem definition

## Decision

At order approval time, estimate the probability that a delivered order will
arrive after its promised delivery date. The score supports early monitoring;
it does not prescribe a logistics action by itself.

## Prediction moment

The cutoff is `order_approved_at`. A feature is eligible only when its source
information exists no later than this timestamp. Historical seller features
may use only orders whose delivery outcomes were already observed before the
current order was approved.

## Outcome

```text
is_late = 1 if order_delivered_customer_date > order_estimated_delivery_date
          0 otherwise
```

The modeling cohort contains delivered orders with complete purchase,
approval, estimated-delivery, actual-delivery, and order-item records. The
model therefore estimates lateness conditional on an order eventually having
a delivery outcome; cancellation risk is outside scope.

## Decision policy

Probability calibration and the binary intervention threshold are learned on
a chronological calibration period. Risk tiers are recomputed from scores in
each operational batch without using outcomes:

- High: the highest-risk 10% of the current scoring batch;
- Medium: the next 20%;
- Low: the remaining orders.

The tier names express review priority, not guaranteed probability ranges.
Batch-relative ranking keeps review workload stable when the score distribution
drifts over time.

## Primary evaluation

Average precision is the primary model-selection metric because late orders
are uncommon. ROC AUC, Brier score, log loss, calibration error, lift, recall
at review capacity, and confusion-matrix metrics provide complementary views.
The final chronological test period is used exactly once after model selection,
calibration, and threshold definition.
