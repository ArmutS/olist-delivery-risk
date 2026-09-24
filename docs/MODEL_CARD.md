# Model card

## Intended use

The model ranks delivered Olist orders by the risk of arriving after their promised date. Scoring occurs at `order_approved_at` and is intended to support a capacity-constrained monitoring queue.

It is not designed for automated customer decisions, cancellation prediction, causal attribution, or deployment outside this dataset without retraining and validation.

## Model and target

- Selected family: random forest (`n_estimators=300`, `min_samples_leaf=20`, `max_features=sqrt`)
- Target: `order_delivered_customer_date > order_estimated_delivery_date`
- Calibration: Platt scaling fitted on a dedicated chronological calibration split
- Model-selection metric: average precision
- Operating threshold: calibration-period F2 optimum
- Monitoring tiers: highest 10%, next 20%, and remaining 70% of each scoring batch

Numeric fields use median imputation with missingness indicators. Categorical fields use most-frequent imputation and one-hot encoding with rare-category grouping. The random seed is 42.

## Evaluation design

| Split | Orders | Date range | Late rate |
|---|---:|---|---:|
| Train | 57,837 | 2016-09-15 – 2018-03-03 | 7.88% |
| Selection | 9,639 | 2018-03-03 – 2018-04-15 | 15.94% |
| Calibration | 9,639 | 2018-04-16 – 2018-05-26 | 7.41% |
| Test | 19,280 | 2018-05-26 – 2018-08-29 | 5.28% |

Rows are sorted by approval timestamp and order ID before contiguous splitting. The sharp prevalence changes are retained rather than randomized away.

## Final test performance

| Metric | Value |
|---|---:|
| ROC AUC | 0.7327 |
| Average precision | 0.1126 |
| Brier score | 0.0486 |
| Log loss | 0.1913 |
| Expected calibration error, 10 bins | 0.0111 |
| Top-10% lift | 2.2495 |
| Top-30% recall | 0.6650 |

The F2 threshold favors recall: it captures 73.6% of late orders but has 10.3% precision. It should not be treated as a universal production cutoff because intervention costs are unavailable.

## Feature sensitivity

Permutation importance and an explicit ablation identify `promised_lead_days` as the dominant signal.

| Feature set | ROC AUC | Average precision | Top-10% lift |
|---|---:|---:|---:|
| All approval-time features | 0.7327 | 0.1126 | 2.2495 |
| Without `promised_lead_days` | 0.4598 | 0.0478 | 0.8939 |

The feature is legitimate at the prediction moment: the estimated delivery date is already communicated when the order is approved. Its dominance nevertheless means the fitted model should be interpreted as promise-aware. It does not demonstrate strong logistics-risk prediction independent of the marketplace's promise-setting process.

## Risk-tier behavior

The High 10% and Medium 20% tiers produced similar test late rates (11.88% and 11.62%), while Low produced 2.53%. The model meaningfully separates the monitored top 30% from Low, but the High/Medium distinction is not strongly supported in the final period.

## Known limitations and monitoring

- Population: delivered orders with complete timestamps and item records only.
- Dataset shift: late-order prevalence changes materially over time.
- External validity: not established outside the historical Olist marketplace data.
- Geographic approximation: ZIP-prefix centroid distance is not route distance.
- Calibration: batch prevalence changes can degrade probabilities after deployment.
- Fairness: state-level performance and resource allocation should be audited before operational use.

Recommended monitoring includes target prevalence, missingness, feature drift, calibration, top-capacity precision/recall, and subgroup performance by customer and seller state. Retraining and recalibration should be triggered by a defined operational policy rather than this offline study alone.
