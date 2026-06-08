# Model Validation Report

## Executive Summary & Tier Rating

Based on the independent validation results of PD Model v1.0, the model is rated as Tier 1 - Not Approved. The main quantitative drivers of this validation run are a KS Statistic of 0.4712 and an Expected vs Actual Calibration Variance of 0.2555. The model is NOT APPROVED for use. Mandatory remediation is required before this model may be used for ECL calculation or any regulatory purpose. A formal Model Action Plan (MAP) must be submitted within 30 days. The model should not be relied upon for business decisions until remediation is complete and a subsequent validation confirms acceptable performance.

## Conceptual Soundness

The Logistic Regression model is theoretically justified and utilizes input features including interest_rate, gdp_growth, lgd, vintage_month to model the Binary default outcome (1 = default, 0 = performing). The model is intended for IFRS 9 ECL calculation and regulatory stress testing using development data from Synthetic loan portfolio — 1000 observations. Key assumptions include stable relationships between macroeconomic conditions and portfolio default rates, with developmental limitations documented accordingly.

## Ongoing Monitoring Results

Ongoing monitoring results indicate that the population stability index (PSI) is 0.0359 and the Kolmogorov-Smirnov (KS) statistic is 0.4712. Expected vs Actual Variance is 0.2555. The performance metrics indicate acceptable model discrimination, though calibration has breached the Red threshold and requires remediation, with no significant structural model drift observed under standard conditions.

### Breach Narrative

The validation results indicate a Red threshold breach in Expected vs Actual Variance with an observed value of 0.2555, exceeding the Red threshold limit of 0.10. This breach is likely driven by data shifts in the validation dataset, requiring a force provision override and ECL adjustment.

## Outcomes Analysis (Backtesting)

| Metric | Value | Status | Independent Challenge Action |
| --- | ---: | --- | --- |
| KS Statistic | 0.4712 | Green | Continue standard monitoring. |
| Population Stability Index | 0.0359 | Green | Continue standard monitoring. |
| Expected vs Actual Variance | 0.2555 | Red | Force provision override and ECL adjustment. |

### Backtesting Details
| Backtest | Result |
| --- | ---: |
| AUC | 0.8034 |
| Gini | 0.6069 |
| Calibration Variance | 0.2555 |
| Hosmer-Lemeshow χ²(8) | 32.0423, p = 0.0001 — Fail |
| Observations | 1000 |

### Metrics Comparison under Stress
| Metric | Base | Stressed | Change |
| --- | ---: | ---: | ---: |
| KS Statistic | 0.4712 | 0.4712 | 0.0000 |
| AUC | 0.8034 | 0.8034 | 0.0000 |
| Calibration Variance | 0.2555 | 0.0310 | -0.2245 |

*Note: Rank-order discrimination metrics (KS, AUC) are invariant to the logit shift stress methodology applied, as PDs are scaled proportionally without altering their relative ordering. Calibration Variance is sensitive to the shift and reflects the true stressed calibration impact.*

## Stress Testing and ECL Impact

**Stress Scenario Applied:** Interest rate (+1.00%) + GDP growth (-0.75%) + Unemployment (+1.25%) from base.  
**Variables shocked:** Interest rate, GDP growth, Unemployment  
**Variables held constant:** All other inputs unchanged.

| ECL Measure | Amount |
| --- | ---: |
| Base ECL | 168,399,248.64 |
| Stressed ECL | 205,846,492.00 |
| ECL Delta | 37,447,243.36 |

Under the applied macro shock (Interest rate (+1.00%) + GDP growth (-0.75%) + Unemployment (+1.25%)), the stressed ECL increased to $205,846,492.00 from the base ECL of $168,399,248.64, resulting in an ECL change of +22.24% ($37,447,243.36). This directional change in credit loss is economically plausible and consistent with historical portfolio sensitivity to stress factors.

## Overrides & Limitations

- Data Limitations: The analysis was performed on the Synthetic loan portfolio — 1000 observations dataset containing 1000 observations, which may not fully capture tail-risk events.
- Model Assumptions: The model assumes standard logistic defaults and macro sensitivities remain constant under stress.
- Stress Test Scope: The stress test was limited to single-variable macro shocks of interest_rate_delta=+1.00%, gdp_growth_delta=-0.75%, unemployment_delta=+1.25% and did not cover joint distributions.
- Analyst Overlay Status: No manual overlay adjustments or overrides were applied during this validation run, subject to the manual logit shift approximation limitation.
