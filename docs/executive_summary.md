# Executive Summary

## Decision Context

HHS care planning requires an early view of future children in care and placement demand. Intake surges, transfers, and discharge capacity can change quickly, making historical reporting alone insufficient for staffing and shelter decisions.

## What This Project Provides

- Short-term forecasts for HHS care load and placement demand.
- Model comparison across baseline, statistical, and machine-learning approaches.
- Confidence intervals to communicate forecast uncertainty.
- Surge lead time and capacity-breach indicators.
- Scenario comparison for expected demand, increased intake, and constrained discharges.

## Recommended Use

Use the 14- to 30-day horizon for operational planning, compare multiple models before high-impact decisions, and monitor net pressure from transfers minus discharges as a leading signal. Treat every forecast as planning evidence that should be reviewed with current capacity, staffing, and placement information.

## Important Limitation

The local project uses synthetic demonstration data because the external operational dataset is not bundled. The dashboard and methodology are ready for replacement with governed HHS data, but thresholds, scenario assumptions, and model performance must be recalibrated before production use.
