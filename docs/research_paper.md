# Predictive Forecasting of Care Load and Placement Demand

## Abstract

The Unaccompanied Alien Children (UAC) program operates under changing intake, transfer, and discharge conditions. This project converts daily operational history into short-term predictive intelligence for HHS care-load and placement planning. The local implementation combines time-series preparation, lag and rolling features, flow-based pressure signals, time-aware validation, multiple forecasting families, confidence intervals, and scenario analysis.

## Background and Problem

Descriptive reporting explains historical workload but does not tell decision-makers how many children may require care in the next several days or whether placement capacity may be sufficient. Delayed planning can increase overcrowding risk, staff pressure, and length of stay. The forecasting workflow provides an early-warning view for staffing, shelter, healthcare, and placement decisions.

## Objectives

- Forecast the number of children in HHS care.
- Estimate the future imbalance between transfers and discharges.
- Predict short-term placement demand.
- Quantify uncertainty and compare model families.
- Provide scenario-based operational planning signals.

## Data and Preparation

The demonstration dataset contains daily reporting dates, CBP intake and care-load measures, transfers to HHS, HHS care load, HHS discharges, and placement demand. Dates are sorted, duplicate dates are removed, and missing daily observations are interpolated for numeric fields. Calendar features include day of week, month, year, and weekend status.

## Methodology

Feature engineering includes one-, seven-, and fourteen-day lags; seven- and fourteen-day rolling means, variances, and standard deviations; and a net-pressure signal defined as transfers to HHS minus discharges from HHS. Evaluation uses strict time-series splits and walk-forward validation rather than random sampling.

The application supports Naive Persistence, Moving Average, Exponential Smoothing, ARIMA, SARIMA, Linear Regression, Random Forest, and Gradient Boosting. The selected model generates the displayed future horizon forecast. MAE measures absolute error, RMSE emphasizes large errors, MAPE expresses relative error, and horizon error can be compared through the model table.

## Operational Outputs

The dashboard provides historical-versus-forecast charts, confidence bands, forecast-horizon tables, model comparison, forecast accuracy, surge lead time, capacity-breach probability, forecast stability, and three planning scenarios: expected demand, an intake surge assumption, and constrained discharge capacity.

## Limitations

The included data is synthetic and is intended only to make the project runnable locally. Scenario multipliers are sensitivity assumptions rather than causal estimates. Before operational use, HHS data governance, capacity thresholds, holiday calendars, external explanatory variables, and forecast calibration must be reviewed with subject-matter experts.

## Conclusion

The project demonstrates a reproducible path from daily UAC reporting to predictive planning intelligence. It gives stakeholders a transparent way to compare forecasts, inspect uncertainty, and test operational pressure scenarios while preserving clear limitations around synthetic demonstration data.
