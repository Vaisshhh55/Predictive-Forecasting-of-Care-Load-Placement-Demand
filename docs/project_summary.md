# Technical Documentation

## Overview

This project provides a practical forecasting workflow for predicting children in care and the likely demand for discharge and placement planning. The application is designed to support operational planning by turning historical workload data into a forward-looking view of near-term demand.

## Data Sources

The app uses a synthetic dataset in [data/sample_uac_forecasting.csv](../data/sample_uac_forecasting.csv). This allows the project to run locally without external dependencies while still demonstrating the full forecasting lifecycle.

## Methodology

1. Load and validate the daily operational series.
2. Convert the date field to datetime and ensure daily continuity.
3. Build lag and rolling features to capture recent trends.
4. Create a net-pressure signal using transfers minus discharges.
5. Use time-based train/test validation to compare models.
6. Generate forecast intervals and summarize model error.

## Dashboard Modules

- forecast target selector
- model selection 
- forecast horizon controls
- actual-vs-forecast chart
- confidence interval band
- KPI summary cards
- model comparison table
- executive summary and methodology sections

## Model Evaluation

The app uses MAE, RMSE, and MAPE for evaluation so stakeholders can interpret both average error and percentage-based performance.

## Current Scope and Extensions

The current implementation includes persistence and moving-average baselines, exponential smoothing, ARIMA, SARIMA, linear regression, random forest, and gradient boosting. It also includes confidence bands, operational KPI calculations, and normal, intake-surge, and constrained-discharge scenarios.

Recommended future extensions are:

- integrate and govern the real HHS dataset
- calibrate capacity thresholds with stakeholder input
- add holiday calendars and external policy or border-activity signals
- export CSV and chart snapshots for leadership review
