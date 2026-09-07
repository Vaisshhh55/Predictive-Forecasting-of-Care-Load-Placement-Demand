# Predictive Forecasting of Care Load & Placement Demand

Detailed guide and project requirements for the Predictive Forecasting of Care Load & Placement Demand analysis.

## Background and Context

The UAC Program operates in a high-uncertainty environment, where sudden changes in border activity, policy enforcement, or humanitarian crises can rapidly increase the number of children entering federal care. While descriptive analytics explain what has already happened, HHS decision-makers require forward-looking intelligence to answer:

- How many children will be under HHS care in the coming days or weeks?
- Will discharge capacity be sufficient to offset incoming transfers?
- When should shelters, medical staff, and caseworkers be scaled up in advance?

This project introduces predictive modeling to enable proactive, rather than reactive, healthcare and child-welfare planning.

## Problem Statement

Despite having high-quality daily time-series data, the UAC Program currently lacks:

- Short-term forecasts of children in HHS care
- Predictive estimates of discharge (placement) demand
- Early-warning indicators of upcoming capacity stress

As a result, operational responses are often delayed, increasing:

- Overcrowding risk
- Staff burnout
- Length of stay for children

## Project Objectives

- Forecast the number of children in HHS care
- Estimate future imbalance between intake and exits
- Predict short-term discharge demand

## Secondary Objectives

- Provide early warnings for healthcare planners
- Quantify forecast uncertainty
- Compare statistical vs machine-learning forecasting approaches

## Dataset Description

| Column | Description |
| --- | --- |
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake volume |
| Children in CBP custody | Active CBP care load |
| Children transferred out of CBP custody | Flow into HHS system |
| Children in HHS Care | Active HHS care load |
| Children discharged from HHS Care | Successful sponsor placements |

## Analytical Methodology (Step-by-Step)

### Time-Series Preparation

- Convert Date to datetime index
- Ensure continuity of daily observations
- Handle missing days via interpolation or masking
- Decompose the series into trend, seasonality, and residuals

### Feature Engineering for Forecasting

- Lag features (t-1, t-7, t-14 values)
- Rolling averages (7-day and 14-day rolling mean and variance)
- Flow-Based Signals: Transfers − Discharges (net pressure indicator)
- Calendar effects: day of week, month, holiday proxies

### Train–Test Strategy

- Strict time-based split (no random sampling)
- Walk-forward validation
- Multi-horizon evaluation

### Forecasting Models

- Baseline Models: Naïve persistence model, moving average forecast
- Statistical Models: ARIMA / SARIMA, Exponential Smoothing
- Machine Learning Models: Random Forest Regressor, Gradient Boosting Regressor

## Key Performance Indicators (KPIs)

- Forecast Accuracy (%)
- Surge Lead Time
- Capacity Breach Probability
- Forecast Stability Index
- Model robustness

## Streamlit Web Application Requirements

### Core Modules

- Future Care Load Forecast Chart
- Discharge Demand Forecast Panel
- Model Selection & Comparison
- Confidence Interval Visualization

### User Capabilities

- Forecast horizon selector
- Model toggle
- Scenario comparison view

## Deliverables and Submission

- Research paper (EDA, insights, recommendations)
- Streamlit dashboard (live analytics)
- Executive summary for government stakeholders

## Conclusion

This project elevates the UAC dataset from historical reporting to predictive intelligence. By applying rigorous time-series and machine-learning techniques, it enables HHS stakeholders to anticipate future care demands, allocate resources proactively, and strengthen child-welfare outcomes through data-driven foresight.

## Project Structure

- app/: dashboard and forecasting logic
- data/: synthetic demo dataset
- docs/: documentation and summary notes
- tests/: smoke tests
- .streamlit/: app configuration

## Setup

```powershell
cd "D:\unified mentor project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/main.py
```

## Run Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

For development and testing, install the test runner separately with `python -m pip install pytest==8.3.2`.

## Implemented Dashboard Requirements

The dashboard includes care-load and placement-demand forecasts, baseline/statistical/machine-learning model comparison, confidence intervals, time-series decomposition, holiday and calendar features, strict time-based validation, seven-, thirty-, and ninety-day evaluation, operational KPIs, capacity-threshold controls, and scenario comparison.

The synthetic CSV is used by default. Use the optional **Use operational CSV** uploader in the sidebar to run the same workflow with a governed operational dataset that contains the required columns.

