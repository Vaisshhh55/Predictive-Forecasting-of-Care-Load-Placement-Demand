from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import build_feature_frame, load_dataset
from app.forecasting import (
    build_confidence_intervals,
    calculate_kpis,
    calculate_operational_kpis,
    compare_model_performance,
    create_forecast_table,
    decompose_series,
    evaluate_horizons,
    train_and_evaluate_model,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_uac_forecasting.csv"


@st.cache_data(show_spinner=False)
def _cached_model_result(feature_df: pd.DataFrame, target: str, model_name: str, horizon: int):
    return train_and_evaluate_model(feature_df, target, model_name, horizon=horizon)


@st.cache_data(show_spinner=False)
def _cached_forecast(feature_df: pd.DataFrame, target: str, horizon: int, model_name: str):
    return create_forecast_table(feature_df, target, horizon=horizon, model_name=model_name)


@st.cache_data(show_spinner=False)
def _cached_model_comparison(feature_df: pd.DataFrame, target: str):
    return compare_model_performance(feature_df, target)


@st.cache_data(show_spinner=False)
def _cached_horizon_evaluation(feature_df: pd.DataFrame, target: str, model_name: str):
    return evaluate_horizons(feature_df, target, model_name)


def render_project_overview() -> None:
    st.markdown(
        """
        <div style='display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; flex-wrap:wrap;'>
            <div style='font-size: 2rem; font-weight: 700;'>Logo</div>
            <div style='display:flex; gap:14px; flex-wrap:wrap; font-size:0.95rem;'>
                <a href='https://projects.unifiedmentor.com/' target='_blank'>Dashboard</a>
                <a href='https://projects.unifiedmentor.com/chat-support' target='_blank'>Chat Support</a>
                <a href='https://projects.unifiedmentor.com/task-calendar' target='_blank'>Calendar</a>
                <a href='https://projects.unifiedmentor.com/submit-project' target='_blank'>Submit Project</a>
                <a href='https://projects.unifiedmentor.com/faqs' target='_blank'>FAQs</a>
                <a href='https://jobs.unifiedmentor.com/' target='_blank'>Job Portal</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 4])
    with left_col:
        st.image("https://api.dicebear.com/5.x/initials/svg?seed=Vaishnavi%20Naik", width=120)
    with right_col:
        st.markdown("[Back to Dashboard](https://projects.unifiedmentor.com/project_instructions)")
        st.title("Projects")
        st.subheader("Predictive Forecasting of Care Load & Placement Demand")

    st.subheader("On this page")
    st.markdown(
        """
        - [Background and Context](#background-and-context)
        - [Problem Statement](#problem-statement)
        - [Project Objectives](#project-objectives)
        - [Secondary Objectives](#secondary-objectives)
        - [Dataset Description](#dataset-description)
        - [Analytical Methodology](#analytical-methodology-step-by-step)
        - [Key Performance Indicators](#key-performance-indicators-kpis)
        - [Streamlit Web Application Requirements](#streamlit-web-application-requirements)
        - [Deliverables and Submission](#deliverables-and-submission)
        - [Conclusion](#conclusion)
        """
    )

    st.header("Background and Context")
    st.markdown(
        """
        The UAC Program operates in a high-uncertainty environment, where sudden changes in border activity, policy enforcement,
        or humanitarian crises can rapidly increase the number of children entering federal care. While descriptive analytics explain
        what has already happened, HHS decision-makers require forward-looking intelligence to answer:

        - How many children will be under HHS care in the coming days or weeks?
        - Will discharge capacity be sufficient to offset incoming transfers?
        - When should shelters, medical staff, and caseworkers be scaled up in advance?

        This project introduces predictive modeling to enable proactive, rather than reactive, healthcare and child-welfare planning.
        """
    )
    st.markdown("[Unified Mentor](https://www.unifiedmentor.com/) | [U.S Department of Health and Human Services](https://www.hhs.gov/)")

    st.header("Problem Statement")
    st.markdown(
        """
        Despite having high-quality daily time-series data, the UAC Program currently lacks:

        - Short-term forecasts of children in HHS care
        - Predictive estimates of discharge (placement) demand
        - Early-warning indicators of upcoming capacity stress

        As a result, operational responses are often delayed, increasing:

        - Overcrowding risk
        - Staff burnout
        - Length of stay for children
        """
    )

    st.header("Project Objectives")
    st.markdown(
        """
        - Forecast the number of children in HHS care
        - Estimate future imbalance between intake and exits
        - Predict short-term discharge demand
        """
    )

    st.header("Secondary Objectives")
    st.markdown(
        """
        - Provide early warnings for healthcare planners
        - Quantify forecast uncertainty
        - Compare statistical vs machine-learning forecasting approaches
        """
    )

    st.header("Dataset Description")
    st.markdown(
        """
        | Column | Description |
        | --- | --- |
        | Date | Reporting date |
        | Children apprehended and placed in CBP custody | Daily intake volume |
        | Children in CBP custody | Active CBP care load |
        | Children transferred out of CBP custody | Flow into HHS system |
        | Children in HHS Care | Active HHS care load |
        | Children discharged from HHS Care | Successful sponsor placements |
        """
    )

    st.header("Analytical Methodology (Step-by-Step)")
    st.markdown(
        """
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
        """
    )

    st.header("Key Performance Indicators (KPIs)")
    st.markdown(
        """
        - Forecast Accuracy (%)
        - Surge Lead Time
        - Capacity Breach Probability
        - Forecast Stability Index
        - Model robustness
        """
    )

    st.header("Streamlit Web Application Requirements")
    st.markdown(
        """
        ### Core Modules
        - Future Care Load Forecast Chart
        - Discharge Demand Forecast Panel
        - Model Selection & Comparison
        - Confidence Interval Visualization

        ### User Capabilities
        - Forecast horizon selector
        - Model toggle
        - Scenario comparison view
        """
    )

    st.header("Deliverables and Submission")
    st.markdown(
        """
        - Research paper (EDA, insights, recommendations)
        - Streamlit dashboard (live analytics)
        - Executive summary for government stakeholders
        """
    )

    st.header("Conclusion")
    st.markdown(
        """
        This project elevates the UAC dataset from historical reporting to predictive intelligence. By applying rigorous time-series
        and machine-learning techniques, it enables HHS stakeholders to anticipate future care demands, allocate resources proactively,
        and strengthen child-welfare outcomes through data-driven foresight.
        """
    )
    st.markdown("[Access Dataset](https://drive.google.com/file/d/1xZo782T4EfnkC0BmCwJTb0DZYYHoOXKm/view?usp=sharing)")


def render_dashboard() -> None:
    project_title = "Predictive Forecasting of Care Load & Placement Demand"
    st.set_page_config(page_title=project_title, layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1500px; }
        h1, h2, h3 { color: #16324f; }
        [data-testid='stMetric'] { background: #f4f7f9; border: 1px solid #dbe4ea; padding: 0.8rem; border-radius: 8px; }
        .brief-banner { background: #16324f; color: white; padding: 1.15rem 1.35rem; border-radius: 8px; margin-bottom: 1.2rem; }
        .brief-banner h1 { color: white; margin: 0; font-size: 2rem; }
        .brief-banner p { margin: 0.35rem 0 0; color: #dceaf2; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Use operational CSV (optional)", type="csv")
    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)
        except ValueError as error:
            st.sidebar.error("Uploaded CSV does not match the required UAC dataset schema.")
            st.sidebar.caption(str(error))
            st.sidebar.info("Using the included synthetic demonstration dataset instead.")
            df = load_dataset(DATA_PATH)
    else:
        df = load_dataset(DATA_PATH)
    feature_df = build_feature_frame(df)

    st.sidebar.title("UAC Planning Console")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Executive Summary", "Technical Methodology"],
    )
    st.sidebar.header("Controls")
    target = st.sidebar.selectbox(
        "Forecast target",
        ["HHS_Care_Load", "Placement_Demand"],
        format_func=lambda value: {
            "HHS_Care_Load": "Children in HHS care",
            "Placement_Demand": "Placement demand",
        }[value],
    )
    horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
    default_capacity = float(df["HHS_Care_Load"].quantile(0.9))
    capacity_threshold = st.sidebar.number_input(
        "Capacity threshold",
        min_value=0.0,
        value=round(default_capacity, 0),
        step=100.0,
        help="Use the known operational capacity when available. The default is the historical 90th percentile.",
    )
    model_name = st.sidebar.selectbox(
        "Model",
        [
            "Naive Persistence",
            "Moving Average",
            "Exponential Smoothing",
            "ARIMA",
            "SARIMA",
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
        ],
    )
    show_scenarios = st.sidebar.checkbox("Show scenario comparison", value=True)

    st.markdown(
        """
        <div class='brief-banner'>
            <h1>Predictive Forecasting of Care Load &amp; Placement Demand</h1>
            <p>Forecasting HHS care capacity, placement demand, and early operational warning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Dashboard":
        target_label = {
            "HHS_Care_Load": "Children in HHS care",
            "Placement_Demand": "Placement demand",
        }[target]
        st.caption(f"Planning horizon: next {horizon} days | Target: {target_label} | Model: {model_name}")
        st.caption("Data source: uploaded operational CSV" if uploaded_file else "Data source: synthetic demonstration dataset")

    if page == "Executive Summary":
        st.subheader("Executive Summary")
        st.markdown(
            """
            The forecasting app provides a practical early-warning system for care-load planning. In a dynamic child-welfare environment,
            proactive planning is essential. By combining historical demand signals with rolling trends, lag effects, and model-based
            forecasting, stakeholders can anticipate surges, compare possible demand scenarios, and allocate staffing and shelter capacity before
            conditions become critical.
            """
        )
        st.markdown(
            """
            Recommended actions:
            - Use a 14- to 30-day operational forecast for staffing and placement planning
            - Monitor net pressure between transfers and discharges as a leading risk indicator
            - Compare model outputs before making high-impact operational decisions
            """
        )
        return

    if page == "Technical Methodology":
        st.subheader("Technical Methodology")
        st.markdown(
            """
            The modeling workflow includes data validation, time-based splitting, feature engineering, model training, and walk-forward
            evaluation. Feature sets include lag values, rolling means, rolling variance, and a net-pressure indicator that captures the
            difference between transfers into the system and discharges from care.
            """
        )
        st.markdown(
            """
            Key metrics used:
            - MAE: average absolute prediction error
            - RMSE: error metric sensitive to large deviations
            - MAPE: percentage-based relative error for operational interpretation
            """
        )
        return

    with st.spinner("Running forecasts..."):
        result = _cached_model_result(feature_df, target, model_name, horizon)
        forecast_table = _cached_forecast(feature_df, target, horizon, model_name)
        confidence_table = build_confidence_intervals(forecast_table, variation=0.12)
        comparison = _cached_model_comparison(feature_df, target)
        horizon_evaluation = _cached_horizon_evaluation(feature_df, target, model_name)

    st.subheader(f"{target_label} forecast overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train MAE", f"{result.train_mae:.2f}")
    with col2:
        st.metric("Test MAE", f"{result.test_mae:.2f}")
    with col3:
        st.metric("Test MAPE", f"{result.test_mape:.2f}%")

    forecast_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_series = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_table["forecast"].values})
    history_plot = df[["Date", target]].rename(columns={target: "Value"})
    future_plot = forecast_series.rename(columns={"Forecast": "Value"})
    history_plot["Series"] = "Historical"
    future_plot["Series"] = "Forecast"
    plot_df = pd.concat([history_plot, future_plot], ignore_index=True)

    fig = px.line(plot_df, x="Date", y="Value", color="Series", title=f"Historical and projected {target}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confidence interval")
    confidence_fig = px.area(
        confidence_table,
        x=confidence_table.index,
        y=["lower_bound", "upper_bound"],
        title="Forecast confidence band",
    )
    st.plotly_chart(confidence_fig, use_container_width=True)

    st.subheader("Time-series decomposition")
    decomposition = decompose_series(df[target])
    decomposition["Date"] = df.loc[decomposition.index, "Date"].to_numpy()
    decomposition_long = decomposition.melt(id_vars="Date", var_name="Component", value_name="Value")
    decomposition_fig = px.line(
        decomposition_long,
        x="Date",
        y="Value",
        facet_row="Component",
        height=760,
        title="Observed series, trend, weekly seasonality, and residuals",
    )
    decomposition_fig.update_yaxes(matches=None)
    st.plotly_chart(decomposition_fig, use_container_width=True)

    st.subheader("Discharge demand forecast panel")
    placement_forecast = _cached_forecast(
        feature_df,
        "Placement_Demand",
        horizon,
        model_name,
    )
    placement_cols = st.columns(3)
    with placement_cols[0]:
        st.metric("Next-day placement demand", f"{placement_forecast.iloc[0]['forecast']:.0f}")
    with placement_cols[1]:
        st.metric("Average forecast demand", f"{placement_forecast['forecast'].mean():.0f}")
    with placement_cols[2]:
        st.metric("Peak forecast demand", f"{placement_forecast['forecast'].max():.0f}")
    placement_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    placement_plot = pd.DataFrame({"Date": placement_dates, "Placement demand": placement_forecast["forecast"]})
    placement_fig = px.line(placement_plot, x="Date", y="Placement demand", title="Projected discharge and placement demand", markers=True)
    st.plotly_chart(placement_fig, use_container_width=True)

    kpis = calculate_kpis(result.actual, result.predictions)
    st.subheader("Key performance indicators")
    operational_kpis = calculate_operational_kpis(
        df[target],
        forecast_table["forecast"],
        capacity_threshold=capacity_threshold,
        validation_accuracy=100.0 - result.test_mape,
        fold_mae_std=result.fold_mae_std,
    )
    all_kpis = {**kpis, **operational_kpis}
    surge_value = operational_kpis["Surge Lead Time (days)"]
    recent_baseline = df[target].tail(7).mean()
    surge_threshold = recent_baseline * 1.15
    if isinstance(surge_value, str):
        st.info(
            f"No surge warning was triggered. The forecast stays below the surge threshold of "
            f"{surge_threshold:.0f}, which is 15% above the recent 7-day baseline of {recent_baseline:.0f}."
        )
    metric_cols = st.columns(len(all_kpis))
    for idx, (name, value) in enumerate(all_kpis.items()):
        with metric_cols[idx]:
            st.metric(name, value)
    st.caption(
        f"Surge Lead Time counts the first forecast day at least 15% above the recent 7-day baseline "
        f"({recent_baseline:.0f}; threshold {surge_threshold:.0f}). No crossing means no surge is expected within the selected horizon."
    )

    st.subheader("Forecast horizon detail")
    st.dataframe(confidence_table.rename(columns={"step": "Day", "forecast": "Forecast"}).assign(Forecast=lambda x: x["Forecast"].round(2)))

    st.subheader("Model comparison")
    st.dataframe(comparison)

    st.subheader("Multi-horizon evaluation")
    st.dataframe(horizon_evaluation)

    if show_scenarios:
        st.subheader("Scenario comparison")
        scenario_df = pd.DataFrame(
            {
                "Day": forecast_table["step"],
                "Expected demand": forecast_table["forecast"],
                "Intake surge (+15%)": forecast_table["forecast"] * 1.15,
                "Constrained discharges (+10%)": forecast_table["forecast"] * 1.10,
            }
        )
        scenario_long = scenario_df.melt(id_vars="Day", var_name="Scenario", value_name="Forecast")
        scenario_fig = px.line(scenario_long, x="Day", y="Forecast", color="Scenario", title="Operational demand scenarios")
        st.plotly_chart(scenario_fig, use_container_width=True)
        st.caption("Scenario multipliers are planning assumptions for sensitivity analysis, not observed operational outcomes.")

    st.subheader("Operational interpretation")
    st.markdown(
        """
        The model output should be treated as an operational planning signal rather than a perfect prediction. Trend-based changes,
        capacity thresholds, and staff availability should be considered alongside the forecast when making placement and shelter decisions.
        """
    )


if __name__ == "__main__":
    render_dashboard()
