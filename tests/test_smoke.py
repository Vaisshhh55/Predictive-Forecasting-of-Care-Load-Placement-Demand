from io import StringIO

from app.data_loader import load_dataset, build_feature_frame
from app.forecasting import create_forecast_table, decompose_series, evaluate_horizons


def test_sample_dataset_loads():
    df = load_dataset()
    assert list(df.columns) == [
        "Date",
        "Children_Apprehended_CBP",
        "CBP_Care_Load",
        "Transfers_to_HHS",
        "HHS_Care_Load",
        "Discharges_from_HHS",
        "Placement_Demand",
    ]
    assert len(df) > 0


def test_feature_building_works():
    df = load_dataset()
    features = build_feature_frame(df)
    assert "HHS_Care_Load_lag_7" in features.columns
    assert "net_pressure" in features.columns
    assert "is_holiday" in features.columns
    assert len(features) > 0


def test_required_forecasting_models_generate_future_values():
    features = build_feature_frame(load_dataset())
    model_names = [
        "Naive Persistence",
        "Moving Average",
        "Exponential Smoothing",
        "ARIMA",
        "SARIMA",
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
    ]

    for model_name in model_names:
        forecast = create_forecast_table(features, "HHS_Care_Load", horizon=5, model_name=model_name)
        assert len(forecast) == 5
        assert forecast["forecast"].notna().all()
        assert (forecast["forecast"] >= 0).all()


def test_required_analysis_outputs_exist():
    dataset = load_dataset()
    features = build_feature_frame(dataset)
    decomposition = decompose_series(dataset["HHS_Care_Load"])
    horizon_results = evaluate_horizons(features, "HHS_Care_Load", "Naive Persistence")

    assert {"Observed", "Trend", "Seasonality", "Residual"}.issubset(decomposition.columns)
    assert list(horizon_results["Horizon (days)"]) == [7, 30, 90]
    assert {"MAE", "RMSE", "MAPE (%)"}.issubset(horizon_results.columns)


def test_brief_dataset_column_names_are_supported():
    source = StringIO(
        "Date,Children apprehended and placed in CBP custody,Children in CBP custody,"
        "Children transferred out of CBP custody,Children in HHS Care,"
        "Children discharged from HHS Care\n"
        "2024-01-01,10,20,5,30,4\n"
        "2024-01-02,11,21,6,31,5\n"
        "2024-01-03,12,22,7,32,6\n"
    )
    dataset = load_dataset(source)

    assert "HHS_Care_Load" in dataset.columns
    assert "Placement_Demand" in dataset.columns
    assert dataset["Placement_Demand"].tolist() == [4, 5, 6]


def test_numeric_text_columns_are_converted_for_feature_engineering():
    source = StringIO(
        "Date,Children_Apprehended_CBP,CBP_Care_Load,Transfers_to_HHS,"
        "HHS_Care_Load,Discharges_from_HHS,Placement_Demand\n"
        "2024-01-01,1,\"1,000\",5,\"2,000\",4,7\n"
        "2024-01-02,2,1001,6,2001,5,8\n"
        "2024-01-03,3,1002,7,2002,6,9\n"
    )
    features = build_feature_frame(load_dataset(source))

    assert features["CBP_Care_Load"].dtype.kind in "fi"
