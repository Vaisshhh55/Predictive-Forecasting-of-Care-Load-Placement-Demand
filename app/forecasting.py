from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


@dataclass
class ForecastResult:
    model_name: str
    train_mape: float
    test_mape: float
    train_mae: float
    test_mae: float
    test_rmse: float
    fold_mae_std: float
    predictions: pd.Series
    actual: pd.Series
    forecast_horizon: int


def _safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def _build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "linear regression":
        return LinearRegression()
    if model_name == "random forest":
        return RandomForestRegressor(n_estimators=200, random_state=42)
    if model_name == "gradient boosting":
        return GradientBoostingRegressor(random_state=42)
    return LinearRegression()


def _is_statistical_model(model_name: str) -> bool:
    return model_name.lower() in {
        "naive persistence",
        "moving average",
        "exponential smoothing",
        "arima",
        "sarima",
    }


def _statistical_forecast(values: pd.Series, model_name: str, horizon: int) -> np.ndarray:
    values = values.astype(float).reset_index(drop=True)
    normalized_name = model_name.lower()
    if normalized_name == "naive persistence":
        return np.repeat(values.iloc[-1], horizon)
    if normalized_name == "moving average":
        return np.repeat(values.tail(7).mean(), horizon)
    if normalized_name == "exponential smoothing":
        model = ExponentialSmoothing(values, trend="add", seasonal=None).fit(optimized=True)
        return model.forecast(horizon).to_numpy()

    seasonal_order = (1, 0, 1, 7) if normalized_name == "sarima" else (0, 0, 0, 0)
    model = ARIMA(values, order=(1, 1, 1), seasonal_order=seasonal_order).fit()
    return model.forecast(horizon).to_numpy()


def _future_feature_frame(history: pd.DataFrame, target: str, model, horizon: int) -> pd.DataFrame:
    """Build recursive future features using known signals and predicted target values."""
    feature_cols = [col for col in history.columns if col not in {"Date", "HHS_Care_Load", "Placement_Demand"}]
    source_columns = ["HHS_Care_Load", "Placement_Demand", "Transfers_to_HHS", "Discharges_from_HHS"]
    values = {column: history[column].astype(float).tolist() for column in source_columns}
    dates = list(history["Date"])
    rows = []

    for step in range(1, horizon + 1):
        next_date = dates[-1] + pd.Timedelta(days=1)
        dates.append(next_date)
        row = history.iloc[-1].copy()
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row["year"] = next_date.year
        row["is_weekend"] = int(next_date.dayofweek >= 5)
        row["is_holiday"] = 0

        for column in source_columns:
            series = values[column]
            if column == target:
                continue
            series.append(series[-1])
            for lag in (1, 7, 14):
                row[f"{column}_lag_{lag}"] = series[-lag]
            for window in (7, 14):
                window_values = series[-window:]
                row[f"{column}_rolling_mean_{window}"] = np.mean(window_values)
                row[f"{column}_rolling_var_{window}"] = np.var(window_values)
                row[f"{column}_rolling_std_{window}"] = np.std(window_values)

        target_series = values[target]
        for lag in (1, 7, 14):
            row[f"{target}_lag_{lag}"] = target_series[-lag]
        for window in (7, 14):
            window_values = target_series[-window:]
            row[f"{target}_rolling_mean_{window}"] = np.mean(window_values)
            row[f"{target}_rolling_var_{window}"] = np.var(window_values)
            row[f"{target}_rolling_std_{window}"] = np.std(window_values)
        row["net_pressure"] = values["Transfers_to_HHS"][-1] - values["Discharges_from_HHS"][-1]

        prediction = max(0.0, float(model.predict(pd.DataFrame([row])[feature_cols])[0]))
        values[target].append(prediction)
        rows.append(row[feature_cols].to_numpy(dtype=float))

    return pd.DataFrame(rows, columns=feature_cols)


def train_and_evaluate_model(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    horizon: int = 30,
    n_splits: int = 3,
) -> ForecastResult:
    """Train a time-based model and evaluate it with walk-forward validation."""
    feature_cols = [col for col in df.columns if col not in {"Date", "HHS_Care_Load", "Placement_Demand"}]
    X = df[feature_cols]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_errors = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if _is_statistical_model(model_name):
            pred = _statistical_forecast(y_train, model_name, len(test_idx))
        else:
            model = _build_model(model_name)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        fold_errors.append(
            {
                "mae": mean_absolute_error(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "mape": _safe_mape(y_test, pred),
            }
        )

    fold_df = pd.DataFrame(fold_errors)
    test_mae = float(fold_df["mae"].mean())
    test_rmse = float(fold_df["rmse"].mean())
    test_mape = float(fold_df["mape"].mean())
    fold_mae_std = float(fold_df["mae"].std(ddof=0))

    split = max(1, len(df) - horizon)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    y_actual = y.iloc[split:]

    if _is_statistical_model(model_name):
        predictions = pd.Series(_statistical_forecast(y_train, model_name, len(y_actual)), index=y_actual.index)
        train_pred = np.repeat(y_train.iloc[-1], len(y_train))
    else:
        model = _build_model(model_name)
        model.fit(X_train, y_train)
        predictions = pd.Series(model.predict(X.iloc[split:]), index=y_actual.index)
        train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_mape = _safe_mape(y_train, train_pred)

    return ForecastResult(
        model_name=model_name,
        train_mape=train_mape,
        test_mape=test_mape,
        train_mae=train_mae,
        test_mae=test_mae,
        test_rmse=test_rmse,
        fold_mae_std=fold_mae_std,
        predictions=predictions,
        actual=y_actual,
        forecast_horizon=horizon,
    )


def create_forecast_table(
    df: pd.DataFrame,
    target: str,
    horizon: int = 30,
    model_name: str = "Exponential Smoothing",
) -> pd.DataFrame:
    """Create future forecasts with the selected model."""
    if _is_statistical_model(model_name):
        predictions = _statistical_forecast(df[target], model_name, horizon)
    else:
        feature_frame = df if "net_pressure" in df.columns else df.copy()
        model_features = [column for column in feature_frame.columns if column not in {"Date", "HHS_Care_Load", "Placement_Demand"}]
        model = _build_model(model_name)
        model.fit(feature_frame[model_features], feature_frame[target])
        future_features = _future_feature_frame(feature_frame, target, model, horizon)
        predictions = model.predict(future_features)

    return pd.DataFrame({"step": range(1, horizon + 1), "forecast": np.maximum(0, np.round(predictions, 2))})


def calculate_kpis(actual: pd.Series, predicted: pd.Series) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = _safe_mape(actual, predicted)
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE": round(float(mape), 3),
    }


def calculate_operational_kpis(
    historical: pd.Series,
    forecast: pd.Series,
    capacity_threshold: float | None = None,
    validation_accuracy: float | None = None,
    fold_mae_std: float | None = None,
) -> dict:
    """Calculate brief-specific reliability and capacity warning indicators."""
    historical = historical.astype(float)
    forecast = forecast.astype(float)
    threshold = capacity_threshold or float(historical.quantile(0.9))
    breach_steps = np.flatnonzero(forecast.to_numpy() >= threshold)
    baseline = float(historical.tail(7).mean())
    surge_steps = np.flatnonzero(forecast.to_numpy() >= baseline * 1.15)
    changes = forecast.diff().dropna()
    stability = max(0.0, 100.0 - float(changes.std(ddof=0) / max(baseline, 1.0) * 100.0))
    return {
        "Forecast Accuracy (%)": round(max(0.0, validation_accuracy if validation_accuracy is not None else 100.0 - _safe_mape(historical.tail(min(len(historical), len(forecast))), forecast.tail(min(len(historical), len(forecast))))), 2),
        "Surge Lead Time (days)": int(surge_steps[0] + 1) if len(surge_steps) else "No surge expected in selected horizon",
        "Capacity Breach Probability (%)": round(float(len(breach_steps) / max(len(forecast), 1) * 100.0), 2),
        "Forecast Stability Index": round(stability, 2),
        "Model Robustness": round(max(0.0, 100.0 - (fold_mae_std or 0.0) / max(baseline, 1.0) * 100.0), 2),
    }


def decompose_series(values: pd.Series, period: int = 7) -> pd.DataFrame:
    """Return observed, trend, seasonal, and residual components."""
    clean_values = values.astype(float).reset_index(drop=True)
    if len(clean_values) < period * 2:
        raise ValueError(f"At least {period * 2} observations are required for decomposition.")
    decomposition = seasonal_decompose(
        clean_values,
        model="additive",
        period=period,
        extrapolate_trend=period,
    )
    return pd.DataFrame(
        {
            "Observed": decomposition.observed,
            "Trend": decomposition.trend,
            "Seasonality": decomposition.seasonal,
            "Residual": decomposition.resid,
        }
    ).dropna()


def evaluate_horizons(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    horizons: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Evaluate the selected model across short, medium, and long horizons."""
    rows = []
    for horizon in horizons:
        result = train_and_evaluate_model(df, target, model_name, horizon=horizon)
        rows.append(
            {
                "Horizon (days)": horizon,
                "MAE": round(result.test_mae, 2),
                "RMSE": round(result.test_rmse, 2),
                "MAPE (%)": round(result.test_mape, 2),
            }
        )
    return pd.DataFrame(rows)


def build_confidence_intervals(forecast_table: pd.DataFrame, variation: float = 0.08) -> pd.DataFrame:
    """Add upper and lower confidence bound estimates around the forecast."""
    table = forecast_table.copy()
    table["lower_bound"] = table["forecast"] * (1 - variation)
    table["upper_bound"] = table["forecast"] * (1 + variation)
    return table


def compare_model_performance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    results = []
    for model_name in [
        "Naive Persistence",
        "Moving Average",
        "Exponential Smoothing",
        "ARIMA",
        "SARIMA",
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
    ]:
        result = train_and_evaluate_model(df, target, model_name, horizon=30)
        results.append(
            {
                "Model": model_name,
                "Train MAE": round(float(result.train_mae), 2),
                "Test MAE": round(float(result.test_mae), 2),
                "Test RMSE": round(float(result.test_rmse), 2),
                "Test MAPE": round(float(result.test_mape), 2),
                "MAE Stability": round(float(result.fold_mae_std), 2),
            }
        )
    return pd.DataFrame(results)
