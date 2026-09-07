from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _normalise_column_name(column: str) -> str:
    return "".join(character.lower() for character in str(column) if character.isalnum())


def _normalise_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both the app's short names and the brief's descriptive names."""
    aliases = {
        "date": "Date",
        "childrenapprehendedandplacedincbpcustody": "Children_Apprehended_CBP",
        "childrenapprehendedcbp": "Children_Apprehended_CBP",
        "childrenincbpcustody": "CBP_Care_Load",
        "cbpcareload": "CBP_Care_Load",
        "childrentransferredoutofcbpcustody": "Transfers_to_HHS",
        "transferstohhs": "Transfers_to_HHS",
        "childreninhhscare": "HHS_Care_Load",
        "hhscareload": "HHS_Care_Load",
        "childrendischargefromhhscare": "Discharges_from_HHS",
        "childrendischargedfromhhscare": "Discharges_from_HHS",
        "dischargesfromhhs": "Discharges_from_HHS",
        "placementdemand": "Placement_Demand",
    }
    renamed = {}
    for column in df.columns:
        target = aliases.get(_normalise_column_name(column))
        if target is not None:
            renamed[column] = target
    out = df.rename(columns=renamed).copy()
    if "Placement_Demand" not in out.columns and "Discharges_from_HHS" in out.columns:
        out["Placement_Demand"] = out["Discharges_from_HHS"]
    return out


def load_dataset(csv_path: str | Path | object | None = None) -> pd.DataFrame:
    """Load the UAC forecasting dataset and validate the required columns."""
    path = csv_path if csv_path is not None else DATA_DIR / "sample_uac_forecasting.csv"
    df = _normalise_dataset_columns(pd.read_csv(path))

    required_columns = {
        "Date",
        "Children_Apprehended_CBP",
        "CBP_Care_Load",
        "Transfers_to_HHS",
        "HHS_Care_Load",
        "Discharges_from_HHS",
        "Placement_Demand",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        available = sorted(str(column) for column in df.columns)
        raise ValueError(f"Dataset is missing required columns: {missing}. Available columns: {available}")

    df["Date"] = pd.to_datetime(df["Date"])
    numeric_columns = [column for column in required_columns if column != "Date"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    invalid_numeric = [column for column in numeric_columns if df[column].isna().all()]
    if invalid_numeric:
        raise ValueError(f"Dataset numeric columns contain no usable numbers: {invalid_numeric}")

    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    daily_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(daily_index)
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].interpolate(method="linear").ffill().bfill()
    df.index.name = "Date"
    df = df.reset_index()
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature dataset for forecasting HHS care load and placement demand."""
    out = df.copy()
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year
    out["is_weekend"] = (out["Date"].dt.dayofweek >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(out["Date"].min(), out["Date"].max())
    out["is_holiday"] = out["Date"].isin(holidays).astype(int)

    for col in ["HHS_Care_Load", "Placement_Demand", "Transfers_to_HHS", "Discharges_from_HHS"]:
        for lag in [1, 7, 14]:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in [7, 14]:
            out[f"{col}_rolling_mean_{window}"] = out[col].shift(1).rolling(window).mean()
            out[f"{col}_rolling_var_{window}"] = out[col].shift(1).rolling(window).var().fillna(0)
            out[f"{col}_rolling_std_{window}"] = out[col].shift(1).rolling(window).std().fillna(0)

    out["net_pressure"] = out["Transfers_to_HHS"] - out["Discharges_from_HHS"]
    out = out.dropna().reset_index(drop=True)
    return out
