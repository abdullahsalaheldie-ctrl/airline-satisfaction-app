"""Preprocessing and feature engineering.

This module reproduces, step by step, the pipeline used in the original notebook
(`predication_air.ipynb`), and is the single code path used for BOTH training and
inference so that the two can never drift apart.

Notebook steps reproduced here:
1. drop_duplicates, then dropna
2. Label-encode Gender / Customer Type / Type of Travel / Class with fixed mappings
3. Drop the `id` column (identifier, not a legitimate predictor)
4. Engineer Total_Delay, Delay_Diff, Average_Service_Rating
"""
from __future__ import annotations

import pandas as pd

TARGET_COL = "satisfaction"
ID_COL = "id"

SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
DELAY_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
ENGINEERED_COLS = ["Total_Delay", "Delay_Diff", "Average_Service_Rating"]

# Exactly the mappings used in the notebook.
CATEGORICAL_MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "Customer Type": {"Loyal Customer": 0, "disloyal Customer": 1},
    "Type of Travel": {"Business travel": 0, "Personal Travel": 1},
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2},
}

# Column order the models were trained on (notebook order after dropping id/target).
FEATURE_ORDER = (
    ["Age", "Flight Distance"]
    + SERVICE_COLS
    + DELAY_COLS
    + list(CATEGORICAL_MAPPINGS)
    + ENGINEERED_COLS
)

# Target encoding: `satisfaction` is boolean in train.csv. The notebook's dataset
# dictionary lists the labels as (`satisfied`, `neutral or dissatisfied`), so
# True / 1 = satisfied.
TARGET_LABELS = {1: "Satisfied", 0: "Neutral or Dissatisfied"}
POSITIVE_CLASS = 1

RAW_INPUT_COLS = ["Age", "Flight Distance"] + SERVICE_COLS + DELAY_COLS + list(CATEGORICAL_MAPPINGS)


def load_raw(path) -> pd.DataFrame:
    """Read the raw training file (.csv, or a .zip containing one csv)."""
    return pd.read_csv(path)


def encode_target(series: pd.Series) -> pd.Series:
    """Return the target as int8 (1 = satisfied, 0 = neutral or dissatisfied)."""
    if series.dtype == bool:
        return series.astype("int8")
    s = series.astype(str).str.strip().str.lower()
    values = set(s.unique())
    if values <= {"true", "false"}:
        return (s == "true").astype("int8")
    if values <= {"satisfied", "neutral or dissatisfied"}:
        return (s == "satisfied").astype("int8")
    raise ValueError(f"Unrecognised target values: {sorted(values)[:5]}")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Notebook cleaning: drop duplicate rows, then rows with missing values."""
    n0 = len(df)
    dedup = df.drop_duplicates()
    out = dedup.dropna()
    report = {
        "rows_before": int(n0),
        "duplicates_removed": int(n0 - len(dedup)),
        "rows_with_missing_removed": int(len(dedup) - len(out)),
        "rows_after": int(len(out)),
    }
    return out.reset_index(drop=True), report


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Total_Delay, Delay_Diff and Average_Service_Rating, as in the notebook."""
    out = df.copy()
    dep, arr = out["Departure Delay in Minutes"], out["Arrival Delay in Minutes"]
    out["Total_Delay"] = dep + arr
    out["Delay_Diff"] = arr - dep
    out["Average_Service_Rating"] = out[SERVICE_COLS].mean(axis=1)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw passenger rows (original string categoricals) -> model matrix.

    Raises ValueError for missing columns, missing values or unknown categories so
    bad input never silently becomes NaN.
    """
    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[RAW_INPUT_COLS].copy()
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        encoded = out[col].map(mapping)
        if encoded.isna().any():
            bad = out.loc[encoded.isna(), col].unique().tolist()
            raise ValueError(f"Unknown value(s) for '{col}': {bad}. Allowed: {list(mapping)}")
        out[col] = encoded.astype("int64")
    if out.isna().any().any():
        raise ValueError("Input contains missing values.")
    out = add_engineered_features(out)
    return out[FEATURE_ORDER]


def data_quality_report(raw: pd.DataFrame) -> dict:
    """Facts about the raw file that the Overview page displays."""
    miss = raw.isnull().sum()
    target = encode_target(raw[TARGET_COL].dropna())
    return {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "missing_by_column": {c: int(v) for c, v in miss.items() if v > 0},
        "missing_total": int(miss.sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(raw.drop(columns=[ID_COL], errors="ignore").duplicated().sum()),
        "target_counts_raw": {
            "satisfied": int(target.sum()),
            "neutral_or_dissatisfied": int((1 - target).sum()),
        },
        "rating_zero_counts": {c: int((raw[c] == 0).sum()) for c in SERVICE_COLS},
    }


def prepare_training_data(path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Full notebook pipeline from the raw file to (X, y, quality report)."""
    raw = load_raw(path)
    quality = data_quality_report(raw)
    cleaned, clean_report = clean(raw)
    quality["cleaning"] = clean_report
    y = encode_target(cleaned[TARGET_COL])
    X = build_features(cleaned)
    return X, y, quality
