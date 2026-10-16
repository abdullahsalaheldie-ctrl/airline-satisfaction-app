"""Artifact loading, input validation and prediction (shared by every page)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocessing as pp

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_FILES = {"xgboost": "xgboost_model.joblib", "lightgbm": "lightgbm_model.joblib"}
MODEL_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}


@dataclass
class Artifacts:
    schema: dict | None = None
    models: dict = field(default_factory=dict)
    report: dict | None = None
    errors: dict = field(default_factory=dict)

    @property
    def available(self) -> list[str]:
        return [k for k in MODEL_FILES if k in self.models]


def load_artifacts(models_dir: Path = MODELS_DIR) -> Artifacts:
    """Load schema, both models and the training report. Never raises: problems are
    recorded in `errors` so the UI can say exactly what is missing."""
    art = Artifacts()
    try:
        art.schema = joblib.load(models_dir / "preprocessing_pipeline.joblib")
        if art.schema.get("feature_order") != pp.FEATURE_ORDER:
            art.errors["schema"] = "Saved feature order differs from src/preprocessing.py - retrain the models."
            art.schema = None
    except Exception as exc:  # noqa: BLE001
        art.errors["schema"] = f"Could not load preprocessing_pipeline.joblib ({type(exc).__name__})."
    try:
        art.report = json.loads((models_dir / "training_report.json").read_text())
    except Exception as exc:  # noqa: BLE001
        art.errors["report"] = f"Could not load training_report.json ({type(exc).__name__})."
    for key, fname in MODEL_FILES.items():
        try:
            model = joblib.load(models_dir / fname)
            if list(model.classes_) != [0, 1]:
                raise ValueError("unexpected class order")
            # LightGBM replaces spaces with underscores in stored names; compare on that basis.
            norm = lambda names: [str(n).replace(" ", "_") for n in names]  # noqa: E731
            if norm(getattr(model, "feature_names_in_", pp.FEATURE_ORDER)) != norm(pp.FEATURE_ORDER):
                raise ValueError("feature names differ from schema")
            art.models[key] = model
        except Exception as exc:  # noqa: BLE001
            art.errors[key] = f"Could not load {fname} ({type(exc).__name__}: {exc})."
    return art


def validate_input(raw: dict, schema: dict | None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation."""
    errors, warnings = [], []
    for col in pp.RAW_INPUT_COLS:
        if col not in raw or raw[col] is None:
            errors.append(f"'{col}' is required.")
    if errors:
        return errors, warnings
    for col, mapping in pp.CATEGORICAL_MAPPINGS.items():
        if raw[col] not in mapping:
            errors.append(f"'{col}' has an unsupported value: {raw[col]}.")
    for col in pp.SERVICE_COLS:
        if raw[col] not in range(0, 6):
            errors.append(f"'{col}' must be a whole number from 0 to 5.")
    for col in ["Age", "Flight Distance", *pp.DELAY_COLS]:
        if raw[col] < 0:
            errors.append(f"'{col}' cannot be negative.")
    if not errors and schema:
        ranges = schema["feature_ranges"]
        for col in ["Age", "Flight Distance", *pp.DELAY_COLS]:
            lo, hi = ranges[col]
            if raw[col] < lo or raw[col] > hi:
                warnings.append(
                    f"{col} = {raw[col]:g} is outside the range seen in training data ({lo:g}-{hi:g}); "
                    "the prediction is an extrapolation and should be treated with extra caution."
                )
        if raw["Arrival Delay in Minutes"] > raw["Departure Delay in Minutes"] + 600:
            warnings.append("Arrival delay is much larger than departure delay; please double-check the values.")
    return errors, warnings


def to_features(raw: dict) -> pd.DataFrame:
    return pp.build_features(pd.DataFrame([raw]))


def predict(art: Artifacts, key: str, raw: dict) -> dict:
    """Run one passenger through one model."""
    if key not in art.models:
        raise KeyError(f"Model '{key}' is not available: {art.errors.get(key, 'not loaded')}")
    X = to_features(raw)
    proba = art.models[key].predict_proba(X)[0]
    p_sat = float(proba[1])
    cls = int(p_sat >= 0.5)
    return {
        "model_key": key,
        "model": MODEL_NAMES[key],
        "class_id": cls,
        "label": pp.TARGET_LABELS[cls],
        "p_satisfied": p_sat,
        "p_dissatisfied": float(proba[0]),
        "confidence": float(max(p_sat, 1 - p_sat)),
        "features": X,
    }


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts() -> Artifacts:
    """Loaded once per server process; reruns never reload or retrain anything."""
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def get_explainer(model_key: str):
    from src.explainability import make_explainer

    return make_explainer(get_artifacts().models[model_key])
