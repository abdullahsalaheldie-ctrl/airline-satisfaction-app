"""Reproducible training, held-out evaluation and artifact saving.

Run via `python train_models.py --data path/to/train.csv`.

Methodology (identical for both models):
* Notebook cleaning + feature engineering (see preprocessing.py)
* Stratified 80/20 split, random_state=42 (the notebook used an unstratified split)
* Hyperparameters are the notebook's fixed values; nothing is tuned on the test set
* Models are fitted on the 80% training split ONLY, so every reported metric is
  computed on data the model never saw (the notebook's final LightGBM was refit on
  100% of the data, which leaves no honest hold-out - that step is not repeated here)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import preprocessing as pp

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparameters copied verbatim from the notebook.
MODEL_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)


def make_models() -> dict:
    return {
        "xgboost": XGBClassifier(**MODEL_PARAMS, n_jobs=-1, eval_metric="logloss"),
        "lightgbm": LGBMClassifier(**MODEL_PARAMS, n_jobs=-1, verbose=-1),
    }


def _timing(model, X_test: pd.DataFrame) -> dict:
    model.predict_proba(X_test.head(1000))  # warm-up
    full = []
    for _ in range(3):
        t = time.perf_counter()
        model.predict_proba(X_test)
        full.append(time.perf_counter() - t)
    single = []
    row = X_test.head(1)
    for _ in range(50):
        t = time.perf_counter()
        model.predict_proba(row)
        single.append(time.perf_counter() - t)
    return {
        "inference_ms_per_1000_rows": float(np.median(full) / len(X_test) * 1000 * 1000),
        "single_row_latency_ms": float(np.median(single) * 1000),
    }


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    assert list(model.classes_) == [0, 1], "Unexpected class order"
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, pred)
    return {
        "accuracy": float(test_acc),
        "precision_satisfied": float(precision_score(y_test, pred)),
        "recall_satisfied": float(recall_score(y_test, pred)),
        "f1_satisfied": float(f1_score(y_test, pred)),
        "precision_dissatisfied": float(precision_score(y_test, pred, pos_label=0)),
        "recall_dissatisfied": float(recall_score(y_test, pred, pos_label=0)),
        "f1_dissatisfied": float(f1_score(y_test, pred, pos_label=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_accuracy": float(train_acc),
        "generalization_gap": float(train_acc - test_acc),
        "roc_curve": {"fpr": fpr[idx].round(5).tolist(), "tpr": tpr[idx].round(5).tolist()},
        "n_test": int(len(y_test)),
    }


def global_importance(model, key: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Three views of global importance, all normalised to shares that sum to 1."""
    cols = list(X_test.columns)
    # 1. native gain importance
    if key == "xgboost":
        gain = model.get_booster().get_score(importance_type="gain")
        native = np.array([gain.get(c, 0.0) for c in cols], dtype=float)
    else:
        native = model.booster_.feature_importance(importance_type="gain").astype(float)
    # 2. mean |SHAP| on a held-out sample
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
    Xs = X_test.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    shap_abs = np.abs(sv).mean(axis=0)
    # 3. permutation importance (ROC-AUC drop) on a held-out sample
    pidx = rng.choice(len(X_test), size=min(8000, len(X_test)), replace=False)
    perm = permutation_importance(
        model, X_test.iloc[pidx], y_test.iloc[pidx], scoring="roc_auc", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1
    ).importances_mean.clip(min=0)

    def norm(a):
        return (a / a.sum()).round(6).tolist() if a.sum() > 0 else a.tolist()

    return {"features": cols, "native_gain": norm(native), "shap_mean_abs": norm(shap_abs), "permutation_auc_drop": norm(perm)}


def train_and_save(data_path, models_dir) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, quality = pp.prepare_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    schema = {
        "feature_order": pp.FEATURE_ORDER,
        "raw_input_columns": pp.RAW_INPUT_COLS,
        "categorical_mappings": pp.CATEGORICAL_MAPPINGS,
        "service_cols": pp.SERVICE_COLS,
        "target_labels": pp.TARGET_LABELS,
        "positive_class": pp.POSITIVE_CLASS,
        "feature_ranges": {
            c: [float(X[c].min()), float(X[c].max())] for c in pp.FEATURE_ORDER
        },
    }
    joblib.dump(schema, models_dir / "preprocessing_pipeline.joblib")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {
            "method": "stratified train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_positive_rate": float(y_test.mean()),
        },
        "hyperparameters": MODEL_PARAMS,
        "data_quality": quality,
        "models": {},
        "importance": {},
    }

    for key, model in make_models().items():
        t = time.perf_counter()
        model.fit(X_train, y_train)
        train_s = time.perf_counter() - t
        path = models_dir / f"{key}_model.joblib"
        joblib.dump(model, path, compress=3)
        metrics = evaluate(model, X_train, y_train, X_test, y_test)
        metrics["training_time_s"] = float(train_s)
        metrics.update(_timing(model, X_test))
        metrics["model_size_mb"] = float(path.stat().st_size / 1e6)
        report["models"][key] = metrics
        report["importance"][key] = global_importance(model, key, X_test, y_test)
        print(f"{key}: acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} f1={metrics['f1_satisfied']:.4f}")

    # Agreement between the two models on the held-out set
    p = {k: joblib.load(models_dir / f"{k}_model.joblib").predict(X_test) for k in ("xgboost", "lightgbm")}
    report["agreement_rate_on_test"] = float((p["xgboost"] == p["lightgbm"]).mean())

    with open(models_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report
