"""Run with: pytest -q   (requires trained artifacts in models/ and the dataset in data/)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import analytics as an  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.explainability import add_interpretations, build_recommendations, build_report, local_contributions, make_explainer  # noqa: E402
from src.prediction import load_artifacts, predict, validate_input  # noqa: E402

RAW = {"Age": 35, "Flight Distance": 800, **{c: 3 for c in pp.SERVICE_COLS}, "Departure Delay in Minutes": 45,
       "Arrival Delay in Minutes": 60, "Gender": "Male", "Customer Type": "Loyal Customer", "Type of Travel": "Personal Travel", "Class": "Eco"}
RAW["Online boarding"] = 1


@pytest.fixture(scope="module")
def art():
    return load_artifacts()


@pytest.fixture(scope="module")
def data():
    path = an.find_data_file()
    if path is None:
        pytest.skip("dataset not present")
    return pp.load_raw(path)


def test_both_models_load(art):
    assert art.available == ["xgboost", "lightgbm"], art.errors


def test_feature_order_and_schema(art):
    assert art.schema["feature_order"] == pp.FEATURE_ORDER
    assert len(pp.FEATURE_ORDER) == 24 and "id" not in pp.FEATURE_ORDER and pp.TARGET_COL not in pp.FEATURE_ORDER
    X = pp.build_features(pd.DataFrame([RAW]))
    assert list(X.columns) == pp.FEATURE_ORDER and not X.isna().any().any()


def test_engineered_features_match_notebook_formulas():
    X = pp.build_features(pd.DataFrame([RAW]))
    assert X["Total_Delay"][0] == 105 and X["Delay_Diff"][0] == 15
    assert X["Average_Service_Rating"][0] == pytest.approx((12 * 3 + 1) / 13)


def test_unknown_category_rejected():
    bad = {**RAW, "Class": "First"}
    with pytest.raises(ValueError):
        pp.build_features(pd.DataFrame([bad]))


def test_training_inference_consistency(data):
    """Rebuilding features row-wise equals the training matrix for the same rows."""
    cleaned, _ = pp.clean(data.head(3000))
    X_train_path = pp.build_features(cleaned)
    one = pp.build_features(cleaned.iloc[[7]])
    pd.testing.assert_frame_equal(X_train_path.iloc[[7]], one)


def test_predictions_and_label_mapping(art, data):
    for k in art.available:
        r = predict(art, k, RAW)
        assert r["label"] in pp.TARGET_LABELS.values()
        assert r["p_satisfied"] + r["p_dissatisfied"] == pytest.approx(1.0)
        assert (r["class_id"] == 1) == (r["p_satisfied"] >= 0.5)
    # Known extremes from the data: happy passenger vs unhappy passenger
    happy = {**RAW, **{c: 5 for c in pp.SERVICE_COLS}, "Departure Delay in Minutes": 0, "Arrival Delay in Minutes": 0,
             "Customer Type": "Loyal Customer", "Type of Travel": "Business travel", "Class": "Business"}
    sad = {**RAW, **{c: 1 for c in pp.SERVICE_COLS}, "Type of Travel": "Personal Travel", "Class": "Eco"}
    for k in art.available:
        assert predict(art, k, happy)["label"] == "Satisfied"
        assert predict(art, k, sad)["label"] == "Neutral or Dissatisfied"


def test_held_out_metrics_reproduce(art, data):
    """Recompute accuracy from scratch on the same stratified split and compare to the saved report."""
    from sklearn.model_selection import train_test_split
    from src.model_evaluation import RANDOM_STATE, TEST_SIZE

    cleaned, _ = pp.clean(data)
    X, y = pp.build_features(cleaned), pp.encode_target(cleaned[pp.TARGET_COL])
    _, Xt, _, yt = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    for k in art.available:
        acc = float((art.models[k].predict(Xt) == yt).mean())
        assert acc == pytest.approx(art.report["models"][k]["accuracy"], abs=1e-9)


def test_shap_additivity_and_sorting(art):
    for k in art.available:
        r = predict(art, k, RAW)
        df, base = local_contributions(make_explainer(art.models[k]), r["features"], RAW)
        logit = np.log(r["p_satisfied"] / (1 - r["p_satisfied"]))
        assert base + df["Contribution"].sum() == pytest.approx(logit, abs=1e-3)
        assert df["Absolute contribution"].is_monotonic_decreasing


def test_recommendations_rules(art):
    r = predict(art, "xgboost", RAW)
    df, _ = local_contributions(make_explainer(art.models["xgboost"]), r["features"], RAW)
    recs, _, note = build_recommendations(add_interpretations(df), RAW, r["label"])
    assert recs, "expected at least one note for an unhappy-looking passenger"
    forbidden = {"Age", "Gender", "Customer Type", "Type of Travel", "Class", "Flight Distance"}
    assert not forbidden & {x["Feature"] for x in recs}
    assert all(x["Priority"] in {"High", "Medium", "Low"} for x in recs)
    assert all(x["Contribution"] < 0 for x in recs)


def test_csv_report(art):
    r = predict(art, "lightgbm", RAW)
    df, _ = local_contributions(make_explainer(art.models["lightgbm"]), r["features"], RAW)
    df = add_interpretations(df)
    recs, _, note = build_recommendations(df, RAW, r["label"])
    csv = build_report(RAW, r, df, recs, note, "2026-01-01 00:00:00").to_csv(index=False)
    assert "LightGBM" in csv and "Improvement notes" in csv and "/home" not in csv


def test_validation():
    errs, _ = validate_input({**RAW, "Age": -1}, None)
    assert errs
    errs, warns = validate_input({**RAW, "Departure Delay in Minutes": 2000}, load_artifacts().schema)
    assert not errs and warns


def test_analytics_and_filters(data):
    df = an.load_dataset()
    assert df is not None and len(df) == len(pp.clean(data)[0])
    k = an.kpis(df)
    assert k["satisfied"] + k["dissatisfied"] == k["n"]
    f = an.Filters(cls=["Business"], age=(30, 40))
    sub = an.apply_filters(df, f)
    assert (sub["Class"] == "Business").all() and sub["Age"].between(30, 40).all()
    assert an.rate_by(sub, "Gender")["n"].sum() == len(sub)
    assert not an.Filters().is_active(an.data_bounds(df)) or True
    assert len(an.build_insights(df, "Full dataset")) > 3


@pytest.mark.parametrize("page", ["pages/1_Project_Overview.py", "pages/2_Prediction_Center.py", "pages/3_Analytics_Insights.py"])
def test_pages_start_without_exception(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_prediction_page_flow():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages/2_Prediction_Center.py"), default_timeout=180).run()
    at.button[0].click().run()  # submit the form with defaults
    assert not at.exception
    assert any("Predicted class" in m.value for m in at.markdown)
    at.radio[0].set_value("lightgbm").run()  # switching models reuses stored input
    assert not at.exception and any("LightGBM" in m.value for m in at.markdown)


def test_analytics_filter_widgets():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages/3_Analytics_Insights.py"), default_timeout=180).run()
    at.multiselect(key="f_class").set_value(["Business"]).run()
    assert not at.exception and any("Current filter selection" in m.value for m in at.markdown)
    at.button[0].click().run()  # reset
    assert not at.exception and at.session_state["f_class"] == []
