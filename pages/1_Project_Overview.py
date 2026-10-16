import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import preprocessing as pp
from src.prediction import MODEL_NAMES, get_artifacts
from src.utils import (BLUE, DISSATISFIED_COLOR, MODEL_COLORS, card, hero, kpi_row, pct, safe_section, setup_page, style_fig)

setup_page("Project Overview")
art = get_artifacts()

hero(
    "Airline Passenger Satisfaction | Project Overview",
    "A machine-learning system that estimates whether an airline passenger is satisfied, using trip details, service ratings and delays, "
    "and explains each estimate in plain language.",
    "Built with XGBoost and LightGBM, SHAP explanations and an interactive analytics dashboard.",
)

if art.report is None:
    st.error("The training report is missing, so verified results cannot be shown. Run `python train_models.py --data data/train.csv` "
             "(see README) and reload this page.")
    st.stop()
rep, models, dq = art.report, art.report["models"], art.report["data_quality"]

# ------------------------------------------------------------------ business problem
st.header("Business problem")
c1, c2 = st.columns(2)
c1.markdown(card("Why satisfaction matters", "<p>Satisfied passengers are more likely to return and recommend an airline, "
             "while dissatisfied ones may switch carriers. Satisfaction is shaped by many factors at once (service, schedule, delays, "
             "trip type), which are hard to untangle by hand across hundreds of thousands of passenger records.</p>"), unsafe_allow_html=True)
c2.markdown(card("What a predictive model can add", "<p>A model can flag which kinds of passengers or trips look at risk of "
             "dissatisfaction and show which factors drive an individual estimate, helping teams prioritise where to investigate service and operations.</p>"),
            unsafe_allow_html=True)
with st.expander("Potential benefits vs. what this project actually demonstrates", expanded=True):
    st.markdown(
        f"""
**Demonstrated by this project (measured on held-out data):**
- Both models separate satisfied from neutral/dissatisfied passengers with about {pct(models['xgboost']['accuracy'], 0)} accuracy on {models['xgboost']['n_test']:,} passengers the models never saw during training.
- Local explanations show which inputs moved a particular prediction.

**Potential benefits (not tested here):**
- Reduced churn or higher revenue from acting on the findings.
- Real-time use in an airline's operations. The app is a portfolio demonstration; it is not connected to live airline data and the benefits above would need a pilot with the airline to confirm.
"""
    )

# ------------------------------------------------------------------ workflow
st.header("Our solution")
steps = [
    ("Data inspection", "Shape, types, missing values, duplicates"),
    ("Cleaning & preprocessing", "Remove duplicates/missing rows, encode categories"),
    ("Feature engineering", "Total delay, delay change, average service rating"),
    ("Exploratory analysis", "Distributions, segments, correlations"),
    ("Model training", "XGBoost and LightGBM"),
    ("Evaluation & comparison", "Held-out test set, shared metrics"),
    ("Interactive prediction", "Enter a passenger, pick a model"),
    ("Explanation & recommendations", "SHAP contributions, improvement notes"),
    ("Analytics & insights", "Filterable dashboard, automated findings"),
]
for row in (steps[:5], steps[5:]):
    cols = st.columns(5)
    for i, (col, (t, d)) in enumerate(zip(cols, row)):
        n = steps.index((t, d)) + 1
        col.markdown(f'<div class="step"><div class="n">STEP {n}</div><div class="t">{t}</div><div class="d">{d}</div></div>', unsafe_allow_html=True)
    st.write("")

# ------------------------------------------------------------------ dataset
st.header("Dataset overview")
clean = dq["cleaning"]
kpi_row([
    ("Rows (raw)", f"{dq['rows']:,}", "passenger records"),
    ("Columns (raw)", f"{dq['columns']}", "incl. id and target"),
    ("Missing values", f"{dq['missing_total']:,}", "all in Arrival Delay in Minutes" if list(dq["missing_by_column"]) == ["Arrival Delay in Minutes"] else "see table"),
    ("Duplicate rows", f"{dq['duplicate_rows']:,}", f"{dq['duplicate_rows_ignoring_id']:,} if `id` is ignored"),
    ("Rows after cleaning", f"{clean['rows_after']:,}", f"{clean['rows_before'] - clean['rows_after']:,} removed"),
])
st.write("")
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Columns and data types")
    cats = {
        "Passenger": ["Gender", "Age", "Customer Type"],
        "Travel": ["Type of Travel", "Class", "Flight Distance"],
        "Service ratings (0-5)": pp.SERVICE_COLS,
        "Flight delays": pp.DELAY_COLS,
        "Identifier (not used by models)": [pp.ID_COL],
        "Target": [pp.TARGET_COL],
    }
    rows = [(grp, c, dq["dtypes"].get(c, "?"), dq["missing_by_column"].get(c, 0)) for grp, cols in cats.items() for c in cols]
    st.dataframe(pd.DataFrame(rows, columns=["Category", "Column", "Data type", "Missing"]), hide_index=True, width="stretch", height=420)
with right:
    st.subheader("Data quality findings")
    t = dq["target_counts_raw"]
    zero_cols = {c: v for c, v in dq["rating_zero_counts"].items() if v}
    st.markdown(
        f"""
- **Target:** `satisfaction` is stored as True/False. The notebook's dictionary names the classes *satisfied* and *neutral or dissatisfied*, and satisfied passengers rate services higher, so **True = Satisfied**. Raw counts: {t['satisfied']:,} satisfied, {t['neutral_or_dissatisfied']:,} neutral or dissatisfied ({t['satisfied'] / dq['rows']:.1%} satisfied).
- **Missing values:** {dq['missing_total']:,} rows have no Arrival Delay; the notebook drops them ({clean['rows_with_missing_removed']:,} rows, {clean['rows_with_missing_removed'] / dq['rows']:.2%} of the data).
- **Duplicates:** {clean['duplicates_removed']:,} removed. {dq['duplicate_rows_ignoring_id']:,} rows are duplicated when ignoring `id`.
- **Zero ratings:** {len(zero_cols)} of the 13 service columns contain 0 (for example Inflight wifi: {dq['rating_zero_counts']['Inflight wifi service']:,}; Departure/Arrival time convenient: {dq['rating_zero_counts']['Departure/Arrival time convenient']:,}). The notebook documents a 0-5 scale and keeps zeros as given; this app does the same. 0 most plausibly means "not rated", but the notebook does not say.
- **Extreme values:** delays are heavily skewed (most flights have none). Valid observations are kept; analyses use log scales or grouped delay bands.
- **Identifier:** `id` is excluded from the models.
"""
    )

# ------------------------------------------------------------------ feature engineering
st.header("Feature engineering")
st.caption("All three features exist in the notebook and are reproduced in `src/preprocessing.py`; training and prediction share that code.")
f1, f2, f3 = st.columns(3)
f1.markdown(card("Total_Delay", "<p><b>Departure delay + arrival delay</b> (minutes).</p><p>Captures the overall delay burden in one number.</p>"), unsafe_allow_html=True)
f2.markdown(card("Delay_Diff", "<p><b>Arrival delay - departure delay</b> (minutes).</p><p>Positive: delay grew in flight. Negative: time was made up.</p>"), unsafe_allow_html=True)
f3.markdown(card("Average_Service_Rating", "<p><b>Mean of the 13 service ratings.</b></p><p>A summary of overall service experience for each passenger.</p>"), unsafe_allow_html=True)
st.info(
    "Methodology note: in the notebook the engineered features gave no measurable accuracy gain (both models stay at 0.92 before and after). "
    "They are kept because they belong to the validated pipeline and are useful for analysis, not because they were shown to improve accuracy."
)

# ------------------------------------------------------------------ models
st.header("Models used")
mc1, mc2 = st.columns(2)
for col, key, how, why, strengths, limits in [
    (mc1, "xgboost",
     "Builds many small decision trees one after another; each new tree focuses on the mistakes of the previous ones and the predictions of all trees are added together.",
     "Tabular data mixes categories, counts and ratings with non-linear interactions. Boosted trees handle this without feature scaling.",
     "Strong accuracy on tabular data; built-in regularisation; fast prediction; mature SHAP support.",
     "Many hyperparameters; less transparent than a single tree; was not tuned in this project."),
    (mc2, "lightgbm",
     "Also a boosted-tree ensemble, but it grows trees leaf-wise (always splitting the most promising leaf) and bins feature values into histograms for speed.",
     "Its histogram-based design makes training fast and memory-efficient on large tabular datasets like this one (hundreds of thousands of rows).",
     "Fast training; efficient on large data; accuracy comparable to XGBoost here.",
     "Leaf-wise growth can overfit small data; not tuned here. In the notebook `subsample=0.8` was passed without `subsample_freq`, so LightGBM ignores row subsampling (kept as-is for fidelity)."),
]:
    m = models[key]
    col.markdown(
        card(
            MODEL_NAMES[key],
            f"<p><b>How it works:</b> {how}</p><p><b>Why it suits this data:</b> {why}</p>"
            f"<p><b>Held-out results:</b> accuracy {m['accuracy']:.3f} | precision (Satisfied) {m['precision_satisfied']:.3f} | "
            f"recall {m['recall_satisfied']:.3f} | F1 {m['f1_satisfied']:.3f} | ROC-AUC {m['roc_auc']:.3f}</p>"
            f"<p><b>Strengths:</b> {strengths}</p><p><b>Limitations:</b> {limits}</p>",
        ),
        unsafe_allow_html=True,
    )
    if key not in art.models:
        col.warning(f"{MODEL_NAMES[key]} artifact could not be loaded: {art.errors.get(key, 'unknown error')}")

# ------------------------------------------------------------------ comparison
st.header("Model comparison")
sp = rep["split"]
st.caption(
    f"Both models were trained on the same {sp['n_train']:,} passengers and evaluated on the same {sp['n_test']:,} held-out passengers "
    f"(stratified {int((1 - sp['test_size']) * 100)}/{int(sp['test_size'] * 100)} split, random_state={sp['random_state']}), with identical preprocessing and the notebook's fixed hyperparameters. "
    "Nothing was tuned on the test set."
)
X, L = models["xgboost"], models["lightgbm"]
# metric, label, higher_is_better, tie tolerance (absolute for scores, relative for cost metrics)
spec = [
    ("accuracy", "Accuracy", True, 0.001), ("precision_satisfied", "Precision (Satisfied)", True, 0.001),
    ("recall_satisfied", "Recall (Satisfied)", True, 0.001), ("f1_satisfied", "F1-score (Satisfied)", True, 0.001),
    ("roc_auc", "ROC-AUC", True, 0.001), ("generalization_gap", "Train-test accuracy gap", False, 0.002),
    ("training_time_s", "Training time (s)", False, 0.10), ("inference_ms_per_1000_rows", "Inference (ms / 1,000 rows)", False, 0.10),
    ("single_row_latency_ms", "Single-row latency (ms)", False, 0.10), ("model_size_mb", "Model file size (MB)", False, 0.10),
]
rows = []
for k, label, hib, tol in spec:
    a, b = X[k], L[k]
    rel = abs(a - b) / max(abs(a), abs(b), 1e-12) if "time" in k or "ms" in k or "size" in k or "latency" in k else abs(a - b)
    if rel < tol:
        better = "Effectively tied"
    else:
        better = "XGBoost" if (a > b) == hib else "LightGBM"
    fmt = (lambda v: f"{v:.4f}") if hib or k == "generalization_gap" else (lambda v: f"{v:.3f}")
    rows.append((label, fmt(a), fmt(b), better))
cmp_df = pd.DataFrame(rows, columns=["Metric", "XGBoost", "LightGBM", "Better"])
st.dataframe(cmp_df, hide_index=True, width="stretch")
st.caption("'Effectively tied' means the difference is below 0.001 for quality metrics (0.002 for the gap) or below 10% for time and size. "
           "Timings come from a single machine run and will vary on other hardware.")

with safe_section("Metric chart"):
    ch1, ch2 = st.columns(2)
    names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    keys = ["accuracy", "precision_satisfied", "recall_satisfied", "f1_satisfied", "roc_auc"]
    fig = go.Figure()
    for key in ("xgboost", "lightgbm"):
        fig.add_bar(name=MODEL_NAMES[key], x=names, y=[models[key][k] for k in keys], marker_color=MODEL_COLORS[key],
                    text=[f"{models[key][k]:.3f}" for k in keys], textposition="outside")
    fig.update_yaxes(range=[0.85, 0.98], title="Score (axis starts at 0.85 to show small differences)")
    fig.update_layout(barmode="group", title="Held-out quality metrics (Satisfied = positive class)")
    ch1.plotly_chart(style_fig(fig), width="stretch")
    fig = go.Figure()
    for key in ("xgboost", "lightgbm"):
        fig.add_scatter(x=models[key]["roc_curve"]["fpr"], y=models[key]["roc_curve"]["tpr"], mode="lines", name=f"{MODEL_NAMES[key]} (AUC {models[key]['roc_auc']:.3f})",
                        line=dict(color=MODEL_COLORS[key], width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#999", dash="dash"))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    ch2.plotly_chart(style_fig(fig), width="stretch")

with safe_section("Confusion matrices"):
    cc = st.columns(2)
    for col, key in zip(cc, ("xgboost", "lightgbm")):
        cm = models[key]["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Pred: Neutral/Dissatisfied", "Pred: Satisfied"], y=["Actual: Neutral/Dissatisfied", "Actual: Satisfied"],
                                   text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"{MODEL_NAMES[key]} confusion matrix (n={models[key]['n_test']:,})")
        col.plotly_chart(style_fig(fig, 320), width="stretch")

with safe_section("Generalisation chart"):
    g = st.columns(2)
    fig = go.Figure()
    for key in ("xgboost", "lightgbm"):
        fig.add_bar(name=MODEL_NAMES[key], x=["Training accuracy", "Held-out test accuracy"], y=[models[key]["train_accuracy"], models[key]["accuracy"]],
                    marker_color=MODEL_COLORS[key], text=[f"{models[key]['train_accuracy']:.4f}", f"{models[key]['accuracy']:.4f}"], textposition="outside")
    fig.update_yaxes(range=[0.88, 0.98])
    fig.update_layout(barmode="group", title="Generalisation: training vs held-out accuracy")
    g[0].plotly_chart(style_fig(fig, 340), width="stretch")
    fig = go.Figure()
    for key in ("xgboost", "lightgbm"):
        fig.add_bar(name=MODEL_NAMES[key], x=["Training time (s)", "Inference (ms/1k rows)", "Model size (MB)"],
                    y=[models[key]["training_time_s"], models[key]["inference_ms_per_1000_rows"], models[key]["model_size_mb"]], marker_color=MODEL_COLORS[key])
    fig.update_layout(barmode="group", title="Cost metrics (lower is better)")
    g[1].plotly_chart(style_fig(fig, 340), width="stretch")

# Overall ranking with explicit criterion
f1x, f1l = X["f1_satisfied"], L["f1_satisfied"]
if abs(f1x - f1l) < 0.001:
    rank_text = (f"**Ranking criterion:** F1-score for the Satisfied class on the held-out set (tie if within 0.001, then ROC-AUC). "
                 f"F1 is {f1x:.4f} (XGBoost) vs {f1l:.4f} (LightGBM): **effectively tied**. ROC-AUC is {X['roc_auc']:.4f} vs {L['roc_auc']:.4f}, "
                 f"{'also within 0.001' if abs(X['roc_auc'] - L['roc_auc']) < 0.001 else 'a difference'}. Neither model can be declared the winner on quality; "
                 f"choose on secondary criteria such as inference speed, size and ecosystem.")
else:
    w = "XGBoost" if f1x > f1l else "LightGBM"
    rank_text = f"**Ranking criterion:** F1-score for the Satisfied class on the held-out set. **{w}** ranks first (F1 {f1x:.4f} vs {f1l:.4f})."
st.markdown(rank_text)

st.subheader("How to read the comparison")
cx, cy = st.columns(2)
cx.markdown(card("Accuracy alone is not enough", f"<p>Accuracy counts all correct predictions equally. "
            f"<b>Precision</b> asks: when the model says Satisfied, how often is it right? <b>Recall</b> asks: of all truly satisfied passengers, how many were found? "
            f"<b>F1</b> balances the two. Here the classes are fairly balanced ({pct(rep['split']['test_positive_rate'])} satisfied in the test set), "
            f"but errors still differ in kind.</p>"), unsafe_allow_html=True)
fn_x, fp_x = X["confusion_matrix"]["fn"], X["confusion_matrix"]["fp"]
cy.markdown(card("Which error costs more?", f"<p>A <b>false positive</b> predicts Satisfied for a passenger who is not (XGBoost: {fp_x:,}), so an at-risk passenger could be missed. "
            f"A <b>false negative</b> predicts not satisfied for a satisfied passenger (XGBoost: {fn_x:,}), which wastes attention. "
            f"If the goal is to catch unhappy passengers, false positives matter more; if retention outreach is expensive, false negatives matter more. "
            f"The decision threshold used here is the default 0.5.</p>"), unsafe_allow_html=True)
st.success(
    f"**In plain terms:** both models perform almost identically, correct on about {pct(X['accuracy'], 0)} of unseen passengers. "
    f"The training-to-test gap is small ({X['generalization_gap'] * 100:.1f} and {L['generalization_gap'] * 100:.1f} points), so neither shows heavy overfitting. "
    f"The agreement rate on the test set is {pct(rep['agreement_rate_on_test'], 2)}, so the model choice rarely changes the answer."
)
with st.expander("How these results relate to the notebook"):
    st.markdown(
        "The notebook reports 0.92 accuracy for both models (True-class precision 0.93 XGBoost / 0.92 LightGBM, recall 0.90, F1 0.91) on an unstratified 80/20 split. "
        "The figures above were **recalculated** here on a stratified split with the same preprocessing and hyperparameters, so they differ slightly in the third decimal. "
        "The notebook's final LightGBM was refit on 100% of the data for a Kaggle submission; that model has no held-out data, so it is not used for evaluation here. "
        "The notebook also trained logistic regression, KNN and naive Bayes baselines (0.82-0.86 accuracy on unscaled features); these are outside this app's scope."
    )
