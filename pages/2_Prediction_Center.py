import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src import preprocessing as pp
from src.explainability import add_interpretations, build_recommendations, build_report, local_contributions
from src.prediction import MODEL_NAMES, get_artifacts, get_explainer, predict, validate_input
from src.utils import DISSATISFIED_COLOR, SATISFIED_COLOR, hero, kpi_row, pct, safe_section, setup_page, style_fig

setup_page("Prediction Center")
art = get_artifacts()

hero("Passenger Satisfaction Prediction",
     "Enter a passenger's details, choose a model and get an estimate with a personal explanation of what drove it.")

# ------------------------------------------------------------------ artifact status
for key in MODEL_NAMES:
    if key not in art.models:
        st.warning(f"{MODEL_NAMES[key]} is unavailable: {art.errors.get(key, 'not loaded')}")
if art.schema is None:
    st.error(f"Preprocessing schema unavailable: {art.errors.get('schema', 'unknown problem')} Run `python train_models.py` (see README).")
    st.stop()
if not art.available:
    st.error("No trained model could be loaded, so predictions are not possible. Run `python train_models.py --data data/train.csv`.")
    st.stop()

# ------------------------------------------------------------------ A. model selection
st.header("1. Choose a model")
model_key = st.radio("Model", art.available, format_func=lambda k: MODEL_NAMES[k], horizontal=True,
                     help="Both models were trained on the same data with the same features; the choice rarely changes the result.")
if art.report:
    m = art.report["models"][model_key]
    kpi_row([
        ("Accuracy", f"{m['accuracy']:.3f}", "held-out test set"),
        ("Precision (Satisfied)", f"{m['precision_satisfied']:.3f}", ""),
        ("Recall (Satisfied)", f"{m['recall_satisfied']:.3f}", ""),
        ("F1 (Satisfied)", f"{m['f1_satisfied']:.3f}", ""),
        ("ROC-AUC", f"{m['roc_auc']:.3f}", ""),
    ])

# ------------------------------------------------------------------ B. form
st.header("2. Passenger details")
RATING_HELP = "1 = lowest, 5 = highest. 0 = not rated / not applicable (the dataset contains 0 values; their exact meaning is not documented)."
SERVICE_HELP = {
    "Inflight wifi service": "Quality of the wifi during the flight.",
    "Departure/Arrival time convenient": "How convenient the departure and arrival times were.",
    "Ease of Online booking": "How easy booking online was.",
    "Gate location": "How convenient the boarding gate location was.",
    "Food and drink": "Quality of food and drinks.",
    "Online boarding": "Experience of boarding online.",
    "Seat comfort": "Comfort of the seat.",
    "Inflight entertainment": "Quality of entertainment on board.",
    "On-board service": "Service provided by the cabin crew.",
    "Leg room service": "Space available for legs.",
    "Baggage handling": "How well baggage was handled.",
    "Checkin service": "Quality of check-in.",
    "Cleanliness": "Cleanliness of the aircraft.",
}
CUSTOMER_OPTS = {"Loyal customer": "Loyal Customer", "Disloyal customer": "disloyal Customer"}
TRAVEL_OPTS = {"Business travel": "Business travel", "Personal travel": "Personal Travel"}
rng = art.schema["feature_ranges"]

with st.form("passenger_form"):
    st.subheader("Passenger information")
    c = st.columns(3)
    gender = c[0].selectbox("Gender", ["Male", "Female"])
    age = c[1].number_input("Age", min_value=1, max_value=110, value=35, step=1, help=f"Training data covers ages {rng['Age'][0]:.0f}-{rng['Age'][1]:.0f}.")
    customer = c[2].selectbox("Customer type", list(CUSTOMER_OPTS), help="Loyal customers are repeat travellers with the airline.")

    st.subheader("Travel information")
    c = st.columns(3)
    travel = c[0].selectbox("Type of travel", list(TRAVEL_OPTS))
    cls = c[1].selectbox("Class", ["Eco", "Eco Plus", "Business"], index=2)
    dist = c[2].number_input("Flight distance", min_value=1, max_value=20000, value=1200, step=50,
                             help=f"Units are not documented in the dataset. Training data covers {rng['Flight Distance'][0]:.0f}-{rng['Flight Distance'][1]:.0f}.")

    st.subheader("Service ratings")
    st.caption(RATING_HELP)
    ratings = {}
    cols = st.columns(3)
    for i, name in enumerate(pp.SERVICE_COLS):
        ratings[name] = cols[i % 3].select_slider(name, options=[0, 1, 2, 3, 4, 5], value=3, help=SERVICE_HELP[name] + " " + RATING_HELP)

    st.subheader("Flight information")
    c = st.columns(2)
    dep = c[0].number_input("Departure delay (minutes)", min_value=0, max_value=3000, value=0, step=5)
    arr = c[1].number_input("Arrival delay (minutes)", min_value=0, max_value=3000, value=0, step=5)
    submitted = st.form_submit_button("Predict Satisfaction", type="primary", width="stretch")

if submitted:
    raw = {"Age": int(age), "Flight Distance": int(dist), **ratings, "Departure Delay in Minutes": int(dep),
           "Arrival Delay in Minutes": int(arr), "Gender": gender, "Customer Type": CUSTOMER_OPTS[customer],
           "Type of Travel": TRAVEL_OPTS[travel], "Class": cls}
    st.session_state["raw_input"] = raw
    st.session_state["predicted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

raw = st.session_state.get("raw_input")
if raw is None:
    st.info("Fill in the form and press **Predict Satisfaction** to see the result.")
    st.stop()

errors, warns = validate_input(raw, art.schema)
if errors:
    for e in errors:
        st.error(e)
    st.stop()
for w in warns:
    st.warning(w)

# ------------------------------------------------------------------ C. result
try:
    pred = predict(art, model_key, raw)
except Exception as exc:  # noqa: BLE001
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.header("3. Prediction result")
sat = pred["class_id"] == 1
r1, r2 = st.columns([1.2, 1])
r1.markdown(
    f'<div class="result {"sat" if sat else "dis"}"><div class="small">Predicted class ({pred["model"]})</div>'
    f'<div class="big">{"Satisfied" if sat else "Neutral or Dissatisfied"}</div>'
    f'<div class="small">Model-estimated probability of this class: {pred["confidence"]:.1%}</div></div>',
    unsafe_allow_html=True,
)
with r2:
    fig = go.Figure(go.Bar(x=[pred["p_satisfied"], pred["p_dissatisfied"]], y=["Satisfied", "Neutral or Dissatisfied"], orientation="h",
                           marker_color=[SATISFIED_COLOR, DISSATISFIED_COLOR], text=[f"{pred['p_satisfied']:.1%}", f"{pred['p_dissatisfied']:.1%}"], textposition="auto"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title="Estimated class probabilities", showlegend=False)
    st.plotly_chart(style_fig(fig, 200), width="stretch")
st.caption("Probabilities are model estimates. Their calibration has not been verified (see the Brier score in the training report), and a high value is not a guarantee that the prediction is correct.")
if sat:
    st.markdown("**Interpretation:** based on these inputs the model expects this passenger to be satisfied. The explanation below shows what supported that and what, if anything, worked against it.")
else:
    st.markdown("**Interpretation:** based on these inputs the model expects this passenger to be neutral or dissatisfied. The notes below highlight the operational factors that weighed most.")

# ------------------------------------------------------------------ explanation
contrib = recs = None
try:
    explainer = get_explainer(model_key)
    contrib, base = local_contributions(explainer, pred["features"], raw)
    contrib = add_interpretations(contrib)
    recs, strengths, note = build_recommendations(contrib, raw, pred["label"])
except Exception as exc:  # noqa: BLE001
    st.warning(f"The explanation is unavailable for this prediction ({type(exc).__name__}). The prediction above is still valid.")

if contrib is not None:
    st.header("4. Personalised improvement notes")
    st.caption("Ranked by each factor's contribution to THIS prediction (local SHAP values), not by global importance. "
               "Age, gender and other fixed traits are never recommended for change.")
    with safe_section("Improvement notes"):
        if recs:
            for r in recs:
                st.markdown(
                    f'<div class="rec {r["Priority"]}"><div class="head">{r["Factor"]}<span class="badge {r["Priority"]}">{r["Priority"]} priority</span></div>'
                    f'<div class="meta">Your value: <b>{r["Your value"]:g}</b> &nbsp;|&nbsp; Contribution: <b>{r["Contribution"]:+.2f}</b> ({r["Direction"]})</div>'
                    f'<div style="margin-top:.35rem"><b>Why it matters:</b> {r["Why it matters"]}</div>'
                    f'<div><b>Suggested action:</b> {r["Recommendation"]}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)
        if strengths:
            st.markdown("**Factors supporting a satisfied outcome:** " + ", ".join(f"{s['Factor']} ({s['Your value']:g})" for s in strengths))
        st.caption("Contributions describe how the model reached this estimate; they do not prove cause and effect, and improving a factor is not guaranteed to change the outcome.")

    st.header("5. Local feature contribution analysis")
    with safe_section("Contribution chart"):
        top = contrib.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["Contribution"], y=top["Label"], orientation="h",
                               marker_color=[SATISFIED_COLOR if v > 0 else DISSATISFIED_COLOR for v in top["Contribution"]],
                               hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.update_layout(title=f"Top 10 factors for this passenger ({pred['model']})", xaxis_title="Contribution to log-odds of Satisfied")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
        st.markdown("**How to read this:** blue bars push the prediction toward **Satisfied**, red bars toward **Neutral/Dissatisfied**. "
                    "Longer bars mean a larger influence on this prediction. Values are SHAP contributions in log-odds; "
                    "they sum with the model's base value to give the model's raw output. This is a *local* explanation for one passenger; "
                    "global importance (Analytics page) describes the model as a whole.")
        show = contrib[["Label", "Your value", "Encoded value", "Contribution", "Direction", "Absolute contribution", "Interpretation"]].rename(columns={"Label": "Feature"})
        show["Your value"] = show["Your value"].map(lambda v: f"{v:g}" if isinstance(v, (int, float)) else str(v))
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"Contribution": st.column_config.NumberColumn(format="%+.3f"),
                                    "Absolute contribution": st.column_config.NumberColumn(format="%.3f"),
                                    "Encoded value": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Encoded value is what the model receives (e.g. Gender: Male=0, Female=1; Class: Eco=0, Eco Plus=1, Business=2; Customer Type: Loyal=0, disloyal=1; Type of Travel: Business=0, Personal=1).")

# ------------------------------------------------------------------ F. compare
st.header("6. Compare both models")
if len(art.available) < 2:
    st.info("Comparison needs both models to be available.")
elif st.checkbox("Run this passenger through both models"):
    with safe_section("Model comparison"):
        px_, pl_ = predict(art, "xgboost", raw), predict(art, "lightgbm", raw)
        c1, c2, c3 = st.columns(3)
        for col, p in ((c1, px_), (c2, pl_)):
            col.metric(p["model"], p["label"], f"P(Satisfied) = {p['p_satisfied']:.1%}", delta_color="off")
        agree = px_["class_id"] == pl_["class_id"]
        c3.metric("Agreement", "Agree" if agree else "Disagree", f"Probability gap: {abs(px_['p_satisfied'] - pl_['p_satisfied']) * 100:.1f} points", delta_color="off")
        st.caption("Disagreement signals uncertainty or sensitivity to model choice; it does not tell you which model is right. Agreement does not guarantee correctness either.")

# ------------------------------------------------------------------ G. download
st.header("7. Download report")
if contrib is not None:
    report = build_report(raw, pred, contrib, recs, note, st.session_state["predicted_at"])
    st.download_button("Download prediction report (CSV)", report.to_csv(index=False).encode("utf-8"),
                       file_name="passenger_satisfaction_report.csv", mime="text/csv")
    with st.expander("Preview report"):
        st.dataframe(report, hide_index=True, width="stretch")
else:
    st.info("A report with explanations is available only when the explanation could be computed.")
