"""Local explanations (SHAP) and rule-based, explanation-driven improvement notes.

Contributions are SHAP values from shap.TreeExplainer in the model's log-odds space for
the *Satisfied* class: positive = pushes the prediction toward Satisfied, negative =
pushes toward Neutral/Dissatisfied. They describe how the model arrived at this one
prediction; they are not causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src import preprocessing as pp

# Features a service owner can plausibly act on (service quality or operations).
ACTIONABLE = {
    "Inflight wifi service": "Review wifi coverage, speed and reliability, and how easy connecting is.",
    "Departure/Arrival time convenient": "Review whether schedules and departure/arrival slots suit the customers on this kind of trip.",
    "Ease of Online booking": "Investigate friction in the online booking flow (search, payment, seat selection).",
    "Gate location": "Look at gate assignment and walking distance/wayfinding at the airport.",
    "Food and drink": "Review meal quality, choice and service timing.",
    "Online boarding": "Investigate the online boarding experience (mobile boarding pass, app reliability, gate process).",
    "Seat comfort": "Review seat condition, pitch and cabin comfort.",
    "Inflight entertainment": "Review the entertainment catalogue, screen/system reliability and personal-device options.",
    "On-board service": "Review cabin-crew service levels, responsiveness and consistency.",
    "Leg room service": "Review legroom and seat-configuration options for this cabin.",
    "Baggage handling": "Look at baggage handling speed, damage and loss rates.",
    "Checkin service": "Investigate check-in waiting times and the counter, kiosk and online check-in experience.",
    "Cleanliness": "Review cabin and lavatory cleaning standards between flights.",
    "Departure Delay in Minutes": "Investigate departure delay management: root causes, passenger communication and compensation or care.",
    "Arrival Delay in Minutes": "Investigate arrival delay management and how affected passengers are informed and supported.",
    "Total_Delay": "Investigate end-to-end delay management (combined departure and arrival delay) and passenger communication.",
    "Delay_Diff": "Look at whether delays grow or shrink in flight (arrival delay minus departure delay) and at recovery procedures.",
    "Average_Service_Rating": "Overall service ratings are low or weighing on this prediction: identify which individual services drag the average down.",
}
CONTEXT_FEATURES = ["Age", "Flight Distance", "Gender", "Customer Type", "Type of Travel", "Class"]
DISPLAY = {"Total_Delay": "Total delay (min)", "Delay_Diff": "Delay change in flight (min)", "Average_Service_Rating": "Average service rating"}


def feature_label(name: str) -> str:
    return DISPLAY.get(name, name)


def make_explainer(model):
    return shap.TreeExplainer(model)


def _positive_class_values(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def local_contributions(explainer, X: pd.DataFrame, raw: dict) -> tuple[pd.DataFrame, float]:
    """Table of local SHAP contributions for a single-row X. Returns (table, base_value)."""
    sv = _positive_class_values(explainer.shap_values(X))[0]
    base = np.atleast_1d(explainer.expected_value)
    base = float(base[-1]) if base.size > 1 else float(base[0])
    rows = []
    for i, f in enumerate(X.columns):
        if f in raw:
            val = raw[f]
        elif f == "Total_Delay":
            val = float(X.iloc[0][f])
        else:
            val = float(X.iloc[0][f])
        rows.append(
            {
                "Feature": f,
                "Label": feature_label(f),
                "Your value": val,
                "Encoded value": float(X.iloc[0][f]),
                "Contribution": float(sv[i]),
                "Absolute contribution": float(abs(sv[i])),
                "Direction": "Toward Satisfied" if sv[i] > 0 else ("Toward Neutral/Dissatisfied" if sv[i] < 0 else "Neutral"),
            }
        )
    df = pd.DataFrame(rows).sort_values("Absolute contribution", ascending=False).reset_index(drop=True)
    return df, base


def _interpret(row) -> str:
    f, c = row["Feature"], row["Contribution"]
    direction = "raises" if c > 0 else "lowers"
    strength = "strongly" if row["Absolute contribution"] >= 0.5 else ("moderately" if row["Absolute contribution"] >= 0.15 else "slightly")
    return f"{feature_label(f)} of {_fmt(row['Your value'])} {strength} {direction} the model's estimate of this passenger being satisfied."


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)


def add_interpretations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Interpretation"] = out.apply(_interpret, axis=1)
    return out


def build_recommendations(contrib: pd.DataFrame, raw: dict, predicted_label: str, max_items: int = 5) -> tuple[list[dict], list[dict], str]:
    """Return (recommendations, strengths, note).

    Recommendations come only from actionable features whose local contribution pushes
    toward Neutral/Dissatisfied, ranked by magnitude. Priority = share of the total
    negative (toward-dissatisfied) contribution carried by that feature:
    >= 25% High, >= 10% Medium, otherwise Low. Immutable traits (age, gender, ...) are
    never the subject of a recommendation.
    """
    neg = contrib[contrib["Contribution"] < 0]
    neg_total = float(neg["Absolute contribution"].sum())
    recs = []
    for _, r in neg.iterrows():
        f = r["Feature"]
        if f not in ACTIONABLE or neg_total <= 0 or r["Absolute contribution"] < 0.02:
            continue
        share = r["Absolute contribution"] / neg_total
        priority = "High" if share >= 0.25 else ("Medium" if share >= 0.10 else "Low")
        val = r["Your value"]
        caveat = ""
        if f in pp.SERVICE_COLS and val >= 4:
            priority = "Low"
            caveat = " The rating itself is high, so this negative contribution likely reflects interaction with other inputs; treat as low-confidence."
        if f in pp.DELAY_COLS + ["Total_Delay", "Delay_Diff"] and val <= 0 and f != "Delay_Diff":
            priority = "Low"
            caveat = " The delay value is zero, so this likely reflects interaction with other inputs; treat as low-confidence."
        recs.append(
            {
                "Factor": feature_label(f),
                "Feature": f,
                "Your value": val,
                "Contribution": float(r["Contribution"]),
                "Direction": r["Direction"],
                "Priority": priority,
                "Why it matters": (
                    f"For this passenger the model reads {feature_label(f)} = {_fmt(val)} as pulling the estimate toward "
                    f"Neutral/Dissatisfied ({share:.0%} of all negative contributions)." + caveat
                ),
                "Recommendation": ACTIONABLE[f],
            }
        )
        if len(recs) >= max_items:
            break
    strengths = []
    pos = contrib[(contrib["Contribution"] > 0) & contrib["Feature"].isin(ACTIONABLE)].head(3)
    for _, r in pos.iterrows():
        strengths.append({"Factor": feature_label(r["Feature"]), "Your value": r["Your value"], "Contribution": float(r["Contribution"])})
    if recs:
        note = (
            "These notes rank operational factors by how much each one lowered this prediction. "
            "They indicate where the model sees risk for this input; they do not prove that improving a factor would change the outcome."
        )
    elif predicted_label == "Satisfied":
        note = "No actionable service or delay factor clearly lowers this prediction, so there is no clear improvement area to highlight for this input."
    else:
        note = (
            "The model's explanation does not isolate a clear, actionable factor for this prediction "
            "(the unfavourable contribution comes mainly from passenger/trip context or is spread thinly). "
            "No recommendation is made rather than guessing."
        )
    return recs, strengths, note


def build_report(raw: dict, pred: dict, contrib: pd.DataFrame, recs: list[dict], note: str, timestamp: str) -> pd.DataFrame:
    """One business-readable table (CSV-ready) covering input, result, contributions and notes."""
    rows = [
        ("Report", "Generated (UTC)", timestamp, "", "", "", ""),
        ("Report", "Model", pred["model"], "", "", "", ""),
        ("Result", "Predicted class", pred["label"], "", "", "", ""),
        ("Result", "Estimated probability: Satisfied", f"{pred['p_satisfied']:.1%}", "", "", "", "Model-estimated probability, not a guarantee"),
        ("Result", "Estimated probability: Neutral or Dissatisfied", f"{pred['p_dissatisfied']:.1%}", "", "", "", "Model-estimated probability, not a guarantee"),
    ]
    rows += [("Passenger input", k, v, "", "", "", "") for k, v in raw.items()]
    for _, r in contrib.head(10).iterrows():
        rows.append(("Top local contributions", r["Label"], r["Your value"], round(r["Contribution"], 4), r["Direction"], "", "Log-odds contribution toward the Satisfied class"))
    for r in recs:
        rows.append(("Improvement notes", r["Factor"], r["Your value"], round(r["Contribution"], 4), r["Direction"], r["Priority"], r["Recommendation"]))
    if not recs:
        rows.append(("Improvement notes", "No clear actionable factor", "", "", "", "", note))
    out = pd.DataFrame(rows, columns=["Section", "Item", "Value", "Contribution", "Direction", "Priority", "Note"])
    for c in ("Value", "Contribution"):
        out[c] = out[c].astype(str)
    return out
