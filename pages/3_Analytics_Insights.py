import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics as an
from src import preprocessing as pp
from src.prediction import MODEL_NAMES, get_artifacts
from src.utils import BLUE, DISSATISFIED_COLOR, SATISFIED_COLOR, hero, kpi_row, safe_section, setup_page, style_fig

setup_page("Analytics & Insights")
hero("Data Analytics and Business Insights",
     "Explore what is associated with passenger satisfaction across ratings, segments and delays. All figures are computed from the dataset.")

df_all = an.load_dataset()
if df_all is None:
    st.error("The dataset was not found. Place `train.csv` (or `train.zip`) in the `data/` folder, see `data/README.md`, then reload.")
    st.stop()
art = get_artifacts()
bounds = an.data_bounds(df_all)

# ------------------------------------------------------------------ I. filters (sidebar)
DEFAULTS = {"f_sat": [an.SAT, an.DIS], "f_gender": [], "f_cust": [], "f_travel": [], "f_class": [],
            "f_age": bounds["age"], "f_dist": bounds["distance"], "f_dep": bounds["dep"], "f_arr": bounds["arr"]}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.subheader("Filters")
    st.caption("Leave a list empty to include all values. Filters apply to every chart except where stated.")
    st.multiselect("Satisfaction class", [an.SAT, an.DIS], key="f_sat")
    st.multiselect("Gender", ["Male", "Female"], key="f_gender")
    st.multiselect("Customer type", sorted(df_all["Customer Type"].cat.categories), key="f_cust")
    st.multiselect("Type of travel", sorted(df_all["Type of Travel"].cat.categories), key="f_travel")
    st.multiselect("Class", ["Eco", "Eco Plus", "Business"], key="f_class")
    st.slider("Age range", *bounds["age"], key="f_age")
    st.slider("Flight distance", *bounds["distance"], key="f_dist")
    st.slider("Departure delay (min)", *bounds["dep"], key="f_dep")
    st.slider("Arrival delay (min)", *bounds["arr"], key="f_arr")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

S = st.session_state
flt = an.Filters(satisfaction=S.f_sat or [an.SAT, an.DIS], gender=S.f_gender, customer=S.f_cust, travel=S.f_travel, cls=S.f_class,
                 age=S.f_age, distance=S.f_dist, dep_delay=S.f_dep, arr_delay=S.f_arr)
active = flt.is_active(bounds)
df = an.apply_filters(df_all, flt)

# ------------------------------------------------------------------ A. executive summary
st.header("Executive summary")
full = an.kpis(df_all)
st.markdown("**Full dataset** (cleaned; never affected by filters)")
kpi_row([
    ("Passenger records", f"{full['n']:,}", ""),
    ("Satisfied", f"{full['satisfied']:,}", f"{full['rate']:.1%} of passengers"),
    ("Neutral / dissatisfied", f"{full['dissatisfied']:,}", f"{1 - full['rate']:.1%}"),
    ("Avg. age", f"{full['age']:.1f}", "years"),
    ("Avg. flight distance", f"{full['distance']:,.0f}", "units not documented"),
])
st.write("")
kpi_row([
    ("Avg. departure delay", f"{full['dep']:.1f} min", "mean, heavily skewed"),
    ("Avg. arrival delay", f"{full['arr']:.1f} min", "mean, heavily skewed"),
    ("Departures delayed >15 min", f"{full['delayed_share']:.1%}", ""),
    ("Data quality", "No missing values", "after removing rows lacking Arrival Delay (see Overview)"),
    ("Duplicates", "None found", "full-row duplicates"),
])
if active:
    st.markdown(f"**Current filter selection** ({len(df):,} passengers, {len(df) / full['n']:.1%} of the data)")
    if len(df) == 0:
        st.warning("No passengers match the current filters. Adjust or reset them.")
        st.stop()
    k = an.kpis(df)
    kpi_row([
        ("Passengers", f"{k['n']:,}", ""),
        ("Satisfied", f"{k['satisfied']:,}", f"{k['rate']:.1%}"),
        ("Avg. age", f"{k['age']:.1f}", ""),
        ("Avg. departure delay", f"{k['dep']:.1f} min", ""),
        ("Avg. arrival delay", f"{k['arr']:.1f} min", ""),
    ])
    st.info("Filters are active: charts below describe the filtered passengers only (except the global feature importance tab). "
            "Note that filtering on satisfaction class changes satisfaction rates by design.")
else:
    st.caption("No filters active: everything below describes the full dataset.")

scope = "Filtered selection" if active else "Full dataset"
tabs = st.tabs(["Target", "Service ratings", "Segments", "Delays", "Correlations", "Engineered features", "Feature importance", "Automated insights"])

# ------------------------------------------------------------------ B. target
with tabs[0]:
    with safe_section("Target analysis"):
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_distribution(df), width="stretch")
        c2.plotly_chart(an.fig_counts(df), width="stretch")
        k = an.kpis(df)
        ratio = max(k["rate"], 1 - k["rate"]) / max(min(k["rate"], 1 - k["rate"]), 1e-9)
        st.markdown(
            f"{scope}: **{k['rate']:.1%}** satisfied vs **{1 - k['rate']:.1%}** neutral or dissatisfied (majority:minority ratio {ratio:.2f}:1). "
            "The classes are moderately balanced, so accuracy is not badly distorted by a dominant class, but precision, recall and F1 should still be read alongside it "
            "because the two kinds of error carry different business costs."
        )

# ------------------------------------------------------------------ C. service ratings
with tabs[1]:
    with safe_section("Service rating analysis"):
        st.plotly_chart(an.fig_rating_heatmap(df), width="stretch")
        st.caption("Each cell = satisfied passengers / all passengers who gave that rating for that service. Denominators differ by cell; see the detail chart for counts. 0 appears in the data and may mean 'not rated'.")
        sel = st.selectbox("Service to inspect", pp.SERVICE_COLS, index=5)
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_rating_detail(df, sel), width="stretch")
        c2.plotly_chart(an.fig_rate_bar(df, sel, f"Satisfied rate by {sel} rating", order=list(range(6))), width="stretch")
        mean_tbl = df.groupby("Satisfaction", observed=True)[pp.SERVICE_COLS].mean().T.round(2)
        mean_tbl["Difference"] = (mean_tbl.get(an.SAT, np.nan) - mean_tbl.get(an.DIS, np.nan)).round(2)
        st.subheader("Average rating by satisfaction class")
        st.dataframe(mean_tbl.sort_values("Difference", ascending=False), width="stretch")
        st.caption("Services with a larger difference separate the two groups more in this data. This is an association, not evidence that improving the service causes satisfaction.")

# ------------------------------------------------------------------ D. segments
with tabs[2]:
    with safe_section("Segmentation"):
        st.caption("Age groups: 17 and under, 18-24, 25-34, 35-44, 45-54, 55-64, 65+. Distance groups use fixed bands (the dataset does not document distance units).")
        seg = [("Gender", ["Male", "Female"]), ("Age group", an.AGE_LABELS), ("Customer Type", None), ("Type of Travel", None),
               ("Class", ["Eco", "Eco Plus", "Business"]), ("Distance group", an.DIST_LABELS)]
        for i in range(0, len(seg), 2):
            cols = st.columns(2)
            for col, (name, order) in zip(cols, seg[i:i + 2]):
                col.plotly_chart(an.fig_rate_bar(df, name, order=order), width="stretch")
        st.markdown("Bars show the share of satisfied passengers in each group; labels include group size (n). Small groups give noisier rates. "
                    "Differences between segments describe who is in each segment, not why they are satisfied.")

# ------------------------------------------------------------------ E. delays
with tabs[3]:
    with safe_section("Delay analysis"):
        st.caption("Delays are heavily skewed: most flights have none and a few exceed several hundred minutes. No observations were removed; "
                   "histograms use a log y-axis and satisfaction is shown by delay band.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_hist_by_class(df, "Departure Delay in Minutes", 60, True), width="stretch")
        c2.plotly_chart(an.fig_hist_by_class(df, "Arrival Delay in Minutes", 60, True), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_rate_bar(df, "Departure delay group", order=an.DELAY_LABELS), width="stretch")
        c2.plotly_chart(an.fig_rate_bar(df, "Arrival delay group", order=an.DELAY_LABELS), width="stretch")
        st.plotly_chart(an.fig_scatter(df, "Departure Delay in Minutes", "Arrival Delay in Minutes"), width="stretch")
        st.subheader("Delays and other features")
        c1, c2 = st.columns(2)
        for col, grp in ((c1, "Class"), (c2, "Type of Travel")):
            g = df.groupby(grp, observed=True)[pp.DELAY_COLS].mean().reset_index().melt(id_vars=grp, var_name="Delay", value_name="Mean minutes")
            fig = px.bar(g, x=grp, y="Mean minutes", color="Delay", barmode="group", title=f"Mean delay by {grp}",
                         color_discrete_sequence=[BLUE, "#6C9BD2"])
            col.plotly_chart(style_fig(fig, 340), width="stretch")
        corr_d = df[pp.SERVICE_COLS + ["Departure Delay in Minutes"]].astype("float32").corr()["Departure Delay in Minutes"].drop("Departure Delay in Minutes").sort_values()
        fig = px.bar(x=corr_d.values, y=corr_d.index, orientation="h", title="Correlation of departure delay with each service rating",
                     labels={"x": "Pearson r", "y": ""}, color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig, 420), width="stretch")

# ------------------------------------------------------------------ F. correlation
with tabs[4]:
    with safe_section("Correlation analysis"):
        fig, corr = an.fig_corr(df)
        st.plotly_chart(fig, width="stretch")
        st.warning("Encoded categorical variables (e.g. Class: Eco=0, Eco Plus=1, Business=2; Gender: Male=0, Female=1) are arbitrary numeric codes, "
                   "so Pearson correlations involving them have no straightforward interpretation. Correlation is not causation.")
        st.plotly_chart(an.fig_target_corr(corr), width="stretch")
        st.subheader("Explore a relationship")
        num_cols = ["Age", "Flight Distance", *pp.DELAY_COLS, *pp.ENGINEERED_COLS, *pp.SERVICE_COLS]
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", num_cols, index=1)
        y = c2.selectbox("Y axis", num_cols, index=4)
        st.plotly_chart(an.fig_scatter(df, x, y), width="stretch")
        dist_col = st.selectbox("Distribution by satisfaction", ["Age", "Flight Distance", "Average_Service_Rating"], key="dist_sel")
        st.plotly_chart(an.fig_hist_by_class(df, dist_col, 40), width="stretch")

# ------------------------------------------------------------------ G. engineered
with tabs[5]:
    with safe_section("Engineered feature analysis"):
        chk_total = float((df["Total_Delay"] - (df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"])).abs().max())
        chk_diff = float((df["Delay_Diff"] - (df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"])).abs().max())
        chk_avg = float((df["Average_Service_Rating"] - df[pp.SERVICE_COLS].mean(axis=1)).abs().max())
        st.success(f"Verification on {len(df):,} rows: max absolute difference between stored and recomputed values - Total_Delay {chk_total:g}, Delay_Diff {chk_diff:g}, Average_Service_Rating {chk_avg:.2g}.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_hist_by_class(df, "Average_Service_Rating", 25), width="stretch")
        c2.plotly_chart(an.fig_rate_bar(df, "Average rating group", "Satisfied rate by average service rating", order=an.AVG_LABELS), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(an.fig_hist_by_class(df, "Total_Delay", 60, True), width="stretch")
        within = df["Delay_Diff"].between(-60, 60)
        sub = df[within]
        fig = an.fig_hist_by_class(sub, "Delay_Diff", 60, True, "Delay_Diff (arrival minus departure delay), shown for -60 to +60 min")
        c2.plotly_chart(fig, width="stretch")
        c2.caption(f"{(~within).sum():,} passengers ({(~within).mean():.2%}) have Delay_Diff outside the plotted range; they are still included in all other analyses.")
        st.markdown("Average_Service_Rating is the engineered feature most clearly related to satisfaction here; the two delay features are mostly near zero for most passengers (association only).")

# ------------------------------------------------------------------ H. importance
with tabs[6]:
    with safe_section("Feature importance"):
        if art.report is None:
            st.info("Importance data is unavailable because the training report is missing.")
        else:
            st.caption("This tab is based on the trained models and the held-out test sample, so it is not affected by the filters.")
            c1, c2 = st.columns(2)
            mk = c1.radio("Model", [k for k in MODEL_NAMES if k in art.report["importance"]], format_func=lambda k: MODEL_NAMES[k], horizontal=True)
            methods = {"shap_mean_abs": "Mean |SHAP| (held-out sample)", "native_gain": "Native gain importance (training)", "permutation_auc_drop": "Permutation importance (held-out ROC-AUC drop)"}
            method = c2.selectbox("Method", list(methods), format_func=lambda m: methods[m])
            st.plotly_chart(an.fig_importance(art.report, mk, method), width="stretch")
            st.markdown(
                """
- **Global importance** (this chart) summarises how much a feature matters to the model across many passengers.
- **Local contribution** (Prediction Center) is how much a feature moved one specific passenger's prediction.
- **Statistical association** means two things vary together in the data.
- **Causal effect** is what would change if a factor were intervened on. None of the above establishes it; importance reflects how the model uses features, including correlated ones that can share credit.
"""
            )

# ------------------------------------------------------------------ J. insights
with tabs[7]:
    with safe_section("Automated insights"):
        st.caption(f"Computed from: **{scope}** ({len(df):,} passengers). Groups with fewer than {an.MIN_GROUP} passengers are ignored.")
        for line in an.build_insights(df, scope):
            st.markdown(f"- {line}")
        st.caption("These are observed associations in the data, not causal claims. Use them to decide where to investigate, not as proof of what to change.")
