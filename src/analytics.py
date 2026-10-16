"""Dataset loading, filtering, aggregation, charts and automated insights for page 3."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import preprocessing as pp
from src.utils import (
    BLUE,
    CLASS_COLORS,
    DISSATISFIED_COLOR,
    MID_GRAY,
    MODEL_COLORS,
    NAVY,
    SATISFIED_COLOR,
    style_fig,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [ROOT / "data" / "train.csv", ROOT / "data" / "train.zip"]
SAT, DIS = "Satisfied", "Neutral or Dissatisfied"
MIN_GROUP = 200  # minimum group size for automated insights

AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["17 and under", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
DIST_BINS = [-1, 499, 999, 1999, 2999, 1e9]
DIST_LABELS = ["Under 500", "500-999", "1,000-1,999", "2,000-2,999", "3,000+"]
DELAY_BINS = [-1, 0, 15, 60, 120, 1e9]
DELAY_LABELS = ["No delay", "1-15 min", "16-60 min", "61-120 min", "Over 120 min"]
AVG_BINS = [-0.01, 1.5, 2.5, 3.5, 4.5, 5.01]
AVG_LABELS = ["0-1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5", "4.5-5"]


def find_data_file() -> Path | None:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> pd.DataFrame | None:
    """Cleaned dataset (notebook cleaning), with readable labels and derived groups."""
    path = find_data_file()
    if path is None:
        return None
    raw = pp.load_raw(path)
    df, _ = pp.clean(raw)
    df["sat"] = pp.encode_target(df[pp.TARGET_COL]).astype("int8")
    df["Satisfaction"] = np.where(df["sat"] == 1, SAT, DIS)
    df = pp.add_engineered_features(df)
    df["Age group"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    df["Distance group"] = pd.cut(df["Flight Distance"], DIST_BINS, labels=DIST_LABELS)
    df["Departure delay group"] = pd.cut(df["Departure Delay in Minutes"], DELAY_BINS, labels=DELAY_LABELS)
    df["Arrival delay group"] = pd.cut(df["Arrival Delay in Minutes"], DELAY_BINS, labels=DELAY_LABELS)
    df["Average rating group"] = pd.cut(df["Average_Service_Rating"], AVG_BINS, labels=AVG_LABELS)
    for c in ["Gender", "Customer Type", "Type of Travel", "Class", "Satisfaction"]:
        df[c] = df[c].astype("category")
    for c in pp.SERVICE_COLS + ["Age", "Flight Distance", "Departure Delay in Minutes"]:
        df[c] = df[c].astype("int32")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].astype("float32")
    return df.drop(columns=[pp.ID_COL, pp.TARGET_COL], errors="ignore")


# ---------------------------------------------------------------- filters
@dataclass
class Filters:
    satisfaction: list = field(default_factory=lambda: [SAT, DIS])
    gender: list = field(default_factory=list)
    customer: list = field(default_factory=list)
    travel: list = field(default_factory=list)
    cls: list = field(default_factory=list)
    age: tuple = (0, 200)
    distance: tuple = (0, 10**9)
    dep_delay: tuple = (0, 10**9)
    arr_delay: tuple = (0, 10**9)

    def is_active(self, bounds: dict) -> bool:
        return (
            len(self.satisfaction) != 2 or bool(self.gender or self.customer or self.travel or self.cls)
            or tuple(self.age) != bounds["age"] or tuple(self.distance) != bounds["distance"]
            or tuple(self.dep_delay) != bounds["dep"] or tuple(self.arr_delay) != bounds["arr"]
        )


def data_bounds(df: pd.DataFrame) -> dict:
    return {
        "age": (int(df["Age"].min()), int(df["Age"].max())),
        "distance": (int(df["Flight Distance"].min()), int(df["Flight Distance"].max())),
        "dep": (0, int(df["Departure Delay in Minutes"].max())),
        "arr": (0, int(np.ceil(df["Arrival Delay in Minutes"].max()))),
    }


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = df["Satisfaction"].isin(f.satisfaction)
    if f.gender:
        m &= df["Gender"].isin(f.gender)
    if f.customer:
        m &= df["Customer Type"].isin(f.customer)
    if f.travel:
        m &= df["Type of Travel"].isin(f.travel)
    if f.cls:
        m &= df["Class"].isin(f.cls)
    m &= df["Age"].between(*f.age)
    m &= df["Flight Distance"].between(*f.distance)
    m &= df["Departure Delay in Minutes"].between(*f.dep_delay)
    m &= df["Arrival Delay in Minutes"].between(*f.arr_delay)
    return df[m]


# ---------------------------------------------------------------- aggregates
def kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    sat = int(df["sat"].sum())
    return {
        "n": n,
        "satisfied": sat,
        "dissatisfied": n - sat,
        "rate": sat / n,
        "age": float(df["Age"].mean()),
        "distance": float(df["Flight Distance"].mean()),
        "dep": float(df["Departure Delay in Minutes"].mean()),
        "arr": float(df["Arrival Delay in Minutes"].mean()),
        "delayed_share": float((df["Departure Delay in Minutes"] > 15).mean()),
    }


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["sat"].agg(["size", "sum"]).reset_index()
    g.columns = [col, "n", "satisfied"]
    g["rate"] = g["satisfied"] / g["n"]
    return g


# ---------------------------------------------------------------- charts
def fig_distribution(df: pd.DataFrame):
    c = df["Satisfaction"].value_counts().reindex([SAT, DIS]).fillna(0)
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.62, marker_colors=[CLASS_COLORS[i] for i in c.index],
                           textinfo="percent+label", sort=False))
    fig.update_layout(title="Satisfaction distribution", showlegend=False)
    return style_fig(fig, 340)


def fig_counts(df: pd.DataFrame):
    c = df["Satisfaction"].value_counts().reindex([SAT, DIS]).fillna(0).reset_index()
    c.columns = ["Satisfaction", "Passengers"]
    c["Share"] = c["Passengers"] / max(c["Passengers"].sum(), 1)
    fig = px.bar(c, x="Satisfaction", y="Passengers", color="Satisfaction", color_discrete_map=CLASS_COLORS,
                 text=c.apply(lambda r: f"{int(r.Passengers):,} ({r.Share:.1%})", axis=1), title="Passenger counts by class")
    fig.update_layout(showlegend=False)
    return style_fig(fig, 340)


def fig_rating_heatmap(df: pd.DataFrame):
    rows = {}
    for c in pp.SERVICE_COLS:
        g = rate_by(df, c).set_index(c)
        rows[c] = g["rate"].reindex(range(6))
    z = pd.DataFrame(rows).T
    fig = px.imshow(z, x=[str(i) for i in range(6)], y=z.index, color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto",
                    text_auto=".0%", labels=dict(x="Rating given", y="", color="Satisfied rate"),
                    title="Satisfied share of passengers at each rating (blank = no passengers)")
    return style_fig(fig, 520)


def fig_rating_detail(df: pd.DataFrame, col: str):
    g = df.groupby([col, "Satisfaction"], observed=True).size().reset_index(name="n")
    g["share"] = g["n"] / g.groupby(col)["n"].transform("sum")
    fig = px.bar(g, x=col, y="n", color="Satisfaction", barmode="group", color_discrete_map=CLASS_COLORS,
                 category_orders={"Satisfaction": [SAT, DIS]}, title=f"{col}: passengers by rating and satisfaction (counts)",
                 hover_data={"share": ":.1%"})
    fig.update_xaxes(type="category")
    return style_fig(fig)


def fig_rate_bar(df: pd.DataFrame, col: str, title: str | None = None, order: list | None = None):
    g = rate_by(df, col)
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    overall = df["sat"].mean() if len(df) else np.nan
    fig = go.Figure(go.Bar(x=g[col].astype(str), y=g["rate"], marker_color=BLUE,
                           text=[f"{r:.0%}<br>n={n:,}" for r, n in zip(g["rate"], g["n"])], textposition="outside",
                           hovertemplate="%{x}<br>Satisfied rate %{y:.1%}<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=DISSATISFIED_COLOR,
                  annotation_text=f"Overall {overall:.0%}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="Share satisfied")
    fig.update_layout(title=title or f"Satisfied rate by {col}")
    return style_fig(fig, 360)


def fig_hist_by_class(df: pd.DataFrame, col: str, nbins: int = 50, log_y: bool = False, title: str | None = None):
    fig = px.histogram(df, x=col, color="Satisfaction", nbins=nbins, barmode="overlay", opacity=0.65, histnorm="percent",
                       color_discrete_map=CLASS_COLORS, category_orders={"Satisfaction": [SAT, DIS]},
                       title=title or f"{col}: distribution by satisfaction (each class = 100%)")
    if log_y:
        fig.update_yaxes(type="log", title="Percent of class (log scale)")
    else:
        fig.update_yaxes(title="Percent of class")
    return style_fig(fig)


def fig_scatter(df: pd.DataFrame, x: str, y: str, n: int = 4000):
    s = df.sample(min(n, len(df)), random_state=1)
    fig = px.scatter(s, x=x, y=y, color="Satisfaction", opacity=0.45, color_discrete_map=CLASS_COLORS,
                     category_orders={"Satisfaction": [SAT, DIS]}, title=f"{y} vs {x} (random sample of {len(s):,} passengers)")
    return style_fig(fig, 420)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = df[["Age", "Flight Distance", *pp.SERVICE_COLS, *pp.DELAY_COLS, *pp.ENGINEERED_COLS]].astype("float32").copy()
    for col, mp in pp.CATEGORICAL_MAPPINGS.items():
        num[f"{col} (encoded)"] = df[col].astype(str).map(mp).astype("float32")
    num["Satisfied (1/0)"] = df["sat"].astype("float32")
    return num


def fig_corr(df: pd.DataFrame):
    corr = numeric_frame(df).corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Pearson correlation matrix")
    return style_fig(fig, 700), corr


def fig_target_corr(corr: pd.DataFrame):
    s = corr["Satisfied (1/0)"].drop("Satisfied (1/0)").sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[SATISFIED_COLOR if v > 0 else DISSATISFIED_COLOR for v in s.values]))
    fig.update_layout(title="Correlation of each feature with being satisfied", xaxis_title="Pearson r")
    return style_fig(fig, 640)


def fig_importance(report: dict, model_key: str, method: str, top: int = 15):
    imp = report["importance"][model_key]
    s = pd.Series(imp[method], index=imp["features"]).sort_values().tail(top)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=MODEL_COLORS[model_key],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="Share of total importance")
    return style_fig(fig, 520)


# ---------------------------------------------------------------- insights
def build_insights(df: pd.DataFrame, scope: str) -> list[str]:
    """Plain-language observations computed from `df`. Only groups with >= MIN_GROUP rows are used."""
    out: list[str] = []
    if len(df) < MIN_GROUP * 2:
        return [f"Too few passengers in the current selection ({len(df):,}) for reliable automated insights."]
    overall = df["sat"].mean()
    out.append(f"{scope}: {overall:.1%} of {len(df):,} passengers are satisfied.")

    gaps = []
    for c in pp.SERVICE_COLS:
        hi, lo = df[df[c] >= 4], df[(df[c] <= 2) & (df[c] >= 1)]
        if len(hi) >= MIN_GROUP and len(lo) >= MIN_GROUP:
            gaps.append((hi["sat"].mean() - lo["sat"].mean(), c, hi["sat"].mean(), lo["sat"].mean()))
    gaps.sort(reverse=True)
    for gap, c, h, l in gaps[:3]:
        out.append(f"Passengers rating '{c}' 4-5 are satisfied {h:.0%} of the time versus {l:.0%} for ratings 1-2 (gap {gap * 100:.0f} points). This is an association, not proof of cause.")
    if gaps:
        gap, c, h, l = gaps[-1]
        out.append(f"'{c}' shows the smallest gap between high and low ratings ({gap * 100:.0f} points), so it separates satisfied from unsatisfied passengers least in this selection.")

    for col, label in [("Class", "travel class"), ("Type of Travel", "travel purpose"), ("Customer Type", "customer type")]:
        g = rate_by(df, col)
        g = g[g["n"] >= MIN_GROUP]
        if len(g) >= 2:
            hi, lo = g.loc[g["rate"].idxmax()], g.loc[g["rate"].idxmin()]
            out.append(f"By {label}: {hi[col]} passengers are satisfied {hi['rate']:.0%} of the time vs {lo['rate']:.0%} for {lo[col]} ({hi['n']:,} vs {lo['n']:,} passengers).")

    g = rate_by(df, "Departure delay group").set_index("Departure delay group")
    if {"No delay", "Over 120 min"} <= set(g.index) and g.loc["No delay", "n"] >= MIN_GROUP and g.loc["Over 120 min", "n"] >= MIN_GROUP:
        out.append(f"Passengers with no departure delay are satisfied {g.loc['No delay', 'rate']:.0%} of the time vs {g.loc['Over 120 min', 'rate']:.0%} with delays over 120 minutes. Delay differences may also reflect who tends to fly delayed routes.")
    elif {"No delay", "61-120 min"} <= set(g.index) and g.loc["61-120 min", "n"] >= MIN_GROUP:
        out.append(f"Passengers with no departure delay are satisfied {g.loc['No delay', 'rate']:.0%} of the time vs {g.loc['61-120 min', 'rate']:.0%} with 61-120 minute delays.")

    a = rate_by(df, "Age group")
    a = a[a["n"] >= MIN_GROUP]
    if len(a) >= 2:
        hi, lo = a.loc[a["rate"].idxmax()], a.loc[a["rate"].idxmin()]
        out.append(f"Among age groups, {hi['Age group']} has the highest satisfied rate ({hi['rate']:.0%}) and {lo['Age group']} the lowest ({lo['rate']:.0%}).")
    return out
