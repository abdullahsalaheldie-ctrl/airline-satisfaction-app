"""Shared UI helpers: theme, CSS, KPI cards and safe rendering."""
from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

NAVY = "#0B1F3A"
BLUE = "#1D4E89"
LIGHT_BLUE = "#6C9BD2"
GRAY = "#F3F5F8"
MID_GRAY = "#C9D1DC"
TEXT_MUTED = "#5B6B82"
SATISFIED_COLOR = "#1D4E89"      # satisfied = navy blue everywhere
DISSATISFIED_COLOR = "#D1495B"   # neutral/dissatisfied = coral everywhere
MODEL_COLORS = {"xgboost": "#1D4E89", "lightgbm": "#6C9BD2"}
CLASS_COLORS = {"Satisfied": SATISFIED_COLOR, "Neutral or Dissatisfied": DISSATISFIED_COLOR}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(135deg, {NAVY} 0%, #17376a 100%); color: #fff; padding: 2rem 2.2rem;
        border-radius: 14px; margin-bottom: 1.2rem; }}
.hero h1 {{ color: #fff; margin: 0 0 .4rem 0; font-size: 2rem; }}
.hero p {{ color: #DCE6F5; margin: .2rem 0; font-size: 1.02rem; }}
.kpi {{ background: {GRAY}; border-left: 4px solid {BLUE}; border-radius: 10px; padding: .85rem 1rem; height: 100%; }}
.kpi .label {{ color: {TEXT_MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .value {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.25; }}
.kpi .sub {{ color: {TEXT_MUTED}; font-size: .78rem; }}
.card {{ background: #fff; border: 1px solid {MID_GRAY}; border-radius: 12px; padding: 1rem 1.15rem; height: 100%; }}
.card h4 {{ margin: 0 0 .35rem 0; color: {NAVY}; }}
.card p, .card li {{ color: #26364f; font-size: .93rem; }}
.step {{ background: {GRAY}; border-radius: 12px; padding: .8rem 1rem; height: 100%; border-top: 3px solid {BLUE}; }}
.step .n {{ color: {BLUE}; font-weight: 700; font-size: .8rem; }}
.step .t {{ color: {NAVY}; font-weight: 600; font-size: .95rem; }}
.step .d {{ color: {TEXT_MUTED}; font-size: .8rem; }}
.result {{ border-radius: 14px; padding: 1.3rem 1.5rem; color: #fff; }}
.result.sat {{ background: linear-gradient(135deg, {BLUE}, #2a6cb5); }}
.result.dis {{ background: linear-gradient(135deg, #a8323f, {DISSATISFIED_COLOR}); }}
.result .big {{ font-size: 1.9rem; font-weight: 700; }}
.result .small {{ opacity: .92; font-size: .95rem; }}
.rec {{ border: 1px solid {MID_GRAY}; border-left-width: 6px; border-radius: 10px; padding: .8rem 1rem; margin-bottom: .7rem; background: #fff; }}
.rec.High {{ border-left-color: {DISSATISFIED_COLOR}; }}
.rec.Medium {{ border-left-color: #E0A030; }}
.rec.Low {{ border-left-color: {MID_GRAY}; }}
.rec .head {{ font-weight: 700; color: {NAVY}; }}
.badge {{ display:inline-block; padding: .05rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 600; margin-left: .4rem; }}
.badge.High {{ background:#fbe3e6; color:#9b2433; }} .badge.Medium {{ background:#fcefd2; color:#8a5a00; }} .badge.Low {{ background:#e8ecf2; color:#4b5a70; }}
.rec .meta {{ color: {TEXT_MUTED}; font-size: .82rem; }}
.note {{ background:{GRAY}; border-radius:10px; padding:.7rem 1rem; color:#26364f; font-size:.9rem; }}
section[data-testid="stSidebar"] {{ background: {GRAY}; }}
</style>
"""


def setup_page(title: str, icon: str = ":material/flight:") -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, extra: str = "") -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><p>{extra}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(kpi(label, value, sub), unsafe_allow_html=True)


def card(title: str, body_html: str) -> str:
    return f'<div class="card"><h4>{title}</h4>{body_html}</div>'


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY),
        title_font=dict(size=15, color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


@contextmanager
def safe_section(name: str):
    """A failing optional chart/analysis shows a warning instead of crashing the page."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        st.warning(f"{name} could not be displayed ({type(exc).__name__}: {exc}).")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception_only(type(exc), exc)))


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
