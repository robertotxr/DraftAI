"""Shared loaders, labels and formatting for the DraftAI Streamlit app (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import path  # noqa: E402
from src.db import Warehouse  # noqa: E402

MODEL, MARKET, ATHLETIC, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
KAGGLE_DATA = "https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data"
BDB26_DATA = "https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data"
POS_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]

MODEL_LABELS = {
    "pick_only": "Draft slot only",
    "athletic_only": "Athleticism only",
    "scouting_single": "Scouting (no slot), single",
    "scouting_per_group": "Scouting (no slot), per position",
    "full_single": "Full model, single",
    "full_per_group": "Full model, per position",
    "blend": "Slot + model blend",
}

FEATURE_LABELS = {
    "log_pick": "Draft slot",
    "pos_group": "Position group",
    "ht_in": "Height",
    "wt_lb": "Weight",
    "ht_in_score": "Height (position percentile)",
    "wt_lb_score": "Weight (position percentile)",
    "forty_size_score": "40-yard dash (size-adjusted)",
    "vertical_size_score": "Vertical jump (size-adjusted)",
    "broad_jump_size_score": "Broad jump (size-adjusted)",
    "cone_size_score": "3-cone drill (size-adjusted)",
    "shuttle_size_score": "Shuttle (size-adjusted)",
    "bench_size_score": "Bench press (size-adjusted)",
    "athletic_score": "Overall athletic score",
    "speed_score": "Speed score",
    "explosion_score": "Explosion score",
    "agility_score": "Agility score",
    "drills_measured": "Number of drills tested",
    "attended_combine": "Attended combine",
    "draft_age": "Age on draft day",
    "power_conf": "Power-conference school",
    "college_seasons": "College seasons played",
    "final_season_age": "Age in final college season",
    "has_def_stats": "Has defensive college stats",
    "breakout_age_rec": "Receiving breakout age",
    "breakout_age_scrim": "Scrimmage breakout age",
    "breakout_age_def": "Defensive breakout age",
    "never_broke_out_rec": "Never broke out (receiving)",
    "never_broke_out_scrim": "Never broke out (scrimmage)",
    "never_broke_out_def": "Never broke out (defense)",
}
_METRIC_LABELS = {
    "dominator_adj": "Dominator rating (opp.-adj.)",
    "scrimmage_share_adj": "Scrimmage yards share (opp.-adj.)",
    "rush_yds_share_adj": "Rushing yards share (opp.-adj.)",
    "rec_yds_pg_adj": "Receiving yds/game (opp.-adj.)",
    "rush_yds_pg_adj": "Rushing yds/game (opp.-adj.)",
    "pass_yds_pg_adj": "Passing yds/game (opp.-adj.)",
    "yds_per_rec": "Yards per reception",
    "yds_per_carry": "Yards per carry",
    "target_share_proxy": "Pass-play usage share (target proxy)",
    "pass_ypa": "Passing yards per attempt",
    "pass_td_rate": "Pass TD rate",
    "pass_int_rate": "Interception rate",
    "pass_cmp_pct": "Completion %",
    "def_playmaking_share_adj": "Defensive playmaking share (opp.-adj.)",
    "pressure_pg_adj": "Pressures/game (opp.-adj.)",
    "sack_share": "Sack share",
    "pd_share": "Passes-defensed share",
    "tackle_share": "Tackle share",
    "int_pg": "Interceptions/game",
    "sos_z": "Strength of schedule (z)",
}
_AGG_LABELS = {"final": "final season", "best": "best season", "career": "career avg"}
for _m, _l in _METRIC_LABELS.items():
    for _a, _al in _AGG_LABELS.items():
        FEATURE_LABELS[f"{_m}_{_a}"] = f"{_l}, {_al}"


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


# Production metrics worth showing per position group (the same fields feed the model and the comps).
PRODUCTION_BY_GROUP = {
    "QB": ["pass_ypa", "pass_td_rate", "pass_int_rate", "pass_cmp_pct", "pass_yds_pg_adj", "rush_yds_pg_adj"],
    "RB": ["scrimmage_share_adj", "rush_yds_pg_adj", "yds_per_carry", "rec_yds_pg_adj", "target_share_proxy"],
    "WR": ["dominator_adj", "rec_yds_pg_adj", "yds_per_rec", "target_share_proxy", "scrimmage_share_adj"],
    "TE": ["dominator_adj", "rec_yds_pg_adj", "yds_per_rec", "target_share_proxy"],
    "OL": [],
    "DL": ["def_playmaking_share_adj", "pressure_pg_adj", "sack_share", "tackle_share"],
    "LB": ["def_playmaking_share_adj", "pressure_pg_adj", "tackle_share", "pd_share", "sack_share"],
    "DB": ["pd_share", "int_pg", "tackle_share", "def_playmaking_share_adj"],
}
PERCENT_METRICS = {"pass_td_rate", "pass_int_rate", "pass_cmp_pct", "dominator_adj", "scrimmage_share_adj",
                   "rush_yds_share_adj", "target_share_proxy", "def_playmaking_share_adj", "sack_share",
                   "pd_share", "tackle_share"}  # fmt: skip


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def pick_label(pick) -> str:
    return "UDFA" if pd.isna(pick) else str(int(pick))


def draft_text(row: pd.Series) -> str:
    if row["undrafted"] == 1:
        return "Undrafted (combine invitee)"
    return f"Round {int(row['round'])}, pick {int(row['pick'])} ({row['team']})"


INK, SUB, LINE, BG, SURFACE = "#18181b", "#71717a", "#e4e4e0", "#f8f8f7", "#f1f1ef"
FONT = "Geist, system-ui, sans-serif"
CSS = f"""
<style>
:root {{--ink: {INK}; --sub: {SUB}; --line: {LINE}; --accent: {MODEL}; --r-card: 12px; --r-input: 8px;}}
header[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu, footer {{display: none !important;}}
.block-container {{padding-top: 2.4rem; padding-bottom: 3.5rem; max-width: 1180px;}}
h1, h2, h3 {{color: var(--ink); letter-spacing: -0.02em;}}
h1 {{font-weight: 650 !important; font-size: 2.2rem !important;}}
h2, h3 {{font-weight: 600 !important;}}
a {{color: var(--accent);}}
div[data-testid="stVerticalBlockBorderWrapper"] {{border-radius: var(--r-card); border-color: var(--line);
  background: {SURFACE};}}
div[data-testid="stExpander"] details {{border-radius: var(--r-card); border-color: var(--line); background: transparent;}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{border-radius: var(--r-input);}}
div[data-testid="stDataFrame"] {{border-radius: var(--r-card); overflow: hidden; border: 1px solid var(--line);}}
div[data-testid="stAlert"] {{border-radius: var(--r-card);}}
.stButton button, a[data-testid="stPageLink-NavLink"] {{border-radius: 999px;}}
div[data-testid="stHorizontalBlock"] {{flex-wrap: wrap; row-gap: 0.75rem;}}
div[data-testid="stColumn"] {{min-width: 180px;}}
@media (max-width: 999px) {{div[data-testid="stColumn"] {{min-width: 280px;}}}}
@media (min-width: 1000px) {{div[data-testid="stHorizontalBlock"] {{flex-wrap: nowrap;}} div[data-testid="stColumn"] {{min-width: 0;}}}}
.sub {{color: var(--sub); font-size: 1.05rem; margin: -0.3rem 0 1.4rem; max-width: 44rem;}}
.hero h1 {{font-size: 2.5rem !important; line-height: 1.08; margin: 0; letter-spacing: -0.03em; padding: 0;}}
.hero h1 em {{font-style: normal; color: var(--accent);}}
.hero p {{font-size: 1.12rem; color: var(--sub); margin: 0.9rem 0 1.2rem; max-width: 26rem;}}
.stat {{border-top: 1px solid var(--line); padding: 12px 4px 4px 0;}}
.stat .v {{font-size: 1.9rem; font-weight: 650; letter-spacing: -0.03em; line-height: 1.1; color: var(--ink);}}
.stat.big .v {{font-size: 3.2rem; color: var(--accent);}}
.stat .l {{font-size: 0.9rem; font-weight: 500; color: var(--ink); margin-top: 4px;}}
.stat .s {{font-size: 0.82rem; color: var(--sub); margin-top: 2px;}}
.chip {{display: inline-block; padding: 3px 11px; border-radius: 999px; background: #e9f0fb; color: #1f5fb0;
  font-size: 0.8rem; font-weight: 550; margin: 0 6px 6px 0;}}
.pname {{font-size: 2.3rem; font-weight: 650; letter-spacing: -0.03em; color: var(--ink); line-height: 1.1;
  margin-bottom: 8px;}}
.eyebrow {{font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 10px;}}
.nav a {{text-decoration: none;}}
</style>
"""


def setup(title: str, icon: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"{title} | DraftAI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if subtitle:
        st.title(title)
        st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(value: str, label: str, sub: str = "", big: bool = False) -> None:
    """Plain stat block: ink number, hairline on top. big = the one headline stat, in the accent."""
    st.markdown(
        f'<div class="stat{" big" if big else ""}"><div class="v">{value}</div><div class="l">{label}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 300, **layout):
    """Consistent clean Plotly look: transparent, light grid, Geist."""
    layout.setdefault("margin", dict(l=10, r=10, t=30, b=10))
    layout.setdefault("title_font", dict(size=14, color=INK))
    fig.update_layout(height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=INK, size=13), hoverlabel=dict(font_family=FONT),
                      **layout)  # fmt: skip
    fig.update_xaxes(gridcolor="#ebebe8", zeroline=False, linecolor=LINE, tickfont=dict(color=SUB))
    fig.update_yaxes(gridcolor="#ebebe8", zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=SUB))
    return fig


@st.cache_data(show_spinner=False)
def sleeper_stats() -> dict | None:
    """Rounds 4-7, out-of-sample classes: starter rate for the model's top 10% of the class vs the rest."""
    b = load_board()
    o = b[(b["prediction_type"] != "in_sample") & (b["undrafted"] == 0) & (b["round"] >= 4) & b["starter"].notna()]
    if o.empty:
        return None
    top = o.groupby("draft_year")["p_starter"].rank(pct=True) >= 0.9
    if top.sum() == 0 or (~top).sum() == 0:
        return None
    return {"top": o.loc[top, "starter"].mean(), "rest": o.loc[~top, "starter"].mean(), "n_top": int(top.sum())}


def delta_text(v) -> str:
    """Value over slot in points with an arrow."""
    return "" if v is None or pd.isna(v) else f"{'▲' if v >= 0 else '▼'} {abs(v) * 100:.1f}"


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    return json.loads((path("artifacts") / "report_2021_2023.json").read_text())


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((path("artifacts") / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def load_shap_global() -> pd.DataFrame:
    return pd.read_csv(path("artifacts") / "shap_global.csv")


@st.cache_data(show_spinner="Loading draft data...")
def load_table(name: str) -> pd.DataFrame:
    wh = Warehouse(read_only=True)
    try:
        return wh.table(f"marts.{name}")
    finally:
        wh.close()


@st.cache_data(show_spinner=False)
def load_board() -> pd.DataFrame:
    """Prospects joined with predictions: one row per drafted player or undrafted combine invitee."""
    pros, pred = load_table("prospects"), load_table("predictions")
    keep = ["player_key", "p_starter", "p_pick_only", "q10", "q50", "q90", "prediction_type", "confidence",
            "value_over_slot"]  # fmt: skip
    df = pros.merge(pred[keep], on="player_key", how="inner")
    df["pick"] = df["pick"].astype("Int64")
    return df.sort_values(["draft_year", "pick"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_comps() -> pd.DataFrame:
    return load_table("comps")


@st.cache_data(show_spinner=False)
def load_shap_row(player_key: str) -> pd.Series:
    sv = load_table("shap_values")
    return sv[sv["player_key"] == player_key].drop(columns=["player_key", "pos_group"], errors="ignore").iloc[0]


def tracking_dir() -> Path:
    return path("artifacts") / "tracking"


@st.cache_data(show_spinner=False)
def load_tracking() -> dict | None:
    """Tracking artifacts, or None while any of them is missing."""
    d = tracking_dir()
    files = ["animation_frames.parquet", "leaderboard.parquet", "validation.json"]
    if not all((d / f).exists() for f in files):
        return None
    return {
        "frames": pd.read_parquet(d / files[0]),
        "leaderboard": pd.read_parquet(d / files[1]),
        "validation": json.loads((d / files[2]).read_text()),
    }


def closing_dir() -> Path:
    return path("artifacts") / "tracking_closing"


@st.cache_data(show_spinner=False)
def load_closing() -> dict | None:
    """Closing Over Expected artifacts, or None while the required ones are missing (draft link is optional)."""
    d = closing_dir()
    need = ["leaderboard.parquet", "animation_frames.parquet", "animation_plays.parquet", "validation.json"]
    if not all((d / f).exists() for f in need):
        return None
    link = d / "draft_link.parquet"
    return {
        "leaderboard": pd.read_parquet(d / need[0]),
        "frames": pd.read_parquet(d / need[1]),
        "plays": pd.read_parquet(d / need[2]),
        "validation": json.loads((d / need[3]).read_text()),
        "draft_link": pd.read_parquet(link) if link.exists() else None,
    }


def select_player(player_key: str) -> None:
    st.session_state["player_key"] = player_key


def outcome_text(row: pd.Series) -> str:
    """Plain-words NFL outcome for a prospect row; only observed seasons are reported."""
    ys = [row.get(f"snap_share_y{k}") for k in (1, 2, 3)]
    seen = [y for y in ys if pd.notna(y)]
    if not seen:
        return "Not yet in the NFL data"
    if pd.notna(row.get("snap_share_3yr")):
        return f"{pct(row['snap_share_3yr'])} avg snap share, {'starter' if row['starter'] == 1 else 'not a starter'}"
    return f"{len(seen)} of 3 seasons observed, avg {pct(float(np.mean(seen)))} snap share so far"
