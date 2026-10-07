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
KAGGLE_RULES = "https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules"
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


def setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} | DraftAI", page_icon=icon, layout="wide")


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
    """Prospects joined with predictions: one row per drafted player."""
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
