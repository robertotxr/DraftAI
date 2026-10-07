"""Tracking: Tackles Over Expected from NFL Big Data Bowl 2024 tracking data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.lib import KAGGLE_RULES, load_tracking, setup  # noqa: E402

setup("Tracking", ":material/timeline:")
st.title("Tackles Over Expected")
st.caption("An original tackling metric built on NFL Big Data Bowl 2024 player tracking (weeks 1-9).")

t = load_tracking()
if t is None:
    st.info(
        "The tracking data is not set up yet. Kaggle requires accepting the competition rules before the files can "
        f"be downloaded: [accept the rules here]({KAGGLE_RULES}). Then run these two commands in the repo:"
    )
    st.code("make bdb\nmake tracking", language="bash")
    st.markdown(
        "Tackles Over Expected compares each defender's tackles and assists with what a frame-level model expects "
        "given distance, closing speed, pursuit angle and blockers. The leaderboard, play animation and validation "
        "numbers appear here once the data is in."
    )
    st.stop()

lb, frames, val = t["leaderboard"], t["frames"], t["validation"]

st.subheader("Leaderboard")
top = int(lb["opportunities"].max())
min_opp = st.slider("Minimum tackle opportunities", 1, max(top, 2), min(20, top))
f = lb[lb["opportunities"] >= min_opp].sort_values("toe_per_100", ascending=False)
st.dataframe(
    f.rename(
        columns={
            "displayName": "Player",
            "position": "Pos",
            "opportunities": "Opportunities",
            "tackles": "Tackles",
            "missed": "Missed",
            "expected": "Expected",
            "toe": "TOE",
            "toe_per_100": "TOE / 100",
            "missed_rate": "Missed rate",
            "pursuit_efficiency": "Pursuit efficiency",
        }
    ).drop(columns="nflId"),
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "Expected": st.column_config.NumberColumn(format="%.1f"),
        "TOE": st.column_config.NumberColumn(format="%+.1f", help="Tackles + assists minus expected"),
        "TOE / 100": st.column_config.NumberColumn(format="%+.1f"),
        "Missed rate": st.column_config.NumberColumn(format="percent"),
        "Pursuit efficiency": st.column_config.NumberColumn(format="percent"),
    },
)
st.caption(f"{len(f)} players with at least {min_opp} opportunities. TOE = tackles and assists minus expected.")

st.subheader("Play viewer")
plays = frames[["gameId", "playId"]].drop_duplicates().sort_values(["gameId", "playId"]).itertuples(index=False)
plays = [(int(g), int(p)) for g, p in plays]
if plays:
    from src.tracking.animate import animate_play

    g, p = st.selectbox("Play", plays, format_func=lambda x: f"Game {x[0]}, play {x[1]}")
    st.plotly_chart(animate_play(frames, g, p), width="stretch")
    st.caption("Offense moves left to right. Defender marker size shows the model's tackle probability.")

st.subheader("Validation")
fm, stab, link = val.get("frame_model", {}), val.get("stability", {}), val.get("missed_tackle_link", {})


def num(x, fmt="{:.3f}"):
    return "n/a" if x is None else fmt.format(x)


v = st.columns(4)
v[0].metric("Frame-model Brier", num(fm.get("brier"), "{:.4f}"))
v[1].metric("Calibration error (ECE)", num(fm.get("ece"), "{:.4f}"))
v[2].metric(
    "Odd/even-week stability (r)",
    num(stab.get("split_half_r")),
    f"Spearman-Brown {num(stab.get('spearman_brown'))}",
    delta_color="off",
)
v[3].metric(
    "TOE vs missed-tackle rate (r)", num(link.get("pearson")), f"{link.get('n_players', 0)} players", delta_color="off"
)
st.caption(
    f"{val.get('n_frames', 0):,} defender-frames, {val.get('n_players', 0)} players. Stability asks whether a player's "
    "TOE per 100 in odd weeks predicts even weeks. A negative link to missed-tackle rate is the expected direction."
)
