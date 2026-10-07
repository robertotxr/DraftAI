"""Tracking: Closing Over Expected from NFL Big Data Bowl 2026 player tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import ATHLETIC, BDB26_DATA, MODEL, MUTED, load_closing, setup  # noqa: E402

setup("Tracking", ":material/timeline:")
st.title("Closing Over Expected")
st.caption(
    "An original coverage metric built on NFL Big Data Bowl 2026 player tracking (2023 pass plays, ball in the air)."
)
st.markdown(
    "When the quarterback lets go of the ball, every throw gives the defender a different job: how far he is from the "
    "receiver, where the ball is going, how long it will hang, and how fast he is already moving. Closing Over "
    "Expected asks how close the defender gets to the receiver by the time the ball arrives, compared with what a "
    "typical defender in the same spot would have done. A positive number means he got closer than expected, in "
    "yards. It only uses what is known at the moment of the throw, so it measures the defender's burst and "
    "ball-in-air play, not the quality of the throw."
)

t = load_closing()
if t is None:
    st.info(
        f"The Closing Over Expected artifacts are not built yet. Download the [BDB 2026 Analytics data]({BDB26_DATA}) "
        "(requires accepting the competition rules on Kaggle), then run:"
    )
    st.code("make bdb && make tracking", language="bash")
    st.stop()

lb, frames, plays, val, link = t["leaderboard"], t["frames"], t["plays"], t["validation"], t["draft_link"]


def num(x, fmt="{:.3f}"):
    return "n/a" if x is None else fmt.format(x)


st.subheader("Leaderboard")
c1, c2 = st.columns([2, 1])
pos = c1.multiselect("Position", sorted(lb["position"].unique()), default=[], placeholder="All positions")
top = int(lb["plays"].max())
min_plays = c2.slider(
    "Minimum plays", int(lb["plays"].min()), max(top, int(lb["plays"].min()) + 1), int(lb["plays"].min())
)
f = lb[(lb["plays"] >= min_plays) & (lb["position"].isin(pos) if pos else True)]
st.dataframe(
    f.rename(
        columns={
            "player_name": "Player", "position": "Pos", "team": "Team", "plays": "Plays",
            "mean_coe": "COE / play", "total_coe": "Total COE", "coe_se": "Std. error",
            "mean_expected": "Expected end dist", "mean_actual": "Actual end dist", "mean_closing": "Closing",
        }
    ).drop(columns="nfl_id"),
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "COE / play": st.column_config.NumberColumn(format="%+.2f", help="Expected minus actual end distance (yards)"),
        "Total COE": st.column_config.NumberColumn(format="%+.0f"),
        "Std. error": st.column_config.NumberColumn(format="%.2f"),
        "Expected end dist": st.column_config.NumberColumn(format="%.1f"),
        "Actual end dist": st.column_config.NumberColumn(format="%.1f"),
        "Closing": st.column_config.NumberColumn(format="%.1f", help="Start distance minus end distance (yards)"),
    },
)  # fmt: skip
st.caption(
    f"{len(f)} defenders with at least {min_plays} tracked plays. COE is in yards per play; the standard error shows "
    "how much a single season's sample can move it."
)

st.subheader("Play viewer")
if len(plays):
    from src.tracking.closing_animate import animate_closing

    desc = {
        (int(g), int(p)): d for g, p, d in plays[["game_id", "play_id", "play_description"]].itertuples(index=False)
    }
    g, p = st.selectbox("Play", list(desc), format_func=lambda k: str(desc[k])[:110])
    st.plotly_chart(animate_closing(frames, g, p), width="stretch")
    st.caption(
        "Offense moves left to right. Dotted lines join each tracked defender to the targeted receiver; the "
        "star is where the ball lands. Time 0 is the throw. Hover a tracked defender for his COE on the play."
    )

st.subheader("Validation")
m, stab, out = val["model"], val["stability"], val["outcome"]
v = st.columns(4)
v[0].metric(
    "Expected-distance error (RMSE)",
    f"{m['lightgbm']['rmse']:.2f} yd",
    f"naive {m['end_equals_start']['rmse']:.2f}",
    delta_color="off",
)
v[1].metric(
    "vs. linear on geometry",
    f"{m['linear_geometry']['rmse']:.2f} yd",
    f"{val['n_defender_plays']:,} defender-plays",
    delta_color="off",
)
v[2].metric(
    "Split-half stability (r)", num(stab["split_half_r"], "{:.2f}"),
    f"Spearman-Brown {num(stab['spearman_brown'], '{:.2f}')}, {stab['n_players']} players", delta_color="off",
)  # fmt: skip
lg = out["logit_completion"]["coe"]
v[3].metric(
    "Completion odds per +1 yd COE", f"x{2.718281828 ** lg['coef']:.2f}", f"{out['n_plays']:,} plays", delta_color="off"
)
q = out["by_quintile"]
fig = go.Figure(go.Bar(x=[f"Q{r['quintile']}" for r in q], y=[r["completion_rate"] for r in q], marker_color=MODEL))
fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", title="Completion rate by COE quintile (nearest defender)",
                  yaxis=dict(tickformat=".0%", range=[0.5, 0.8]))  # fmt: skip
st.plotly_chart(fig, width="stretch")
st.caption(
    f"The model is out-of-fold by game, so a game never trains its own expectation. Stability compares a player's mean "
    f"COE in odd and even weeks (players with at least {stab['min_plays_per_half']} plays in each); a modest correlation "
    "means one season is a noisy read on any single defender. Outcome relevance uses the defender closest to the "
    "receiver at the throw and controls for the starting and expected distances. Q5 is the group that closed the "
    f"most above expectation: completion {out['q5_minus_q1_completion'] * 100:+.0f} points and EPA "
    f"{out['q5_minus_q1_epa']:+.2f} per play versus Q1."
)

st.subheader("Link to the draft")
if link is None or "draft_link" not in val or "correlations" not in val["draft_link"]:
    st.info("The draft warehouse was not available when the metric was built, so the link to prospects is missing.")
else:
    d = val["draft_link"]
    st.markdown(
        f"{d['n_matched']} of {d['n_players']} tracked defenders ({d['match_rate'] * 100:.0f}%) were matched to a "
        "drafted or combine prospect by name and birth date (or name and position group when unique). "
        "This is exploratory: it asks whether pre-draft athleticism or draft slot foreshadow ball-in-air closing."
    )
    labels = {"speed_score": "Speed score", "athletic_score": "Athletic score", "agility_score": "Agility score",
              "pick": "Draft pick", "p_starter": "Model P(starter)"}  # fmt: skip
    col = st.selectbox("Pre-draft feature", list(labels), format_func=labels.get)
    s = link[(link["plays"] >= d["correlations"]["min_plays"])].dropna(subset=[col])
    fig = go.Figure(go.Scatter(x=s[col], y=s["mean_coe"], mode="markers", text=s["player_name"] + " (" + s["position"] + ")",
                               hoverinfo="text", marker=dict(color=ATHLETIC, size=8, line=dict(color="white", width=1))))  # fmt: skip
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", xaxis_title=labels[col],
                      yaxis_title="Mean COE (yards per play)")  # fmt: skip
    st.plotly_chart(fig, width="stretch")
    c = d["correlations"]["all"][col]
    st.caption(
        f"Spearman {num(c['spearman'], '{:+.2f}')} (n = {c['n']}) for players with at least "
        f"{d['correlations']['min_plays']} plays. Small samples per player and a single season: read this as "
        "'no clear relationship', not as proof either way."
    )
