"""Tracking: Closing Over Expected from NFL Big Data Bowl 2026 player tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import ATHLETIC, BDB26_DATA, MODEL, MUTED, kpi, load_closing, pct, setup, style_fig  # noqa: E402

setup(
    "Closing Over Expected",
    ":material/timeline:",
    "How much closer a defender gets to the receiver than a typical defender would, from the throw to the catch point "
    "(yards, 2023 pass plays, Big Data Bowl 2026).",
)

t = load_closing()
if t is None:
    st.info(
        f"Tracking artifacts are not built yet. Download the [BDB 2026 Analytics data]({BDB26_DATA}) "
        "(accept the rules on Kaggle), then run:"
    )
    st.code("make bdb && make tracking", language="bash")
    st.stop()

lb, frames, plays, val, link = t["leaderboard"], t["frames"], t["plays"], t["validation"], t["draft_link"]
m, stab, out = val["model"], val["stability"], val["outcome"]
q = out["by_quintile"]


def num(x, fmt="{:.2f}"):
    return "n/a" if x is None else fmt.format(x)


k = st.columns(3)
with k[0]:
    kpi(
        f"{m['lightgbm']['rmse']:.2f} yd", "model error", f"vs {m['end_equals_start']['rmse']:.2f} yd for a naive guess"
    )
with k[1]:
    kpi(num(stab["split_half_r"]), "stability (split-half r)", f"{stab['n_players']} players, odd vs even weeks", MUTED)
with k[2]:
    kpi(f"{out['q5_minus_q1_completion'] * 100:+.0f} pts", "completion rate, best vs worst quintile",
        f"{pct(q[-1]['completion_rate'])} vs {pct(q[0]['completion_rate'])}, {out['n_plays']:,} plays", ATHLETIC)  # fmt: skip
st.write("")

left, right = st.columns([3, 2])
with left:
    st.subheader("Leaderboard")
    c1, c2 = st.columns([2, 1])
    pos = c1.multiselect("Position", sorted(lb["position"].unique()), default=[], placeholder="All positions")
    top = int(lb["plays"].max())
    min_plays = c2.slider(
        "Minimum plays", int(lb["plays"].min()), max(top, int(lb["plays"].min()) + 1), int(lb["plays"].min())
    )
    f = lb[(lb["plays"] >= min_plays) & (lb["position"].isin(pos) if pos else True)]
    st.dataframe(
        f[["player_name", "position", "team", "plays", "mean_coe", "coe_se"]].rename(
            columns={
                "player_name": "Player",
                "position": "Pos",
                "team": "Team",
                "plays": "Plays",
                "mean_coe": "COE / play",
                "coe_se": "Std. error",
            }  # fmt: skip
        ),
        hide_index=True,
        width="stretch",
        height=380,
        column_config={
            "COE / play": st.column_config.NumberColumn(format="%+.2f", help="Yards closer than expected per play"),
            "Std. error": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(f"{len(f)} defenders. One season is a noisy read, so mind the standard error.")
with right:
    st.subheader("Does it matter?")
    fig = go.Figure(
        go.Bar(
            x=[f"Q{r['quintile']}" for r in q],
            y=[r["completion_rate"] for r in q],
            marker_color=MODEL,
            text=[pct(r["completion_rate"]) for r in q],
        )  # fmt: skip
    )
    style_fig(fig, 340, title="Completion rate by COE quintile", yaxis=dict(tickformat=".0%", range=[0.5, 0.8]),
              xaxis=dict(title="Q1 = least closing, Q5 = most"))  # fmt: skip
    st.plotly_chart(fig, width="stretch")

st.subheader("Play viewer")
if len(plays):
    from src.tracking.closing_animate import animate_closing

    desc = {
        (int(g), int(p)): d for g, p, d in plays[["game_id", "play_id", "play_description"]].itertuples(index=False)
    }
    g, p = st.selectbox("Play", list(desc), format_func=lambda k: str(desc[k])[:110])
    st.plotly_chart(animate_closing(frames, g, p), width="stretch")
    st.caption("Offense moves left to right. Star = catch point, time 0 = the throw. Hover a defender for his COE.")

with st.expander("How this works and how well it holds up"):
    st.markdown(
        f"- Expected closing comes from a model trained out-of-fold by game on {val['n_defender_plays']:,} defender-plays.\n"
        f"- Stability: odd vs even weeks correlate at r = {num(stab['split_half_r'])} "
        f"(Spearman-Brown {num(stab['spearman_brown'])}). Modest, so one season is noisy.\n"
        f"- Q5 vs Q1: completion {out['q5_minus_q1_completion'] * 100:+.0f} points, "
        f"EPA {out['q5_minus_q1_epa']:+.2f} per play.\n"
        "- It uses only what is known at the throw: it measures burst and ball-in-air play, not throw quality."
    )

with st.expander("Link to the draft (exploratory)"):
    if link is None or "correlations" not in val.get("draft_link", {}):
        st.info("The draft warehouse was not available when the metric was built, so the link is missing.")
    else:
        d = val["draft_link"]
        st.caption(
            f"{d['n_matched']} of {d['n_players']} tracked defenders ({pct(d['match_rate'])}) matched to a "
            "prospect. Does pre-draft athleticism or slot foreshadow closing?"
        )
        labels = {"speed_score": "Speed score", "athletic_score": "Athletic score", "agility_score": "Agility score",
                  "pick": "Draft pick", "p_starter": "Model P(starter)"}  # fmt: skip
        col = st.selectbox("Pre-draft feature", list(labels), format_func=labels.get)
        s = link[(link["plays"] >= d["correlations"]["min_plays"])].dropna(subset=[col])
        fig = go.Figure(go.Scatter(x=s[col], y=s["mean_coe"], mode="markers",
                                   text=s["player_name"] + " (" + s["position"] + ")", hoverinfo="text",
                                   marker=dict(color=ATHLETIC, size=8, line=dict(color="white", width=1))))  # fmt: skip
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        style_fig(fig, 340, xaxis_title=labels[col], yaxis_title="Mean COE (yd per play)")
        st.plotly_chart(fig, width="stretch")
        c = d["correlations"]["all"][col]
        st.caption(f"Spearman {num(c['spearman'], '{:+.2f}')} (n = {c['n']}): no clear relationship.")
