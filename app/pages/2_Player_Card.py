"""Player Card: one prospect in depth."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import (  # noqa: E402
    ATHLETIC,
    MARKET,
    MODEL,
    MUTED,
    PERCENT_METRICS,
    PRODUCTION_BY_GROUP,
    draft_text,
    feature_label,
    kpi,
    load_board,
    load_comps,
    load_shap_row,
    outcome_text,
    pct,
    pick_label,
    setup,
    style_fig,
)

setup("Player Card", ":material/person:", "One prospect: the call, the profile, what drives it, and who he looks like.")

board = load_board()
labels = {
    r.player_key: f"{r.player_name} ({r.position}, {r.college}, {r.draft_year} #{pick_label(r.pick)})"
    for r in board.itertuples()
}
keys = list(labels)
default = st.session_state.get("player_key")
if default not in labels:
    # Showcase a fully profiled prospect: tested at the combine and with college production.
    latest = board[(board["draft_year"] == board["draft_year"].max()) & board["athletic_score"].notna()
                   & board["college_seasons"].notna() & (board["pos_group"] != "OL")]  # fmt: skip
    default = latest.sort_values("p_starter").iloc[-1]["player_key"]
st.session_state["player_key"] = default  # a widget key on this page would be dropped when leaving it
key = st.selectbox("Player", keys, index=keys.index(default), format_func=labels.get)
st.session_state["player_key"] = key
p = board[board["player_key"] == key].iloc[0]

body = (
    f"{p['ht_in'] // 12:.0f}'{p['ht_in'] % 12:.0f}\" {p['wt_lb']:.0f} lb"
    if pd.notna(p["ht_in"]) and pd.notna(p["wt_lb"])
    else ""
)
with st.container():
    st.markdown(f'<div class="pname">{p["player_name"]}</div>', unsafe_allow_html=True)
    chips = [p["position"], p["college"], f"Class of {p['draft_year']}", draft_text(p), body]
    st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips if c), unsafe_allow_html=True)
    st.write("")
    k = st.columns([4, 3, 3, 2], gap="small")
    with k[0]:
        kpi(
            pct(p["p_starter"]),
            "chance to become a starter",
            "" if p["undrafted"] == 1 else f"{p['value_over_slot'] * 100:+.0f} pts vs his pick",
            big=True,
        )
    with k[1]:
        und = p["undrafted"] == 1
        kpi("n/a" if und else pct(p["p_pick_only"]), "chance from the pick alone",
            "Undrafted, so no pick to read" if und else "Ignores everything but the slot")  # fmt: skip
    with k[2]:
        kpi(f"{p['q10'] * 100:.0f}-{p['q90'] * 100:.0f}%", "likely snap share (80% range)", f"Median {pct(p['q50'])}")
    with k[3]:
        kpi(str(p["confidence"]), "confidence", "Data and range quality")
if p["prediction_type"] == "in_sample":
    st.info("The model saw this class's results while training. Treat the numbers as a best case.")

# Actual outcome
if pd.notna(p["snap_share_y1"]):
    st.caption(f"In the NFL: {outcome_text(p)}")
    obs = {f"Year {k}": p[f"snap_share_y{k}"] for k in (1, 2, 3) if pd.notna(p[f"snap_share_y{k}"])}
    fig = go.Figure(go.Bar(x=list(obs), y=list(obs.values()), marker_color=MODEL, text=[pct(v) for v in obs.values()]))
    fig.add_hline(
        y=0.5, line_dash="dot", line_color=MARKET, annotation_text="starter line (50%)", annotation_position="top left"
    )
    style_fig(fig, 220, yaxis=dict(range=[0, 1], tickformat=".0%"), title="Share of team snaps, by NFL season")
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("In the NFL: too early to tell.")

left, right = st.columns(2)

# Athletic radar
with left:
    st.subheader("Athletic profile")
    size = p[["ht_in_score", "wt_lb_score"]].mean()
    axes = {
        "Speed": p["speed_score"],
        "Explosion": p["explosion_score"],
        "Agility": p["agility_score"],
        "Size": size,
        "Strength (bench)": p["bench_size_score"],
    }
    if pd.isna(p["athletic_score"]):
        st.info("Fewer than 3 drills tested, so no overall athletic score.")
    else:
        st.caption(f"Athletic score {p['athletic_score']:.1f} out of 10, against his position group.")
    names = list(axes)[::-1]
    vals = [axes[n] for n in names]
    fig = go.Figure(
        go.Bar(
            x=[0 if pd.isna(v) else v for v in vals],
            y=names,
            orientation="h",
            marker_color=[ATHLETIC if pd.notna(v) else MUTED for v in vals],
            text=["not tested" if pd.isna(v) else f"{v:.1f}" for v in vals],
            textposition="outside",
            width=0.55,
        )
    )
    style_fig(fig, 280, xaxis=dict(range=[0, 11.5], tickvals=[0, 2, 4, 6, 8, 10], title="0-10, against his position"),
              showlegend=False, margin=dict(l=0, r=10, t=10, b=0))  # fmt: skip
    st.plotly_chart(fig, width="stretch")
    missing = [n for n, v in axes.items() if pd.isna(v)]
    if missing:
        st.caption(f"Not tested: {', '.join(missing)}. Left blank, never guessed.")
    raw = {"40-yard": "forty", "Vertical": "vertical", "Broad jump": "broad_jump", "3-cone": "cone",
           "Shuttle": "shuttle", "Bench": "bench"}  # fmt: skip
    drills = " | ".join(f"{n} {p[c]:g}" for n, c in raw.items() if pd.notna(p[c]))
    st.caption(f"Raw drills: {drills or 'none recorded'}")

# Production
with right:
    st.subheader("College production")
    metrics = PRODUCTION_BY_GROUP[p["pos_group"]]
    if not metrics:
        st.info("No public college stats exist for offensive linemen.")
    else:
        rows = []
        for m in metrics:
            vals = [p.get(f"{m}_{a}") for a in ("final", "best", "career")]
            if all(pd.isna(v) for v in vals):
                continue
            f = (lambda v: pct(v, 1)) if m in PERCENT_METRICS else (lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
            rows.append({"Metric": feature_label(f"{m}_final").split(",")[0], "Final": f(vals[0]), "Best": f(vals[1]),
                         "Career": f(vals[2])})  # fmt: skip
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.TextColumn(width="small") for c in ("Final", "Best", "Career")})  # fmt: skip
        else:
            st.info("No college stats found for this player.")
        meta = [f"{int(p['college_seasons'])} college seasons"] if pd.notna(p["college_seasons"]) else []
        for col, nm in (
            ("breakout_age_rec", "receiving"),
            ("breakout_age_scrim", "scrimmage"),
            ("breakout_age_def", "defensive"),
        ):
            if pd.notna(p[col]) and ((nm == "defensive") == (p["pos_group"] in ("DL", "LB", "DB"))):
                meta.append(f"{nm} breakout age {p[col]:.1f}")
        if meta:
            st.caption(" | ".join(meta))

# SHAP
st.subheader("What moves the number")
sv = load_shap_row(key)
sv = sv[sv.abs() > 1e-9]
top = sv.reindex(sv.abs().sort_values(ascending=False).index)[:8][::-1]
fig = go.Figure(
    go.Bar(
        x=top.values,
        y=[feature_label(n) for n in top.index],
        orientation="h",
        marker_color=[MODEL if v > 0 else MARKET for v in top.values],
        text=[f"{v:+.2f}" for v in top.values],
        textposition="outside",
    )  # fmt: skip
)
style_fig(
    fig, 340, margin=dict(l=0, r=40, t=10, b=0), xaxis=dict(title="Blue pushes the chance up. Orange pushes it down.")
)
fig.update_xaxes(zeroline=True, zerolinecolor=MUTED)
st.plotly_chart(fig, width="stretch")

# Comps
st.subheader("Players he looks like")
cp = load_comps()
cp = (
    cp[cp["player_key"] == key]
    .sort_values("rank")
    .merge(board.rename(columns={"player_key": "comp_key"}), on="comp_key", how="left", suffixes=("", "_c"))
)
if cp.empty:
    st.info("No comps for this player.")
else:
    out = pd.DataFrame(
        {
            "Comp": cp["player_name"],
            "Pos": cp["position"],
            "Class": cp["draft_year"],
            "Pick": cp["pick"].map(pick_label),
            "Similarity": cp["similarity"],
            "Model said": cp["p_starter"] * 100,
            "Real snap share": cp["snap_share_3yr"] * 100,
            "Starter": cp["starter"].map({1.0: "Yes", 0.0: "No"}).fillna("Pending"),
        }
    )
    st.dataframe(
        out,
        hide_index=True,
        width="stretch",
        column_config={
            "Similarity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Model said": st.column_config.NumberColumn(format="%.0f%%"),
            "Real snap share": st.column_config.NumberColumn(format="%.0f%%"),
            "Class": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption("Comps only come from classes with known results before this player was drafted.")
