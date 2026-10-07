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
    feature_label,
    load_board,
    load_comps,
    load_shap_row,
    outcome_text,
    pct,
    setup,
)

setup("Player Card", ":material/person:")
st.title("Player Card")

board = load_board()
labels = {
    r.player_key: f"{r.player_name} ({r.position}, {r.college}, {r.draft_year} #{r.pick})" for r in board.itertuples()
}
keys = list(labels)
default = st.session_state.get("player_key")
if default not in labels:
    default = board[board["draft_year"] == board["draft_year"].max()].sort_values("p_starter").iloc[-1]["player_key"]
st.session_state["player_key"] = default  # a widget key on this page would be dropped when leaving it
key = st.selectbox("Player", keys, index=keys.index(default), format_func=labels.get)
st.session_state["player_key"] = key
p = board[board["player_key"] == key].iloc[0]

# Header
st.header(p["player_name"])
st.caption(
    f"{p['position']} | {p['college']} | Class of {p['draft_year']} | Round {p['round']}, pick {p['pick']} "
    f"({p['team']}) | {p['ht_in'] // 12:.0f}'{p['ht_in'] % 12:.0f}\" {p['wt_lb']:.0f} lb"
    if pd.notna(p["ht_in"]) and pd.notna(p["wt_lb"])
    else f"{p['position']} | {p['college']} | Class of {p['draft_year']} | Round {p['round']}, pick {p['pick']} ({p['team']})"
)

# Key numbers
k1, k2, k3, k4 = st.columns(4)
k1.metric("P(starter)", pct(p["p_starter"]), f"{p['value_over_slot'] * 100:+.1f} pts vs slot")
k2.metric("Slot-only P(starter)", pct(p["p_pick_only"]))
k3.metric(
    "80% range, 3-yr snap share", f"{pct(p['q10'])} to {pct(p['q90'])}", f"median {pct(p['q50'])}", delta_color="off"
)
k4.metric("Confidence", p["confidence"])
if p["prediction_type"] == "in_sample":
    st.info("This class predates the walk-forward window: the model saw its outcomes, so the numbers are in sample.")

# Actual outcome
if pd.notna(p["snap_share_y1"]):
    st.markdown(f"**NFL outcome:** {outcome_text(p)}")
    obs = {f"Year {k}": p[f"snap_share_y{k}"] for k in (1, 2, 3) if pd.notna(p[f"snap_share_y{k}"])}
    fig = go.Figure(go.Bar(x=list(obs), y=list(obs.values()), marker_color=MODEL, text=[pct(v) for v in obs.values()]))
    fig.add_hline(y=0.5, line_dash="dot", line_color=MARKET, annotation_text="starter line (50%)")
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 1], tickformat=".0%"),
                      plot_bgcolor="white")  # fmt: skip
    st.plotly_chart(fig, width="stretch")
else:
    st.markdown("**NFL outcome:** not yet known (no NFL seasons observed).")

st.divider()
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
        st.info("Insufficient testing: fewer than 3 drills measured, so no overall athletic score is shown.")
    else:
        st.metric("Athletic score (0-10, vs position group)", f"{p['athletic_score']:.1f}")
    names, vals = list(axes), [None if pd.isna(v) else float(v) for v in axes.values()]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + vals[:1],
            theta=names + names[:1],
            fill="toself",
            connectgaps=False,
            line_color=ATHLETIC,
            fillcolor="rgba(27,175,122,0.25)",
            name=p["player_name"],
        )  # fmt: skip
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10])), showlegend=False,
                      height=380, margin=dict(l=40, r=40, t=20, b=20))  # fmt: skip
    st.plotly_chart(fig, width="stretch")
    missing = [n for n, v in axes.items() if pd.isna(v)]
    if missing:
        st.caption(f"Not tested (shown as gaps): {', '.join(missing)}. Nothing is imputed.")
    raw = {"40-yard": "forty", "Vertical": "vertical", "Broad jump": "broad_jump", "3-cone": "cone",
           "Shuttle": "shuttle", "Bench": "bench"}  # fmt: skip
    drills = " | ".join(f"{n} {p[c]:g}" for n, c in raw.items() if pd.notna(p[c]))
    st.caption(f"Raw drills: {drills or 'none recorded'}")

# Production
with right:
    st.subheader("College production")
    metrics = PRODUCTION_BY_GROUP[p["pos_group"]]
    if not metrics:
        st.info("No production stats for offensive linemen: public college data has none for blocking.")
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
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No college production matched for this player.")
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
st.divider()
st.subheader("Why the model says this")
sv = load_shap_row(key)
sv = sv[sv.abs() > 1e-9]
top = sv.reindex(sv.abs().sort_values(ascending=False).index)[:10][::-1]
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
fig.update_layout(height=400, margin=dict(l=0, r=40, t=10, b=0), plot_bgcolor="white",
                  xaxis=dict(title="Effect on log-odds of being a starter (right = more likely)", zeroline=True,
                             zerolinecolor=MUTED))  # fmt: skip
st.plotly_chart(fig, width="stretch")
st.caption("Contributions from the explanation model (blue raises, orange lowers). Draft slot usually dominates; "
           "the rest is what the scouting data adds.")  # fmt: skip

# Comps
st.divider()
st.subheader("Historical comps")
cp = load_comps()
cp = (
    cp[cp["player_key"] == key]
    .sort_values("rank")
    .merge(board.rename(columns={"player_key": "comp_key"}), on="comp_key", how="left", suffixes=("", "_c"))
)
if cp.empty:
    st.info("No comps available (too few shared features).")
else:
    out = pd.DataFrame(
        {
            "Comp": cp["player_name"],
            "Pos": cp["position"],
            "Class": cp["draft_year"],
            "Pick": cp["pick"],
            "Similarity": cp["similarity"],
            "P(starter) then": cp["p_starter"],
            "3-yr snap share": cp["snap_share_3yr"],
            "Starter": cp["starter"].map({1.0: "Yes", 0.0: "No"}).fillna("Pending"),
        }
    )
    st.dataframe(
        out,
        hide_index=True,
        width="stretch",
        column_config={
            "Similarity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "P(starter) then": st.column_config.NumberColumn(format="percent"),
            "3-yr snap share": st.column_config.NumberColumn(format="percent"),
            "Class": st.column_config.NumberColumn(format="%d"),
            "Pick": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption("Comps come only from classes whose 3-year outcome was known before this player's draft.")
