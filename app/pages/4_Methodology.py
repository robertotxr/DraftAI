"""Methodology in scout / GM language."""

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
    MODEL_LABELS,
    MUTED,
    ROOT,
    feature_label,
    kpi,
    load_metrics,
    load_shap_global,
    pct,
    setup,
    style_fig,
)

setup("Methodology", ":material/science:", "What we predict, how we test it, and where it falls short.")
m = load_metrics()
boot, w = m["bootstrap_blend_vs_pick"], m["blend_weights"]
slot, blend = m["models"]["pick_only"], m["models"][m["chosen_model"]]
last = str(max(int(k) for k in w))

st.subheader("The question")
with st.container(border=True):
    st.markdown(
        "- **Starter** = averages 50%+ of team snaps over his first three NFL seasons.\n"
        "- We predict that chance, plus an 80% range for the snap share.\n"
        "- The real question: **how much does scouting data add beyond the draft slot?**"
    )
with st.expander("Why snaps, not Approximate Value?"):
    st.markdown(
        "- AV is career-cumulative, so it cannot be cut into clean per-season outcomes.\n"
        "- Snaps are observed every season and show how much the staff trusts the player.\n"
        "- Caveat: high picks get snaps partly because teams are invested in them. That is why the slot is a "
        "strong baseline."
    )

st.subheader("Honest result")
c = st.columns(3)
with c[0]:
    kpi(f"{slot['brier_skill'] * 100:.1f}% to {blend['brier_skill'] * 100:.1f}%", "Brier skill, slot vs blend",
        "share of naive-guess error removed", MODEL)  # fmt: skip
with c[1]:
    kpi(f"{boot['brier_gain']:.4f}", "Brier gain over slot", f"95% interval {boot['ci95'][0]:.4f} to {boot['ci95'][1]:.4f}", MARKET)  # fmt: skip
with c[2]:
    kpi(pct(boot["p_better"]), "bootstrap runs where blend wins", "interval includes zero: modest, not proven", MUTED)
st.write("")
st.markdown(
    "- The draft market already prices most of what public data can see.\n"
    "- Models without the slot lose to it clearly.\n"
    "- The model is most useful at the margin: players the slot over- or under-rates."
)

st.subheader("Models compared")
rows = [
    {"Model": MODEL_LABELS[k], "Brier": v["brier"], "Skill": v["brier_skill"] * 100, "AUC": v["auc"], "ECE": v["ece"]}
    for k, v in m["models"].items()
]
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Brier": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
        "Skill": st.column_config.NumberColumn(format="%.1f%%", help="1 minus Brier / Brier of the base rate"),
        "AUC": st.column_config.NumberColumn(format="%.3f"),
        "ECE": st.column_config.NumberColumn(format="%.3f", help="Calibration error, lower is better"),
    },
)
with st.expander("What each model is"):
    st.markdown(
        "- **Draft slot only**: the market baseline (pick number, per position group).\n"
        "- **Athleticism / scouting only**: testing and college production without the pick.\n"
        "- **Full model**: scouting plus slot, single or per position.\n"
        f"- **Blend** (final): combines slot-only and full model, weights learned on earlier classes "
        f"(class {last}: slot {w[last]['slot']}, model {w[last]['model']})."
    )

st.subheader("Tested out of sample")
with st.container(border=True):
    st.markdown(
        "- Class Y is scored by a model trained on classes up to Y minus 3: no random splits, no peeking.\n"
        f"- Headline results: classes {m['classes'][0]}-{m['classes'][-1]} ({m['n_test']:,} players).\n"
        "- Classes before 2017 are scored in sample; the newest classes are scored as of draft night."
    )

st.subheader("Calibration")
st.caption("When we say 60%, do about 60% of such players become starters?")
c1, c2 = st.columns(2)
for col, name in ((c1, "reliability.png"), (c2, "model_skill.png")):
    img = ROOT / "reports" / "figures" / name
    if img.exists():
        col.image(str(img), width="stretch")
    else:
        col.info(f"{name} not found. Run the model training step.")

st.subheader("Outcome ranges")
c = st.columns(3)
with c[0]:
    kpi(pct(m["interval_80_coverage"]), "coverage of the 80% range", "out of sample", ATHLETIC)
with c[1]:
    kpi(f"{m['interval_median_abs_error'] * 100:.1f} pts", "median snap-share error", "average miss", MUTED)
with c[2]:
    kpi("High / Med / Low", "confidence grade", "data and range quality", MODEL)
with st.expander("How ranges are built"):
    st.markdown(
        "- Quantile models (10th, 50th, 90th percentile), widened by cross-conformal calibration.\n"
        "- Calibrated separately for round 1, rounds 2-3, rounds 4-7 and undrafted players."
    )

st.subheader("What drives the model")
sg = load_shap_global().head(10)
fig = go.Figure(go.Bar(x=sg["mean_abs_shap"][::-1], y=sg["feature"].map(feature_label)[::-1], orientation="h",
                       marker_color=MODEL))  # fmt: skip
style_fig(fig, 340, xaxis=dict(title="Average influence (log-odds). Draft slot dominates."))
st.plotly_chart(fig, width="stretch")

st.subheader("Undrafted players")
u = m["undrafted"]
um = u["models"][m["base_model"]]
c = st.columns(3)
with c[0]:
    kpi(
        str(u["starters"]),
        f"starters among {u['n']:,} undrafted",
        f"classes {u['classes'][0]}-{u['classes'][-1]}",
        MARKET,
    )
with c[1]:
    kpi(pct(um["top_decile_snap_share"], 1), "snap share, model's top decile", f"vs {pct(um['rest_snap_share'], 1)} for the rest", MODEL)  # fmt: skip
with c[2]:
    kpi(f"{um['auc']:.2f}", "AUC", "ranking, not probability", MUTED)
st.caption("With so few undrafted starters, probabilities are rough and there is no value over slot.")

st.subheader("Limitations")
with st.container(border=True):
    st.markdown(
        "- Gain over the draft slot is small and not statistically significant.\n"
        "- Snap share favours high picks: teams are invested in them.\n"
        "- Combine results only: no pro-day data, so non-attendees lack athletic scores.\n"
        "- College defensive stats start in 2016; offensive linemen have no production stats.\n"
        "- Target share is a proxy; routes run are not in free data.\n"
        "- Four backtest classes and small position groups limit every claim."
    )
with st.expander("Athletic scores and comps"):
    st.markdown(
        "- Each drill is a 0-10 percentile within the position; size-adjusted scores compare similar builds.\n"
        "- Missing drills stay missing; the overall score needs 3+ drills.\n"
        "- Comps are nearest neighbours drawn only from classes known before the draft."
    )
