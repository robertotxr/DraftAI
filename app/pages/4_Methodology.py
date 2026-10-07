"""Methodology in scout / GM language."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import (  # noqa: E402
    MODEL,
    MODEL_LABELS,
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
        "- A starter plays 50%+ of team snaps over his first three NFL seasons.\n"
        "- We predict the chance of that, plus a likely range for his snap share.\n"
        "- The real question: does scouting data add anything beyond the draft slot?"
    )
with st.expander("Why snaps and not Approximate Value?"):
    st.markdown(
        "- AV adds up over a career. It can't be split into clean seasons.\n"
        "- Snaps are seen every season. They show how much the staff trusts the player.\n"
        "- Catch: high picks get snaps partly because teams paid for them. That makes the slot a tough baseline."
    )

st.subheader("The honest result")
c = st.columns([4, 3, 3], gap="medium")
with c[0]:
    kpi(f"{slot['brier_skill'] * 100:.1f}% to {blend['brier_skill'] * 100:.1f}%", "error removed vs a naive guess (Brier skill)",
        "Pick alone, then the blend", big=True)  # fmt: skip
with c[1]:
    kpi(
        f"{boot['brier_gain']:.4f}",
        "Brier gain over the pick",
        f"95% interval {boot['ci95'][0]:.4f} to {boot['ci95'][1]:.4f}",
    )
with c[2]:
    kpi(pct(boot["p_better"]), "of resamples where the blend wins", "The interval includes zero. Modest, not proven.")
st.write("")
st.markdown(
    "- The draft already prices most of what public data can see.\n"
    "- Models without the pick lose to it, clearly.\n"
    "- The model helps at the edges: players the pick over- or under-rates."
)

st.subheader("Every model, side by side")
rows = [
    {"Model": MODEL_LABELS[k], "Brier": v["brier"], "Skill": v["brier_skill"] * 100, "AUC": v["auc"], "ECE": v["ece"]}
    for k, v in m["models"].items()
]
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Brier": st.column_config.NumberColumn(format="%.4f", help="Average squared miss. Lower is better."),
        "Skill": st.column_config.NumberColumn(format="%.1f%%", help="Share of naive-guess error removed"),
        "AUC": st.column_config.NumberColumn(
            "Ranking (AUC)", format="%.3f", help="Chance a starter outranks a non-starter"
        ),
        "ECE": st.column_config.NumberColumn(
            "Calibration gap", format="%.3f", help="Expected calibration error. Lower is better."
        ),
    },
)
with st.expander("What each model is"):
    st.markdown(
        "- **Draft slot only**: the market baseline. Pick number, by position group.\n"
        "- **Athleticism or scouting only**: testing and college stats, no pick.\n"
        "- **Full model**: scouting plus the pick, one model or one per position.\n"
        f"- **Blend** (the final one): mixes slot-only and the full model. Weights come from earlier classes "
        f"(class {last}: slot {w[last]['slot']}, model {w[last]['model']})."
    )

st.subheader("Tested on the future")
with st.container(border=True):
    st.markdown(
        "- Each class is scored by a model trained only on classes at least 3 years older. No random splits. No peeking.\n"
        f"- Headline results: classes {m['classes'][0]}-{m['classes'][-1]} ({m['n_test']:,} players).\n"
        "- Classes before 2017 are scored in sample. The newest are scored as of draft night."
    )

st.subheader("Are the odds honest?")
st.caption("When we say 60%, do about 60% of those players start? That is calibration.")
c1, c2 = st.columns(2)
for col, name in ((c1, "reliability.png"), (c2, "model_skill.png")):
    img = ROOT / "reports" / "figures" / name
    if img.exists():
        col.image(str(img), width="stretch")
    else:
        col.info(f"{name} not found. Run the model training step.")

st.subheader("Snap-share ranges")
c = st.columns([4, 3, 3], gap="medium")
with c[0]:
    kpi(
        pct(m["interval_80_coverage"]),
        "of outcomes inside the 80% range",
        "Out of sample. The target is 80%.",
        big=True,
    )
with c[1]:
    kpi(f"{m['interval_median_abs_error'] * 100:.1f} pts", "typical snap-share miss", "Median absolute error")
with c[2]:
    kpi("High / Med / Low", "confidence grade", "Data and range quality")
with st.expander("How ranges are built"):
    st.markdown(
        "- Quantile models set the 10th, 50th and 90th percentile. Cross-conformal calibration widens them.\n"
        "- Calibrated separately for round 1, rounds 2-3, rounds 4-7 and undrafted players."
    )

st.subheader("What the model leans on")
sg = load_shap_global().head(10)
fig = go.Figure(go.Bar(x=sg["mean_abs_shap"][::-1], y=sg["feature"].map(feature_label)[::-1], orientation="h",
                       marker_color=MODEL))  # fmt: skip
style_fig(fig, 340, xaxis=dict(title="Average influence on the prediction (log-odds). The pick dominates."))
st.plotly_chart(fig, width="stretch")

st.subheader("Undrafted players")
u = m["undrafted"]
um = u["models"][m["base_model"]]
c = st.columns([4, 3, 3], gap="medium")
with c[0]:
    kpi(
        str(u["starters"]),
        f"starters among {u['n']:,} undrafted players",
        f"Classes {u['classes'][0]}-{u['classes'][-1]}",
        big=True,
    )
with c[1]:
    kpi(pct(um["top_decile_snap_share"], 1), "snap share for the model's top 10%", f"Everyone else: {pct(um['rest_snap_share'], 1)}")  # fmt: skip
with c[2]:
    kpi(f"{um['auc']:.2f}", "ranking accuracy (AUC)", "Ranks well. Odds are rough.")
st.caption("So few undrafted starters make the odds rough. And there is no pick to compare against.")

st.subheader("Limitations")
with st.container(border=True):
    st.markdown(
        "- The edge over the draft slot is small. It is not statistically significant.\n"
        "- Snap share favors high picks. Teams paid for them, so they play.\n"
        "- Public data misses medicals, pro days and routes run.\n"
        "- Combine results only. Players who skipped it have no athletic score.\n"
        "- College defensive stats start in 2016. Offensive linemen have none.\n"
        "- Target share is a proxy.\n"
        "- Few test classes and small position groups limit every claim."
    )
with st.expander("How athletic scores and comps work"):
    st.markdown(
        "- Each drill gets a 0-10 score against his position. Size-adjusted scores compare similar builds.\n"
        "- Missing drills stay missing. The overall score needs 3+ drills.\n"
        "- Comps are nearest neighbors, drawn only from classes with known results."
    )
