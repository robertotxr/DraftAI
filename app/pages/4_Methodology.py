"""Methodology in scout / GM language."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import MODEL_LABELS, ROOT, feature_label, load_metrics, load_shap_global, pct, setup  # noqa: E402

setup("Methodology", ":material/science:")
m = load_metrics()
st.title("Methodology")

st.header("What we predict")
st.markdown(
    """
A player is a **starter** if he averages **50% or more of his team's offensive or defensive snaps over his first
three NFL seasons**. Seasons in which he never played count as zero. We also predict an 80% range for that
3-year average snap share.

**Why snaps and not Approximate Value (AV)?** The AV in nflverse is a career-cumulative number, so it cannot be
cut into clean per-season outcomes, and it rewards longevity and team success. Snap counts are observed every
season and reflect something a GM cares about directly: **how much the coaching staff trusts the player on the field**.

**The draft-capital caveat.** High picks get playing time partly *because* they were high picks: teams
are invested in them. Snap share therefore blends talent with organizational commitment. This is why the draft
slot is such a strong baseline, and why the question we ask is how much *beyond the slot* the scouting data adds.
"""
)

st.header("Walk-forward validation")
st.markdown(
    f"""
Class Y is scored only by a model trained on classes up to **Y minus 3**, because a class's 3-year outcome is not
known until three seasons after its draft. No random splits, no peeking: the code asserts that no feature is
post-draft and no training label was unknown on the test draft night. Headline results use classes
{m["classes"][0]}-{m["classes"][-1]} ({m["n_test"]:,} players). Classes 2013-2016 have too little history and are
scored in sample; 2024-2026 are scored as of draft night with outcomes still arriving.
"""
)

st.header("Baselines and the blend")
rows = [
    {
        "Model": MODEL_LABELS[k],
        "Brier": v["brier"],
        "Skill vs base rate": v["brier_skill"],
        "AUC": v["auc"],
        "ECE": v["ece"],
    }
    for k, v in m["models"].items()
]
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Brier": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
        "Skill vs base rate": st.column_config.NumberColumn(
            format="percent", help="1 minus Brier / Brier of the base rate"
        ),
        "AUC": st.column_config.NumberColumn(format="%.3f"),
        "ECE": st.column_config.NumberColumn(format="%.3f", help="Calibration error, lower is better"),
    },
)
boot = m["bootstrap_blend_vs_pick"]
w = m["blend_weights"]
last = str(max(int(k) for k in w))
st.markdown(
    f"""
- **Draft slot only** is the market baseline. A logistic model on pick number, fit per position group.
- **Athleticism only** and **scouting (no slot)** show what testing and college production say without the pick.
  Both are clearly worse than the slot.
- **Full model** adds the slot to the scouting features. **Single** fits one model for all positions;
  **per position** fits one per group. We pick between them on classes before the backtest.
- **Blend** is the final product: a logistic regression that combines the slot-only and model probabilities, with weights
  learned only from earlier out-of-sample classes (latest weights, class {last}: slot {w[last]["slot"]}, model {w[last]["model"]}).

Against the slot alone, the blend improves Brier by **{boot["brier_gain"]:.4f}** (95% interval
{boot["ci95"][0]:.4f} to {boot["ci95"][1]:.4f}; better in {pct(boot["p_better"])} of paired bootstrap resamples).
The interval includes zero, so this is **evidence of a modest edge, not proof**. Individual models that drop the slot
lose to it clearly, which is the real finding: *the draft market already prices most of what public data can see.*
"""
)

st.header("Calibration")
st.markdown(
    "When we say 60%, do about 60% of such players become starters? The reliability chart checks this, "
    "and the skill chart compares every model."
)
c1, c2 = st.columns(2)
for col, name in ((c1, "reliability.png"), (c2, "model_skill.png")):
    img = ROOT / "reports" / "figures" / name
    if img.exists():
        col.image(str(img), width="stretch")
    else:
        col.info(f"{name} not found. Run the model training step to regenerate it.")

st.header("Outcome ranges")
st.markdown(
    f"""
The 80% range comes from quantile models (10th, 50th, 90th percentile of 3-year snap share), widened by
**cross-conformal calibration**: leave-classes-out residuals within the training classes, computed separately for
round 1, rounds 2-3, rounds 4-7 and undrafted players. Out of sample, the range covered the true outcome
**{pct(m["interval_80_coverage"])}** of the time against an 80% target, and the median was off by
{m["interval_median_abs_error"] * 100:.1f} points of snap share on average.
{"Ranges run slightly wide (conservative)." if m["interval_80_coverage"] >= 0.8 else "Treat ranges as somewhat optimistic."}
The **confidence grade** (High / Medium / Low) reflects data completeness (testing, college record, consensus rank) and range width.
"""
)

st.header("What drives the model")
sg = load_shap_global().head(12)
sg = sg.assign(Feature=sg["feature"].map(feature_label))
st.bar_chart(sg.set_index("Feature")["mean_abs_shap"], horizontal=True, color="#2a78d6")
st.caption("Mean absolute SHAP contribution (log-odds) in the explanation model. Draft slot dominates.")

st.header("Undrafted players")
u = m["undrafted"]
um = u["models"][m["base_model"]]
st.markdown(
    f"""
Combine invitees who went undrafted are scored too ({u["n"]:,} labeled players in the walk-forward classes
{u["classes"][0]}-{u["classes"][-1]}). Only **{u["starters"]}** became starters, so the useful question is ranking, not
probability. The classifiers are trained on drafted players; an undrafted term in the blend sets the probability level.
Their predicted P(starter) averaged {pct(u["held_out"]["mean_p_blend"], 1)} in the backtest classes, against an actual
{pct(u["held_out"]["starter_rate"], 1)}. The model's top decile averaged {pct(um["top_decile_snap_share"], 1)} of snaps
over three years, against {pct(um["rest_snap_share"], 1)} for the rest. There is no slot price for these players, so
value over slot is blank.
"""
)

st.header("Athletic scores and comps")
st.markdown(
    """
Each drill is scored 0-10 as a percentile within the position group; the **size-adjusted** version compares a player
with others of similar height and weight (a 4.55 forty at 250 lb outranks a 4.55 at 190 lb). A missing drill stays missing:
nothing is imputed, and the overall athletic score appears only with at least 3 drills tested.
Comps are nearest neighbours on position-specific profiles, drawn only from classes whose outcomes were known
before the prospect's draft.
"""
)

st.header("Limitations")
st.markdown(
    """
- No 10-yard split and no pro-day data: only combine results, so non-attendees lack athletic scores.
- College defensive stats exist only from 2016 (CFBD), so earlier defenders have missing, not zero, production.
- Offensive linemen have no production stats; they are judged on slot, size and testing.
- Undrafted players are covered only if they were invited to the combine; with so few undrafted starters, their
  probabilities are rough.
- Target share is a usage proxy; true targets and routes run (yards per route run) are not in free data.
- Snap share depends on scheme, depth chart and injuries, and mixes talent with draft-capital commitment.
- Small samples per position group and a four-class backtest mean the confidence on any one claim is limited.
"""
)
