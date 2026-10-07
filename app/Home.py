"""DraftAI home: headline result, what the tool is, where to go next."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.lib import MARKET, MODEL, load_board, load_metrics, pct, setup  # noqa: E402

setup("Home", ":material/sports_football:")
m = load_metrics()
board = load_board()
blend, slot = m["models"][m["chosen_model"]], m["models"]["pick_only"]
boot = m["bootstrap_blend_vs_pick"]
classes = m["classes"]

st.title("DraftAI")
st.subheader("How likely is a drafted player to become a starter, and what does the draft slot already tell you?")

st.markdown(
    """
For every player drafted 2013-2026, DraftAI estimates the chance he plays at least half of his team's
snaps, on average, over his first three NFL seasons (our definition of a **starter**). It then asks the
question front offices care about: **how much does that estimate differ from what the draft slot alone implies?**
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Backtest players", f"{m['n_test']:,}", f"classes {classes[0]}-{classes[-1]}", delta_color="off")
c2.metric("Starter base rate", pct(m["base_rate"]))
c3.metric(
    "Model + slot skill",
    pct(blend["brier_skill"], 1),
    f"{(blend['brier_skill'] - slot['brier_skill']) * 100:+.1f} pts vs slot only",
)
c4.metric("AUC (ranking quality)", f"{blend['auc']:.3f}", f"{blend['auc'] - slot['auc']:+.3f} vs slot only")

st.markdown(
    f"""
**Headline.** Tested strictly out of sample (each class scored only with what was known on its draft night),
the draft slot is already a strong predictor: it explains {pct(slot["brier_skill"])} of the variance a naive
"everyone is {pct(m["base_rate"])}" guess leaves. Blending slot with college production and athletic testing
lifts that to {pct(blend["brier_skill"], 1)}. The gain is small: a paired bootstrap puts the Brier improvement at
{boot["brier_gain"]:.4f} (95% interval {boot["ci95"][0]:.4f} to {boot["ci95"][1]:.4f}), and the blend beats the
slot in {pct(boot["p_better"])} of resamples. So the honest read is **the market is hard to beat, and the model is
most useful at the margin**: flagging players the slot over- or under-rates, with a stated level of uncertainty.
The 80% outcome range covers the real 3-year snap share {pct(m["interval_80_coverage"])} of the time,
{"at or above its 80% target, so ranges are slightly conservative." if m["interval_80_coverage"] >= 0.8 else "below its 80% target, so treat ranges as slightly narrow."}
"""
)

st.divider()
left, right = st.columns(2)
with left:
    st.markdown("#### What you can do here")
    st.page_link(
        "pages/1_Big_Board.py",
        label="Big Board: every class ranked by P(starter)",
        icon=":material/format_list_numbered:",
    )
    st.page_link(
        "pages/2_Player_Card.py", label="Player Card: one prospect in depth, with comps", icon=":material/person:"
    )
    st.page_link(
        "pages/3_Tracking.py",
        label="Tracking: Tackles Over Expected from Big Data Bowl data",
        icon=":material/timeline:",
    )
    st.page_link(
        "pages/4_Methodology.py", label="Methodology: how it works and where it falls short", icon=":material/science:"
    )
with right:
    st.markdown("#### Reading the numbers")
    st.markdown(
        f"""
- <span style="color:{MODEL}">**P(starter)**</span>: model probability of a 50%+ snap share over three seasons.
- <span style="color:{MARKET}">**Slot-only P**</span>: the same probability from the pick number alone.
- **Value over slot**: the gap between the two. Positive means the model likes him more than his pick.
- **80% range**: where his 3-year average snap share should land 8 times out of 10.
""",
        unsafe_allow_html=True,
    )

n26 = int((board["draft_year"] == board["draft_year"].max()).sum())
st.caption(
    f"{len(board):,} drafted players, classes {board['draft_year'].min()}-{board['draft_year'].max()} "
    f"({n26} in the latest class). Outcomes for the latest classes are still being observed."
)
