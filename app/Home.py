"""DraftAI home: headline results and navigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.lib import (  # noqa: E402
    ATHLETIC,
    MARKET,
    MODEL,
    kpi,
    load_board,
    load_metrics,
    load_report,
    pct,
    pick_label,
    setup,
)

setup("Home", ":material/sports_football:")
m, rep, board = load_metrics(), load_report(), load_board()
blend, slot = m["models"][m["chosen_model"]], m["models"]["pick_only"]
t40, und = rep["top40_upgrades"], rep["undrafted"]
udfa_top = sum(s["model_pct_rank"] >= 0.84 for s in und["starters"])

st.markdown(
    '<div class="hero"><h1>DraftAI</h1><p>Which draft picks will become NFL starters, and where does the model '
    "disagree with the draft slot?</p></div>",
    unsafe_allow_html=True,
)
st.write("")

c = st.columns(4)
with c[0]:
    kpi(pct(t40["actual"], 1), "of the model's top-40 upgrades became starters", f"vs {pct(t40['slot'], 1)} expected from slot",
        MODEL)  # fmt: skip
with c[1]:
    kpi(f"{blend['auc']:.3f}", "ranking accuracy (AUC)", f"vs {slot['auc']:.3f} from draft slot alone", MODEL)
with c[2]:
    kpi(pct(m["interval_80_coverage"]), "of outcomes inside the 80% range", "out of sample, 80% target", ATHLETIC)
with c[3]:
    kpi(f"{udfa_top} of {len(und['starters'])}", "undrafted starters ranked in model's top 16%",
        f"{und['n']} undrafted invitees, classes 2017-2023", MARKET)  # fmt: skip
st.caption(
    f"Tested out of sample on classes {m['classes'][0]}-{m['classes'][-1]} ({m['n_test']:,} players). "
    "The edge over the draft slot is small and not statistically significant."
)

st.write("")
nav = [
    ("pages/1_Big_Board.py", "Big Board", "Every class ranked by chance to start.", ":material/format_list_numbered:"),
    ("pages/2_Player_Card.py", "Player Card", "One prospect in depth, with comps.", ":material/person:"),
    ("pages/3_Tracking.py", "Tracking", "Closing Over Expected from tracking data.", ":material/timeline:"),
    ("pages/4_Methodology.py", "Methodology", "How it works and where it falls short.", ":material/science:"),
]
for col, (page, name, line, icon) in zip(st.columns(4), nav, strict=True):
    with col, st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(line)
        st.page_link(page, label="Open", icon=icon)

latest = board["draft_year"].max()
top = board[(board["draft_year"] == latest) & (board["undrafted"] == 0)].nlargest(6, "value_over_slot")
st.subheader(f"Biggest upgrades of the {latest} class")
st.dataframe(
    top.assign(
        v=top["value_over_slot"] * 100,
        p_starter=top["p_starter"] * 100,
        p_pick_only=top["p_pick_only"] * 100,
        pick=top["pick"].map(pick_label),
    )[["pick", "player_name", "position", "college", "p_starter", "p_pick_only", "v"]].rename(
        columns={
            "pick": "Pick",
            "player_name": "Player",
            "position": "Pos",
            "college": "College",
            "p_starter": "P(starter)",
            "p_pick_only": "Slot-only P",
            "v": "Value over slot",
        }
    ),  # fmt: skip
    hide_index=True,
    width="stretch",
    column_config={
        "Pick": st.column_config.TextColumn(width="small"),
        "P(starter)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Slot-only P": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Value over slot": st.column_config.NumberColumn(format="▲ %.1f pts"),
    },
)
with st.expander("How this works"):
    st.markdown(
        "- **Starter**: averages 50%+ of team snaps over his first three NFL seasons.\n"
        f"- **P(starter)**: model chance. **Slot-only P**: chance from the pick number alone.\n"
        "- **Value over slot**: the gap, in points. Positive means the model likes him more than his pick.\n"
        f"- Covers {len(board) - int(board['undrafted'].sum()):,} drafted players and "
        f"{int(board['undrafted'].sum()):,} undrafted combine invitees, "
        f"{board['draft_year'].min()}-{board['draft_year'].max()}."
    )
