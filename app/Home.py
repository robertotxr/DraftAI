"""DraftAI home: headline results and navigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import (  # noqa: E402
    MARKET,
    MODEL,
    kpi,
    load_board,
    load_metrics,
    load_report,
    pct,
    pick_label,
    setup,
    sleeper_stats,
    style_fig,
)

setup("Home", ":material/sports_football:")
m, rep, board = load_metrics(), load_report(), load_board()
blend, slot = m["models"][m["chosen_model"]], m["models"]["pick_only"]
t40, und = rep["top40_upgrades"], rep["undrafted"]
udfa_top = sum(s["model_pct_rank"] >= 0.84 for s in und["starters"])

left, right = st.columns([5, 6], gap="medium", vertical_alignment="center")
with left:
    st.markdown(
        '<div class="hero"><h1>Who becomes a starter?<br><em>The draft slot is not the whole story.</em></h1>'
        "<p>DraftAI scores every prospect, then shows where it disagrees with the pick.</p></div>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Big_Board.py", label="Open the Big Board", icon=":material/arrow_forward:")
with right:
    st.markdown("**Starter rate, by how the model sees a player vs his pick**")
    buckets = rep["disagreement"]
    fig = go.Figure()
    fig.add_bar(name="Expected from pick", x=[b["bucket"] for b in buckets], y=[b["slot"] for b in buckets],
                marker_color=MARKET, text=[pct(b["slot"]) for b in buckets], textposition="outside")  # fmt: skip
    fig.add_bar(name="Actually became starters", x=[b["bucket"] for b in buckets], y=[b["actual"] for b in buckets],
                marker_color=MODEL, text=[pct(b["actual"]) for b in buckets], textposition="outside")  # fmt: skip
    style_fig(
        fig,
        330,
        barmode="group",
        bargap=0.3,
        legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"),
        yaxis=dict(tickformat=".0%", range=[0, 0.68]),
        xaxis=dict(title="Model's view vs draft slot (lowest to highest)"),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

st.write("")
c = st.columns([5, 3, 3, 3], gap="small")
with c[0]:
    kpi(pct(t40["actual"], 1), "of the model's 40 biggest upgrades became starters",
        f"The pick alone said {pct(t40['slot'], 1)}", big=True)  # fmt: skip
with c[1]:
    kpi(f"{blend['auc']:.3f}", "ranking accuracy (AUC)", f"Pick alone: {slot['auc']:.3f}")
with c[2]:
    kpi(pct(m["interval_80_coverage"]), "of outcomes land in the 80% range", "Out of sample, as promised")
with c[3]:
    kpi(f"{udfa_top} of {len(und['starters'])}", "undrafted starters sat in the model's top 16%",
        f"Out of {und['n']} undrafted invitees")  # fmt: skip
st.caption(
    f"Tested out of sample on classes {m['classes'][0]}-{m['classes'][-1]} ({m['n_test']:,} players). "
    "The edge over the draft slot is small. It is not statistically significant."
)

st.write("")
sl = sleeper_stats()
nav, side = st.columns([3, 2], gap="medium")
with nav:
    st.subheader("Explore")
    for page, name, line, icon in [
        ("pages/1_Big_Board.py", "Big Board", "Every class, ranked by chance to start.", ":material/format_list_numbered:"),
        ("pages/2_Player_Card.py", "Player Card", "One prospect, with comps and drivers.", ":material/person:"),
        ("pages/3_Tracking.py", "Tracking", "Who closes on receivers, from tracking data.", ":material/timeline:"),
        ("pages/4_Methodology.py", "Methodology", "How it works. Where it falls short.", ":material/science:"),
    ]:  # fmt: skip
        a, b = st.columns([2, 3], vertical_alignment="center")
        a.page_link(page, label=name, icon=icon)
        b.caption(line)
with side:
    if sl:
        with st.container(border=True):
            st.markdown("**Late-round sleepers**")
            kpi(pct(sl["top"]), "of rounds 4-7 picks in the model's top 10% became starters",
                f"Everyone else in those rounds: {pct(sl['rest'])}", big=True)  # fmt: skip
            st.caption(f"Out-of-sample classes only. {sl['n_top']} players in the top group.")

latest = board["draft_year"].max()
top = board[(board["draft_year"] == latest) & (board["undrafted"] == 0)].nlargest(6, "value_over_slot")
st.subheader(f"Biggest upgrades in the {latest} class")
st.dataframe(
    top.assign(
        v=top["value_over_slot"] * 100,
        p_starter=top["p_starter"] * 100,
        pick=top["pick"].map(pick_label),
    )[["pick", "player_name", "position", "college", "p_starter", "v"]].rename(
        columns={
            "pick": "Pick",
            "player_name": "Player",
            "position": "Pos",
            "college": "College",
            "p_starter": "Chance to start",
            "v": "Vs. his pick",
        }
    ),  # fmt: skip
    hide_index=True,
    width="stretch",
    column_config={
        "Pick": st.column_config.TextColumn(width="small"),
        "Pos": st.column_config.TextColumn(width="small"),
        "Chance to start": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Vs. his pick": st.column_config.NumberColumn(format="%+.0f pts"),
    },
)
with st.expander("How to read this"):
    st.markdown(
        "- **Starter**: plays 50%+ of team snaps over his first three NFL seasons.\n"
        "- **Chance to start**: the model's number.\n"
        "- **Vs. his pick**: the model minus what the pick number alone says. Positive means the model likes him more.\n"
        f"- Covers {len(board) - int(board['undrafted'].sum()):,} drafted players and "
        f"{int(board['undrafted'].sum()):,} undrafted combine invitees, "
        f"{board['draft_year'].min()}-{board['draft_year'].max()}."
    )
