"""Big Board: model-ranked draft classes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import POS_GROUPS, load_board, pct, pick_label, select_player, setup  # noqa: E402

setup(
    "Big Board",
    ":material/format_list_numbered:",
    "Every class, ranked by chance to start. Click a row to open the player.",
)

board = load_board()
years = sorted(board["draft_year"].unique(), reverse=True)

f1, f2, f3, f4, f5 = st.columns([1, 3, 1.4, 1, 1])
year = f1.selectbox("Class", years, index=0)
groups = f2.multiselect("Position group", POS_GROUPS, default=[], placeholder="All positions")
sort_by = f3.selectbox("Sort by", ["Chance to start", "Vs. his pick", "Pick"])
with_udfa = f4.toggle("UDFA", value=False, help="Add combine invitees nobody drafted")
details = f5.toggle("Details", value=False, help="Add the 80% range and what really happened")

df = board[board["draft_year"] == year]
if not with_udfa:
    df = df[df["undrafted"] == 0]
if groups:
    df = df[df["pos_group"].isin(groups)]
df = df.sort_values({"Chance to start": "p_starter", "Vs. his pick": "value_over_slot", "Pick": "pick"}[sort_by],
                    ascending=sort_by == "Pick")  # fmt: skip

known = df["snap_share_y1"].notna().any()
if df["prediction_type"].eq("in_sample").all():
    st.warning(
        "The model saw this class's results while training, so these scores look better than they should. Honest tests start with the 2017 class."
    )

view = pd.DataFrame(
    {
        "Pick": df["pick"].map(pick_label),
        "Player": df["player_name"],
        "Pos": df["position"],
        "College": df["college"],
        "Chance to start": df["p_starter"] * 100,
        "From pick alone": df["p_pick_only"] * 100,
        "Vs. his pick": (df["value_over_slot"] * 100).round(0) + 0.0,
        "Confidence": df["confidence"],
    }
)
cfg = {
    "Pick": st.column_config.TextColumn(width="small"),
    "Pos": st.column_config.TextColumn(width="small"),
    "Chance to start": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    "From pick alone": st.column_config.NumberColumn(format="%.0f%%", help="What the pick number alone predicts"),
    "Vs. his pick": st.column_config.NumberColumn(format="%+.0f pts", help="Model minus pick-only chance"),
    "Confidence": st.column_config.TextColumn(width="small"),
}
if details:
    view["80% low"], view["80% high"] = df["q10"] * 100, df["q90"] * 100
    cfg["80% low"] = st.column_config.NumberColumn(
        format="%.0f%%", help="Low end of the likely snap share (10th percentile)"
    )
    cfg["80% high"] = st.column_config.NumberColumn(
        format="%.0f%%", help="High end of the likely snap share (90th percentile)"
    )
    if known:
        view["Real snap share"] = df["snap_share_3yr"].to_numpy() * 100
        view["Starter"] = df["starter"].map({1.0: "Yes", 0.0: "No"}).fillna("Pending").to_numpy()
        cfg["Real snap share"] = st.column_config.NumberColumn(format="%.0f%%")


def tint(v):
    return "" if pd.isna(v) else f"color: {'#2a78d6' if v >= 0 else '#eb6834'}; font-weight: 600"


event = st.dataframe(view.style.map(tint, subset=["Vs. his pick"]), column_config=cfg, hide_index=True,
                     width="stretch", height=560, on_select="rerun", selection_mode="single-row")  # fmt: skip
rows = event.selection.rows
if rows:
    select_player(df.iloc[rows[0]]["player_key"])
    st.page_link("pages/2_Player_Card.py", label=f"Open {df.iloc[rows[0]]['player_name']}'s card",
                 icon=":material/person:")  # fmt: skip
else:
    st.caption("Vs. his pick is in percentage points. Positive means the model likes him more than the draft did.")

if known:
    obs = df[df["starter"].notna()]
    if len(obs):
        st.caption(
            f"What happened: {pct(obs['starter'].mean())} of this view became starters "
            f"({int(obs['starter'].sum())} of {len(obs)}). The model expected {pct(obs['p_starter'].mean())}."
        )
