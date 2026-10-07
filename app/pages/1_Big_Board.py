"""Big Board: model-ranked draft classes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import POS_GROUPS, load_board, pct, pick_label, select_player, setup  # noqa: E402

setup("Big Board", ":material/format_list_numbered:", "Every draft class ranked by chance to become a starter.")

board = load_board()
years = sorted(board["draft_year"].unique(), reverse=True)

f1, f2, f3, f4, f5 = st.columns([1, 3, 1.4, 1, 1])
year = f1.selectbox("Draft class", years, index=0)
groups = f2.multiselect("Position group", POS_GROUPS, default=[], placeholder="All positions")
sort_by = f3.selectbox("Sort by", ["P(starter)", "Value over slot", "Pick"])
with_udfa = f4.toggle("UDFA", value=False, help="Add combine invitees who went undrafted")
details = f5.toggle("Details", value=False, help="Show 80% range and actual outcomes")

df = board[board["draft_year"] == year]
if not with_udfa:
    df = df[df["undrafted"] == 0]
if groups:
    df = df[df["pos_group"].isin(groups)]
df = df.sort_values({"P(starter)": "p_starter", "Value over slot": "value_over_slot", "Pick": "pick"}[sort_by],
                    ascending=sort_by == "Pick")  # fmt: skip

known = df["snap_share_y1"].notna().any()
if df["prediction_type"].eq("in_sample").all():
    st.warning("This class is scored in sample (the model saw its outcomes). Out-of-sample classes start in 2017.")

view = pd.DataFrame(
    {
        "Pick": df["pick"].map(pick_label),
        "Player": df["player_name"],
        "Pos": df["position"],
        "College": df["college"],
        "P(starter)": df["p_starter"] * 100,
        "Slot-only P": df["p_pick_only"] * 100,
        "Value over slot": df["value_over_slot"] * 100,
        "Confidence": df["confidence"],
    }
)
cfg = {
    "Pick": st.column_config.TextColumn(width="small"),
    "Pos": st.column_config.TextColumn(width="small"),
    "P(starter)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    "Slot-only P": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    "Value over slot": st.column_config.NumberColumn(format="%+.1f pts", help="Model minus slot-only P"),
    "Confidence": st.column_config.TextColumn(width="small"),
}
if details:
    view["80% low"], view["80% high"] = df["q10"] * 100, df["q90"] * 100
    cfg["80% low"] = st.column_config.NumberColumn(format="%.0f%%", help="10th percentile of 3-yr snap share")
    cfg["80% high"] = st.column_config.NumberColumn(format="%.0f%%", help="90th percentile of 3-yr snap share")
    if known:
        view["Actual snap share"] = df["snap_share_3yr"].to_numpy() * 100
        view["Starter"] = df["starter"].map({1.0: "Yes", 0.0: "No"}).fillna("Pending").to_numpy()
        cfg["Actual snap share"] = st.column_config.NumberColumn(format="%.0f%%")


def tint(v):
    return "" if pd.isna(v) else f"color: {'#1baf7a' if v >= 0 else '#eb6834'}; font-weight: 600"


event = st.dataframe(view.style.map(tint, subset=["Value over slot"]), column_config=cfg, hide_index=True,
                     width="stretch", height=560, on_select="rerun", selection_mode="single-row")  # fmt: skip
rows = event.selection.rows
if rows:
    select_player(df.iloc[rows[0]]["player_key"])
    st.page_link("pages/2_Player_Card.py", label=f"Open {df.iloc[rows[0]]['player_name']}'s card",
                 icon=":material/person:")  # fmt: skip
else:
    st.caption("Select a row to open the player card. Value over slot is in percentage points.")

if known:
    obs = df[df["starter"].notna()]
    if len(obs):
        st.caption(
            f"Realized: {pct(obs['starter'].mean())} of this view became starters "
            f"({int(obs['starter'].sum())} of {len(obs)}); the model expected {pct(obs['p_starter'].mean())}."
        )
