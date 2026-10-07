"""Big Board: model-ranked draft classes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.lib import POS_GROUPS, load_board, pct, select_player, setup  # noqa: E402

setup("Big Board", ":material/format_list_numbered:")
st.title("Big Board")

board = load_board()
years = sorted(board["draft_year"].unique(), reverse=True)

f1, f2, f3 = st.columns([1, 3, 1])
year = f1.selectbox("Draft class", years, index=0)
groups = f2.multiselect("Position group", POS_GROUPS, default=[], placeholder="All positions")
sort_by = f3.selectbox("Sort by", ["P(starter)", "Value over slot", "Pick"])

df = board[board["draft_year"] == year]
if groups:
    df = df[df["pos_group"].isin(groups)]
df = df.sort_values({"P(starter)": "p_starter", "Value over slot": "value_over_slot", "Pick": "pick"}[sort_by],
                    ascending=sort_by == "Pick")  # fmt: skip

known = df["snap_share_y1"].notna().any()
if df["prediction_type"].eq("in_sample").all():
    st.warning("These classes are scored in sample (the model saw their outcomes). Walk-forward classes start in 2017.")
else:
    st.caption("Predictions are as of draft night: the model only used classes whose 3-year outcome was already known.")

view = pd.DataFrame(
    {
        "Pick": df["pick"],
        "Player": df["player_name"],
        "Pos": df["position"],
        "College": df["college"],
        "P(starter)": df["p_starter"],
        "Slot-only P": df["p_pick_only"],
        "Value over slot": df["value_over_slot"],
        "Range low": df["q10"],
        "Range high": df["q90"],
        "Confidence": df["confidence"],
    }
)
cfg = {
    "Pick": st.column_config.NumberColumn(format="%d", width="small"),
    "P(starter)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
    "Slot-only P": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
    "Value over slot": st.column_config.NumberColumn(
        format="%+.1f%%", help="Model P(starter) minus slot-only P, in points"
    ),
    "Range low": st.column_config.NumberColumn("80% low", format="percent", help="10th percentile of 3-yr snap share"),
    "Range high": st.column_config.NumberColumn(
        "80% high", format="percent", help="90th percentile of 3-yr snap share"
    ),
}
view["Value over slot"] = view["Value over slot"] * 100
if known:
    view["Seasons observed"] = df[["snap_share_y1", "snap_share_y2", "snap_share_y3"]].notna().sum(axis=1).to_numpy()
    view["Actual 3-yr snap share"] = df["snap_share_3yr"].to_numpy()
    view["Starter"] = df["starter"].map({1.0: "Yes", 0.0: "No"}).fillna("Pending").to_numpy()
    cfg["Actual 3-yr snap share"] = st.column_config.NumberColumn(format="percent")
    cfg["Seasons observed"] = st.column_config.NumberColumn(format="%d", width="small")

st.caption("Select a row, then open the Player Card. Value over slot is in percentage points.")
event = st.dataframe(view, column_config=cfg, hide_index=True, width="stretch", height=600,
                     on_select="rerun", selection_mode="single-row")  # fmt: skip
rows = event.selection.rows
if rows:
    key = df.iloc[rows[0]]["player_key"]
    select_player(key)
    st.success(f"Selected {df.iloc[rows[0]]['player_name']}.")
    st.page_link("pages/2_Player_Card.py", label="Open Player Card", icon=":material/person:")

if known:
    obs = df[df["starter"].notna()]
    if len(obs):
        st.caption(
            f"Realized: {int(obs['starter'].sum())} of {len(obs)} players in this view became starters "
            f"({pct(obs['starter'].mean())}); model expected {pct(obs['p_starter'].mean())}."
        )
