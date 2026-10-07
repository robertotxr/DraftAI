"""Build model-ready marts from staging.

marts.athletic          every combine participant, scored
marts.college_seasons   every college player-season with context-adjusted production
marts.prospects         one row per drafted player: pre-draft features + NFL outcomes (labels)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import cfg
from src.db import Warehouse
from src.features import athletic, production

log = logging.getLogger(__name__)

P5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "FBS Independents"}


def outcomes(picks: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """First-three-season snap shares. Seasons with no snaps count as 0; unobserved seasons are NaN."""
    t = cfg()["target"]
    last_season = cfg()["years"]["last_nfl_season"]
    out = picks[["player_key", "pfr_player_id", "draft_year"]].copy()
    sh = shares.set_index(["pfr_player_id", "season"])["snap_share"]
    for k in range(t["horizon_seasons"]):
        season = out["draft_year"] + k
        vals = pd.Series(list(zip(out["pfr_player_id"], season))).map(sh.to_dict()).astype(float).fillna(0).to_numpy()
        out[f"snap_share_y{k + 1}"] = np.where(season <= last_season, vals, np.nan)
    ycols = [f"snap_share_y{k + 1}" for k in range(t["horizon_seasons"])]
    out["snap_share_3yr"] = out[ycols].mean(axis=1, skipna=False)
    out["starter"] = (
        (out["snap_share_3yr"] >= t["starter_threshold"]).astype(float).where(out["snap_share_3yr"].notna())
    )
    # Labels for class Y become known after season Y + horizon - 1, i.e. before the draft of Y + horizon.
    out["label_known_from_draft"] = out["draft_year"] + t["horizon_seasons"]
    return out.drop(columns=["pfr_player_id", "draft_year"])


def run() -> None:
    wh = Warehouse()
    c = cfg()
    picks = wh.table("staging.draft_picks")
    ident = wh.table("staging.player_identity")
    births = wh.table("staging.birth_dates")

    ath = athletic.build(wh.table("staging.combine"), c["athletic"]["norm_max_year"])
    wh.write("marts.athletic", ath)

    seasons = production.season_metrics(
        wh.table("staging.college_player_seasons"),
        wh.table("staging.college_team_seasons"),
        wh.table("staging.college_usage"),
    )
    seasons = production.apply_opponent_adjustment(seasons)
    wh.write("marts.college_seasons", seasons)

    pros = picks.merge(
        ident[["season", "pick", "college_player_id", "match_method"]],
        left_on=["draft_year", "pick"],
        right_on=["season", "pick"],
        how="left",
    ).drop(columns="season")
    pros = pros.merge(births, left_on="pfr_player_id", right_on="pfr_id", how="left").drop(columns="pfr_id")
    career = production.career_features(seasons, pros)
    pros = pros.merge(career, on="player_key", how="left")

    ath_cols = [
        col
        for col in ath.columns
        if col.endswith(("_score", "_measured"))
        or col in ("drills_measured", "ht_in", "wt_lb", *c["athletic"]["drills"])
    ]
    a = ath.sort_values("combine_year").drop_duplicates("pfr_id", keep="last")
    pros = pros.merge(
        a[["pfr_id", "combine_year", *ath_cols]], left_on="pfr_player_id", right_on="pfr_id", how="left"
    ).drop(columns="pfr_id")
    # Only use a combine that happened before the draft.
    late = pros["combine_year"] > pros["draft_year"]
    pros.loc[late, ath_cols] = np.nan
    pros["attended_combine"] = pros["combine_year"].notna().astype(int)
    pros["power_conf"] = pros["college_conference"].isin(P5).astype(int)
    pros["log_pick"] = np.log(pros["pick"])

    pros = pros.merge(outcomes(picks, wh.table("staging.nfl_snap_shares")), on="player_key", how="left")
    wh.write("marts.prospects", pros)
    log.info("marts.prospects: %d rows, %d labeled", len(pros), pros["starter"].notna().sum())
    wh.close()
