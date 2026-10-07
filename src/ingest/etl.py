"""ETL: raw pulls -> raw.* tables -> staging.* tables. Idempotent: every table is fully replaced.

raw.*      source payloads as returned (nflverse frames, CFBD JSON flattened)
staging.*  typed, cleaned, one grain per table, keys resolved
marts.*    model-ready tables, built in src/features and src/models
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import cfg, position_group
from src.db import Warehouse
from src.ingest import cfbd_client, identity, nflverse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- raw
def pull_cfbd() -> dict[str, pd.DataFrame]:
    y = cfg()["years"]
    seasons = range(y["college_first"], y["college_last"] + 1)
    stats, games, usage, draft = [], [], [], []
    for yr in seasons:
        stats += cfbd_client.fetch("player_season_stats", year=yr)
        games += cfbd_client.fetch("games", year=yr)
        if yr >= y["usage_first"]:
            usage += cfbd_client.fetch("player_usage", year=yr)
    for yr in range(y["draft_first"], y["draft_last"] + 1):
        draft += cfbd_client.fetch("draft_picks", year=yr)
    usage_df = pd.json_normalize(usage)
    usage_df.columns = [c.replace("usage.", "usage_") for c in usage_df.columns]
    game_cols = [
        "id",
        "season",
        "week",
        "seasonType",
        "homeTeam",
        "awayTeam",
        "homeClassification",
        "awayClassification",
        "homePregameElo",
        "awayPregameElo",
        "homePoints",
        "awayPoints",
    ]
    games_df = pd.DataFrame(games)
    return {
        "cfbd_player_season_stats": pd.DataFrame(stats),
        "cfbd_games": games_df[[c for c in game_cols if c in games_df]],
        "cfbd_player_usage": usage_df,
        "cfbd_draft_picks": pd.DataFrame(draft).drop(columns=["hometownInfo"], errors="ignore"),
    }


def build_raw(wh: Warehouse) -> None:
    for name, df in {**nflverse.pull_all(), **pull_cfbd()}.items():
        wh.write(f"raw.{name}", df)
        log.info("raw.%s: %d rows", name, len(df))


# ---------------------------------------------------------------- staging
def height_inches(ht: pd.Series) -> pd.Series:
    """'6-2' -> 74; already-numeric values pass through."""

    def conv(v):
        if isinstance(v, str) and "-" in v:
            ft, inch = v.split("-")[:2]
            return int(ft) * 12 + float(inch or 0)
        return pd.to_numeric(v, errors="coerce")

    return ht.map(conv).astype(float)


def stage_draft_picks(wh: Warehouse) -> pd.DataFrame:
    p = wh.table("raw.nfl_draft_picks")
    p = p[p["season"] >= cfg()["years"]["draft_first"]].copy()
    p["pos_group"] = p["position"].map(position_group)
    p = p[p["pos_group"].notna()].copy()
    cf = wh.table("raw.cfbd_draft_picks")[
        [
            "year",
            "overall",
            "preDraftRanking",
            "preDraftPositionRanking",
            "preDraftGrade",
            "collegeConference",
            "collegeTeam",
        ]
    ]
    p = p.merge(cf, left_on=["season", "pick"], right_on=["year", "overall"], how="left").drop(
        columns=["year", "overall"]
    )
    keep = [
        "season",
        "round",
        "pick",
        "team",
        "gsis_id",
        "pfr_player_id",
        "cfb_player_id",
        "pfr_player_name",
        "position",
        "pos_group",
        "side",
        "college",
        "collegeTeam",
        "collegeConference",
        "age",
        "preDraftRanking",
        "preDraftPositionRanking",
        "preDraftGrade",
        "w_av",
        "dr_av",
        "games",
    ]
    p = p[keep].rename(
        columns={
            "season": "draft_year",
            "pfr_player_name": "player_name",
            "collegeTeam": "college_team",
            "collegeConference": "college_conference",
            "age": "draft_age",
            "preDraftRanking": "consensus_rank",
            "preDraftPositionRanking": "consensus_pos_rank",
            "preDraftGrade": "consensus_grade",
        }
    )
    p["player_key"] = p["draft_year"].astype(str) + "-" + p["pick"].astype(str)
    return p


def stage_combine(wh: Warehouse) -> pd.DataFrame:
    c = wh.table("raw.nfl_combine")
    c["ht_in"] = height_inches(c["ht"])
    c = c.rename(columns={"season": "combine_year", "pos": "combine_pos", "wt": "wt_lb"})
    return c[
        [
            "combine_year",
            "pfr_id",
            "player_name",
            "combine_pos",
            "school",
            "ht_in",
            "wt_lb",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ]
    ]


def stage_nfl_snap_shares(wh: Warehouse) -> pd.DataFrame:
    """Season-level share of team snaps on the player's primary side (offense or defense).

    Denominator = team games in the season x league-average side snaps per game, so missed games
    (injury, inactive, cut) count as zero rather than being dropped.
    """
    s = wh.table("raw.nfl_snap_counts")
    s = s[s["game_type"] == "REG"].copy()
    s["team_off"] = s["offense_snaps"] / s["offense_pct"].replace(0, np.nan)
    s["team_def"] = s["defense_snaps"] / s["defense_pct"].replace(0, np.nan)
    team_game = s.groupby(["season", "game_id", "team"])[["team_off", "team_def"]].median()
    per_game = team_game.groupby("season").mean()
    games = team_game.reset_index().groupby(["season", "team"])["game_id"].nunique().groupby("season").max()
    p = s.groupby(["season", "pfr_player_id"])[["offense_snaps", "defense_snaps"]].sum().reset_index()
    p = p.join(per_game, on="season").join(games.rename("season_games"), on="season")
    off = p["offense_snaps"] / (p["season_games"] * p["team_off"])
    dfn = p["defense_snaps"] / (p["season_games"] * p["team_def"])
    p["snap_share"] = np.maximum(off, dfn).clip(0, 1)
    return p[["season", "pfr_player_id", "offense_snaps", "defense_snaps", "snap_share"]]


def stage_college_player_seasons(wh: Warehouse) -> pd.DataFrame:
    """Wide player-season-team table of counting stats from CFBD long format."""
    s = wh.table("raw.cfbd_player_season_stats")
    s = s[s["category"].isin(["passing", "rushing", "receiving", "defensive", "interceptions"])]
    s["col"] = (s["category"] + "_" + s["statType"]).str.lower().str.replace(" ", "_")
    s["stat"] = pd.to_numeric(s["stat"], errors="coerce")
    wide = s.pivot_table(
        index=["season", "playerId", "player", "position", "team", "conference"],
        columns="col",
        values="stat",
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"playerId": "player_id", "position": "college_pos"})
    wide["player_id"] = wide["player_id"].astype(str)
    return wide


def opponent_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Per team-season schedule strength from game scores only (no player data, so no talent confounding).

    sos_z            z-score of mean opponent pregame Elo within the season
    opp_def_factor   league points allowed per game / opponents' points allowed per game
                     (> 1 means the defenses faced were stingier than average)
    opp_off_factor   opponents' points scored per game / league points scored per game
                     (> 1 means the offenses faced were better than average)
    Opponents without a rating (mostly FCS) get the season's 5th-percentile Elo and 95th-percentile
    points allowed, i.e. they are treated as weak opponents rather than average ones.
    """
    g = games[(games["seasonType"] == "regular") & games["homePoints"].notna()]
    home = pd.DataFrame(
        {
            "season": g["season"],
            "team": g["homeTeam"],
            "opp": g["awayTeam"],
            "opp_elo": g["awayPregameElo"],
            "pf": g["homePoints"],
            "pa": g["awayPoints"],
        }
    )
    away = pd.DataFrame(
        {
            "season": g["season"],
            "team": g["awayTeam"],
            "opp": g["homeTeam"],
            "opp_elo": g["homePregameElo"],
            "pf": g["awayPoints"],
            "pa": g["homePoints"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    rec = (
        long.groupby(["season", "team"]).agg(pf_pg=("pf", "mean"), pa_pg=("pa", "mean"), n=("pf", "size")).reset_index()
    )
    rec = rec[rec["n"] >= 4]  # teams with few games in the feed (FCS) don't get a reliable record
    long = long.merge(
        rec.rename(columns={"team": "opp", "pf_pg": "opp_pf_pg", "pa_pg": "opp_pa_pg"}).drop(columns="n"),
        on=["season", "opp"],
        how="left",
    )
    q = long.groupby("season")
    long["opp_elo"] = long["opp_elo"].fillna(q["opp_elo"].transform(lambda x: x.quantile(0.05)))
    long["opp_pa_pg"] = long["opp_pa_pg"].fillna(q["opp_pa_pg"].transform(lambda x: x.quantile(0.95)))
    long["opp_pf_pg"] = long["opp_pf_pg"].fillna(q["opp_pf_pg"].transform(lambda x: x.quantile(0.05)))
    sos = (
        long.groupby(["season", "team"])
        .agg(
            sos_elo=("opp_elo", "mean"),
            opp_pa_pg=("opp_pa_pg", "mean"),
            opp_pf_pg=("opp_pf_pg", "mean"),
            team_games=("opp", "size"),
        )
        .reset_index()
    )
    league = rec.groupby("season")["pf_pg"].mean().rename("league_pg")
    sos = sos.join(league, on="season")
    sos["opp_def_factor"] = (sos["league_pg"] / sos["opp_pa_pg"]).clip(0.7, 1.3)
    sos["opp_off_factor"] = (sos["opp_pf_pg"] / sos["league_pg"]).clip(0.7, 1.3)
    sos["sos_z"] = sos.groupby("season")["sos_elo"].transform(lambda x: (x - x.mean()) / x.std())
    return sos.drop(columns=["league_pg"])


def stage_team_seasons(wh: Warehouse, players: pd.DataFrame) -> pd.DataFrame:
    """Team totals (sum of player lines) plus schedule strength."""
    stat_cols = [c for c in players.columns if c.startswith(("passing_", "rushing_", "receiving_", "defensive_"))]
    totals = players.groupby(["season", "team"])[stat_cols].sum().add_prefix("team_").reset_index()
    return totals.merge(opponent_factors(wh.table("raw.cfbd_games")), on=["season", "team"], how="left")


def stage_usage(wh: Warehouse) -> pd.DataFrame:
    u = wh.table("raw.cfbd_player_usage").rename(columns={"id": "player_id"})
    u["player_id"] = u["player_id"].astype(str)
    return u[["season", "player_id", "team", "usage_overall", "usage_pass", "usage_rush"]]


def stage_birth_dates(wh: Warehouse) -> pd.DataFrame:
    r = wh.table("raw.nfl_rosters")
    r = r[r["pfr_id"].notna() & r["birth_date"].notna()]
    return r.groupby("pfr_id")["birth_date"].first().reset_index()


def build_staging(wh: Warehouse) -> None:
    picks = stage_draft_picks(wh)
    players = stage_college_player_seasons(wh)
    tables = {
        "draft_picks": picks,
        "combine": stage_combine(wh),
        "nfl_snap_shares": stage_nfl_snap_shares(wh),
        "college_player_seasons": players,
        "college_team_seasons": stage_team_seasons(wh, players),
        "college_usage": stage_usage(wh),
        "birth_dates": stage_birth_dates(wh),
    }
    stat_players = players[["player_id", "player", "team", "season"]]
    raw_picks = wh.table("raw.nfl_draft_picks")
    raw_picks = raw_picks[
        raw_picks["position"].map(position_group).notna() & (raw_picks["season"] >= cfg()["years"]["draft_first"])
    ]
    ident = identity.resolve(raw_picks, wh.table("raw.cfbd_draft_picks"), stat_players)
    tables["player_identity"] = ident
    for name, df in tables.items():
        wh.write(f"staging.{name}", df)
        log.info("staging.%s: %d rows", name, len(df))
    rate = (
        ident.assign(m=ident.match_method != "unmatched")
        .groupby(
            picks.set_index(["draft_year", "pick"])["pos_group"].reindex(list(zip(ident.season, ident.pick))).values
        )["college_player_id"]
        .apply(lambda x: x.notna().mean())
    )
    log.info("college match rate by position group:\n%s", rate.round(3).to_string())


def run() -> None:
    wh = Warehouse()
    build_raw(wh)
    build_staging(wh)
    wh.close()
