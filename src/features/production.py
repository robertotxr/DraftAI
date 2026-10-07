"""Context-adjusted college production.

Season level (one row per college player-season-team):
  * shares of team production: receiving yards/TD share, dominator rating, scrimmage share,
    rushing share, sack/TFL share, pass-defensed share
  * passing efficiency for QBs
  * usage share of team pass plays (CFBD usage, 2013+) as the public proxy for target share;
    true targets and routes run (YPRR) are not in the free data, so YPRR is not computed
  * pressure proxy for defenders: (sacks + QB hurries) per team game. CFBD defensive season
    stats only exist from 2016, so earlier defenders have NaN here (missing, not zero)
  * opponent adjustment: volume is scaled by how many points the opponents faced allowed/scored
    relative to the league (score-based, so it is not confounded by the player's own talent)
  * age on Sept 1 of the season

Career level (one row per drafted player, using only seasons before the draft):
  final-season, best-season and career-average values, breakout age, seasons played.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import cfg

OFFENSE_ADJ = ["dominator", "scrimmage_share", "rush_yds_share", "rec_yds_pg", "rush_yds_pg", "pass_yds_pg"]
DEFENSE_ADJ = ["def_playmaking_share", "pressure_pg"]


def _div(a, b):
    return (a / b.replace(0, np.nan)).astype(float)


def season_metrics(players: pd.DataFrame, teams: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    s = players.merge(teams, on=["season", "team"], how="left")
    s = s.merge(usage.drop(columns=["team"]), on=["season", "player_id"], how="left")
    g = s["team_games"]
    col = lambda c: s[c] if c in s else pd.Series(np.nan, index=s.index)  # noqa: E731

    s["rec_yds_share"] = _div(col("receiving_yds"), col("team_receiving_yds"))
    s["rec_td_share"] = _div(col("receiving_td"), col("team_receiving_td"))
    s["dominator"] = s[["rec_yds_share", "rec_td_share"]].mean(axis=1, skipna=False)
    s["rush_yds_share"] = _div(col("rushing_yds"), col("team_rushing_yds"))
    s["scrimmage_share"] = _div(
        col("rushing_yds").fillna(0) + col("receiving_yds").fillna(0),
        col("team_rushing_yds") + col("team_receiving_yds"),
    )
    s["rec_yds_pg"] = _div(col("receiving_yds"), g)
    s["rush_yds_pg"] = _div(col("rushing_yds"), g)
    s["yds_per_rec"] = _div(col("receiving_yds"), col("receiving_rec"))
    s["yds_per_carry"] = _div(col("rushing_yds"), col("rushing_car"))
    s["target_share_proxy"] = col("usage_pass")

    s["pass_yds_pg"] = _div(col("passing_yds"), g)
    s["pass_ypa"] = _div(col("passing_yds"), col("passing_att"))
    s["pass_td_rate"] = _div(col("passing_td"), col("passing_att"))
    s["pass_int_rate"] = _div(col("passing_int"), col("passing_att"))
    s["pass_cmp_pct"] = _div(col("passing_completions"), col("passing_att"))
    s["pass_att_share"] = _div(col("passing_att"), col("team_passing_att"))

    s["sack_share"] = _div(col("defensive_sacks"), col("team_defensive_sacks"))
    s["tfl_share"] = _div(col("defensive_tfl"), col("team_defensive_tfl"))
    s["def_playmaking_share"] = _div(
        col("defensive_sacks").fillna(0) + col("defensive_tfl"), col("team_defensive_sacks") + col("team_defensive_tfl")
    )
    s["pressure_pg"] = _div(col("defensive_sacks").fillna(0) + col("defensive_qb_hur"), g)
    s["pd_share"] = _div(col("defensive_pd"), col("team_defensive_pd"))
    s["tackle_share"] = _div(col("defensive_tot"), col("team_defensive_tot"))
    s["int_pg"] = _div(col("interceptions_int"), g)
    return s


def apply_opponent_adjustment(seasons: pd.DataFrame) -> pd.DataFrame:
    """Scale per-game volume by the quality of opponents faced.

    Offensive volume is multiplied by `opp_def_factor` (stingier defenses -> credit up) and defensive
    volume by `opp_off_factor`. Within-team shares (dominator, playmaking share) are already relative to
    teammates facing the same schedule, so they get the same factor only as a separate `_adj` column;
    the raw share is kept alongside.
    """
    s = seasons.copy()
    off = s["opp_def_factor"].fillna(1.0)
    dfn = s["opp_off_factor"].fillna(1.0)
    for m in OFFENSE_ADJ:
        s[f"{m}_adj"] = s[m] * off
    for m in DEFENSE_ADJ:
        s[f"{m}_adj"] = s[m] * dfn
    return s


def season_age(season: pd.Series, birth_date: pd.Series) -> pd.Series:
    sept1 = pd.to_datetime(season.astype(int).astype(str) + "-09-01")
    return (sept1 - pd.to_datetime(birth_date)).dt.days / 365.25


def breakout_age(career: pd.DataFrame, metric: str, threshold: float) -> float:
    hit = career.loc[career[metric] >= threshold, "age"]
    return float(hit.min()) if len(hit) else np.nan


CAREER_METRICS = [
    "dominator_adj",
    "scrimmage_share_adj",
    "rush_yds_share_adj",
    "rec_yds_pg_adj",
    "rush_yds_pg_adj",
    "yds_per_rec",
    "yds_per_carry",
    "target_share_proxy",
    "pass_yds_pg_adj",
    "pass_ypa",
    "pass_td_rate",
    "pass_int_rate",
    "pass_cmp_pct",
    "def_playmaking_share_adj",
    "pressure_pg_adj",
    "sack_share",
    "pd_share",
    "tackle_share",
    "int_pg",
    "sos_z",
]


def career_features(seasons: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """One row per prospect from college seasons strictly before the draft year.

    prospects: player_key, college_player_id, draft_year, birth_date (may be NaT), draft_age
    """
    p = cfg()["production"]
    pro = prospects.dropna(subset=["college_player_id"])[
        ["player_key", "college_player_id", "draft_year", "birth_date", "draft_age"]
    ]
    s = seasons.merge(pro, left_on="player_id", right_on="college_player_id")
    s = s[s["season"] < s["draft_year"]].copy()
    # Approximate birth date from integer draft age when the roster birth date is unknown.
    approx_birth = pd.to_datetime(
        (s["draft_year"] - s["draft_age"]).astype("Int64").astype(str) + "-10-25", errors="coerce"
    )
    s["age"] = season_age(s["season"], s["birth_date"].fillna(approx_birth))
    # A player can have two rows in one season (transfer mid-year is rare, but team splits exist).
    s = s.sort_values(["player_key", "season", "dominator"]).groupby(["player_key", "season"]).tail(1)

    rows = []
    for key, c in s.groupby("player_key"):
        c = c.sort_values("season")
        last = c.iloc[-1]
        r = {
            "player_key": key,
            "college_seasons": len(c),
            "final_season_age": last["age"],
            "final_team": last["team"],
            "final_conference": last["conference"],
        }
        for m in CAREER_METRICS:
            r[f"{m}_final"] = last[m]
            r[f"{m}_best"] = c[m].max()
            r[f"{m}_career"] = c[m].mean()
        r["breakout_age_rec"] = breakout_age(c, "dominator", p["breakout_dominator"])
        r["breakout_age_scrim"] = breakout_age(c, "scrimmage_share", p["breakout_dominator"])
        r["breakout_age_def"] = breakout_age(c, "def_playmaking_share", p["breakout_def_share"])
        # "Never broke out" is information, distinct from "no data": flag it explicitly.
        for name, metric in [("rec", "dominator"), ("scrim", "scrimmage_share"), ("def", "def_playmaking_share")]:
            has = c[metric].notna().any()
            r[f"never_broke_out_{name}"] = float(np.isnan(r[f"breakout_age_{name}"])) if has else np.nan
        r["has_def_stats"] = int(c["defensive_tot"].notna().any()) if "defensive_tot" in c else 0
        rows.append(r)
    return pd.DataFrame(rows)
