import numpy as np
import pandas as pd

from src.config import cfg
from src.features import production as P

BREAKOUT = cfg()["production"]["breakout_dominator"]


def test_season_metrics_shares_and_zero_division():
    players = pd.DataFrame(
        {
            "season": 2020,
            "team": ["A", "B"],
            "player_id": ["1", "2"],
            "receiving_yds": [500.0, 300.0],
            "receiving_td": [5.0, 2.0],
        }
    )
    teams = pd.DataFrame(
        {
            "season": 2020,
            "team": ["A", "B"],
            "team_games": [10, 10],
            "team_receiving_yds": [2000.0, 1500.0],
            "team_receiving_td": [10.0, 0.0],  # team B: zero TDs -> undefined share
        }
    )
    usage = pd.DataFrame({"season": [2020], "player_id": ["1"], "team": ["A"], "usage_pass": [0.3]})
    s = P.season_metrics(players, teams, usage).set_index("player_id")
    assert s.loc["1", "rec_yds_share"] == 0.25
    assert s.loc["1", "rec_td_share"] == 0.5
    assert s.loc["1", "dominator"] == (0.25 + 0.5) / 2
    assert np.isnan(s.loc["2", "rec_td_share"]) and np.isnan(s.loc["2", "dominator"])
    assert not np.isinf(s.select_dtypes("number").to_numpy(float)).any()


def test_opponent_adjustment_multiplies_by_the_right_factor():
    cols = P.OFFENSE_ADJ + P.DEFENSE_ADJ
    df = pd.DataFrame({c: [10.0, 10.0] for c in cols})
    df["opp_def_factor"] = [1.2, np.nan]
    df["opp_off_factor"] = [0.8, np.nan]
    out = P.apply_opponent_adjustment(df)
    assert np.allclose(out[[f"{m}_adj" for m in P.OFFENSE_ADJ]].iloc[0], 12.0)
    assert np.allclose(out[[f"{m}_adj" for m in P.DEFENSE_ADJ]].iloc[0], 8.0)
    assert np.allclose(out[[f"{m}_adj" for m in cols]].iloc[1], 10.0)  # missing factor -> neutral


def _seasons(rows):
    df = pd.DataFrame(rows)
    df["conference"] = "SEC"
    for c in P.CAREER_METRICS + ["dominator", "scrimmage_share", "def_playmaking_share"]:
        if c not in df:
            df[c] = np.nan
    return df


def test_career_features_use_only_pre_draft_seasons_and_breakout_flags():
    rows = [
        {"player_id": "1", "team": "A", "season": 2019, "dominator": 0.10, "dominator_adj": 0.10},
        {"player_id": "1", "team": "A", "season": 2020, "dominator": 0.25, "dominator_adj": 0.25},
        {"player_id": "1", "team": "A", "season": 2021, "dominator": 0.30, "dominator_adj": 0.30},
        # post-draft season with absurd numbers: must be ignored everywhere
        {"player_id": "1", "team": "A", "season": 2022, "dominator": 9.0, "dominator_adj": 9.0},
        {"player_id": "2", "team": "B", "season": 2021, "dominator": 0.05, "dominator_adj": 0.05},
    ]
    pros = pd.DataFrame(
        {
            "player_key": ["k1", "k2", "k3"],
            "college_player_id": ["1", "2", None],
            "draft_year": [2022, 2022, 2022],
            "birth_date": pd.to_datetime(["2000-01-01", None, "2000-01-01"]),
            "draft_age": [22.0, 21.0, 22.0],
        }
    )
    out = P.career_features(_seasons(rows), pros).set_index("player_key")
    assert set(out.index) == {"k1", "k2"}  # no college id -> no row
    a = out.loc["k1"]
    assert a["college_seasons"] == 3
    assert a["dominator_adj_best"] == 0.30 and a["dominator_adj_final"] == 0.30
    assert np.isclose(a["dominator_adj_career"], (0.10 + 0.25 + 0.30) / 3)
    expected_age = (pd.Timestamp("2020-09-01") - pd.Timestamp("2000-01-01")).days / 365.25
    assert np.isclose(a["breakout_age_rec"], expected_age)
    assert a["never_broke_out_rec"] == 0
    assert 0.25 >= BREAKOUT > 0.10

    b = out.loc["k2"]  # has data, never reached the threshold
    assert np.isnan(b["breakout_age_rec"]) and b["never_broke_out_rec"] == 1
    # no scrimmage / defensive data at all -> unknown, not "never broke out"
    for name in ("scrim", "def"):
        assert np.isnan(b[f"never_broke_out_{name}"])
