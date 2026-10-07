import numpy as np
import pandas as pd

from src.config import cfg
from src.features.athletic import score

DRILLS = cfg()["athletic"]["drills"]
MIN_DRILLS = cfg()["athletic"]["min_drills"]


def _reference(n=200, seed=0):
    """WR-like group where heavier players are slower (weight-speed tradeoff)."""
    r = np.random.default_rng(seed)
    wt = r.uniform(180, 250, n)
    return pd.DataFrame(
        {
            "pos_group": "WR",
            "ht_in": r.uniform(68, 76, n),
            "wt_lb": wt,
            "forty": 4.3 + 0.004 * (wt - 180) + r.normal(0, 0.03, n),
            "vertical": r.normal(35, 3, n),
            "broad_jump": r.normal(120, 6, n),
            "cone": r.normal(7.0, 0.2, n),
            "shuttle": r.normal(4.3, 0.15, n),
            "bench": r.normal(15, 4, n),
        }
    )


def _player(**kw):
    row = {"pos_group": "WR", "ht_in": 72.0, "wt_lb": 215.0, "forty": 4.5, "vertical": 35.0,
           "broad_jump": 120.0, "cone": 7.0, "shuttle": 4.3, "bench": 15.0}  # fmt: skip
    row.update(kw)
    return row


def test_lower_is_better_is_flipped_and_bounded():
    ref = _reference()
    out = score(pd.DataFrame([_player(forty=4.35), _player(forty=4.80)]), reference=ref)
    assert out["forty_raw_score"][0] > out["forty_raw_score"][1]
    assert out["forty_size_score"][0] > out["forty_size_score"][1]
    # higher-is-better drill keeps its direction
    out = score(pd.DataFrame([_player(vertical=45.0), _player(vertical=25.0)]), reference=ref)
    assert out["vertical_raw_score"][0] > out["vertical_raw_score"][1]
    cols = [c for c in out if c.endswith(("_raw_score", "_size_score"))] + ["athletic_score"]
    assert out[cols].stack().between(0, 10).all()


def test_size_adjustment_rewards_heavier_player_with_same_forty():
    ref = _reference()
    out = score(pd.DataFrame([_player(wt_lb=190.0), _player(wt_lb=245.0)]), reference=ref)
    assert out["forty_size_score"][1] > out["forty_size_score"][0]
    assert out["forty_raw_score"][0] == out["forty_raw_score"][1]


def test_missing_drills_are_nan_and_flagged():
    ref = _reference()
    out = score(pd.DataFrame([_player(vertical=np.nan, bench=np.nan)]), reference=ref)
    for d in ("vertical", "bench"):
        assert out[f"{d}_measured"][0] == 0
        assert np.isnan(out[f"{d}_raw_score"][0]) and np.isnan(out[f"{d}_size_score"][0])
    assert out["forty_measured"][0] == 1
    assert out["drills_measured"][0] == len(DRILLS) - 2
    assert out["athletic_score"].notna().all()


def test_composite_requires_min_drills():
    ref = _reference()
    few = {d: np.nan for d in DRILLS[MIN_DRILLS - 1 :]}
    enough = {d: np.nan for d in DRILLS[MIN_DRILLS:]}
    out = score(pd.DataFrame([_player(**few), _player(**enough)]), reference=ref)
    assert out["drills_measured"].tolist() == [MIN_DRILLS - 1, MIN_DRILLS]
    assert np.isnan(out["athletic_score"][0])
    assert not np.isnan(out["athletic_score"][1])
    # the measured drills still get individual scores
    assert out["forty_size_score"].notna().all()
