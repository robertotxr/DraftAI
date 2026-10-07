import numpy as np
import pandas as pd

from src.ingest.etl import stage_undrafted
from src.models.train import interval_stratum, widen


def _combine(name, year, pfr, draft_ovr=np.nan, pos="CB"):
    return {"combine_year": year, "pfr_id": pfr, "combine_key": pfr or f"{name}|{year}|X", "cfb_id": None,
            "draft_ovr": draft_ovr, "player_name": name, "combine_pos": pos, "school": "X"}  # fmt: skip


def test_undrafted_excludes_drafted_players_missing_draft_info():
    combine = pd.DataFrame([
        _combine("J.C. Jackson", 2018, "JackJ00"),
        _combine("Drafted NoOvr", 2018, "DrafN00"),  # nflverse left draft_ovr empty, but he was picked
        _combine("Name Match Jr.", 2018, None),  # no PFR id, matches a pick by name + year
        _combine("Real Pick", 2018, "RealP00", draft_ovr=10),
        _combine("Kicker Guy", 2018, "KickG00", pos="K"),
    ])  # fmt: skip
    picks = pd.DataFrame({"player_name": ["Drafted NoOvr", "Name Match", "Real Pick"], "draft_year": [2018] * 3,
                          "pfr_player_id": ["DrafN00", None, "RealP00"]})  # fmt: skip
    u = stage_undrafted(combine, picks)
    assert u["player_name"].tolist() == ["J.C. Jackson"]
    assert u["pick"].isna().all() and (u["undrafted"] == 1).all()
    assert u["player_key"].iloc[0] == "2018-UDFA-JackJ00"


def test_widening_is_per_draft_capital_stratum():
    df = pd.DataFrame({"round": [1, 2, 5, np.nan], "q10": [0.5] * 4, "q50": [0.6] * 4, "q90": [0.7] * 4})
    assert interval_stratum(df).tolist() == ["R1", "R2-3", "R4-7", "UDFA"]
    out = widen(df, {"R1": 0.2, "R2-3": 0.1, "R4-7": 0.0, "UDFA": 0.6})
    assert np.allclose(out["q10"], [0.3, 0.4, 0.5, 0.0])  # clipped at 0
    assert np.allclose(out["q90"], [0.9, 0.8, 0.7, 1.0])  # clipped at 1
