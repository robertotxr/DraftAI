import numpy as np
import pandas as pd

from src.comps.knn import MIN_SHARED, comp_features, find_comps, nan_distance
from src.config import cfg

LAG = cfg()["model"]["label_lag"]


def test_nan_distance_ignores_nans_and_counts_shared():
    a = np.array([0.0, 0.0, np.nan, 0.0])
    B = np.array([[3.0, 4.0, 9.0, np.nan], [np.nan, np.nan, 1.0, np.nan], [0.0, 0.0, 0.0, 0.0]])
    d, n = nan_distance(a, B)
    assert n.tolist() == [2, 0, 3]
    assert np.isclose(d[0], np.sqrt((9 + 16) / 2))
    assert d[2] == 0


def _pros(seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for group in ("WR", "DB"):
        feats = comp_features(group)
        for i in range(60):
            year = 2014 + i % 8
            row = dict(zip(feats, r.normal(size=len(feats))))
            row.update(
                player_key=f"{group}{i}",
                pos_group=group,
                draft_year=year,
                starter=float(i % 2) if year <= 2019 else np.nan,
            )
            rows.append(row)
    pros = pd.DataFrame(rows)
    pros.loc[0, comp_features("WR")[:4]] = np.nan  # a player with sparse data still works
    return pros


def test_find_comps_respects_time_labels_group_and_ranking():
    pros = _pros()
    k = 5
    comps = find_comps(pros, k=k, lag=LAG)
    assert len(comps)
    info = pros.set_index("player_key")
    me, other = info.loc[comps["player_key"]], info.loc[comps["comp_key"]]
    assert (comps["player_key"].to_numpy() != comps["comp_key"].to_numpy()).all()
    assert (other["draft_year"].to_numpy() <= me["draft_year"].to_numpy() - LAG).all()
    assert other["starter"].notna().all()
    assert (other["pos_group"].to_numpy() == me["pos_group"].to_numpy()).all()
    assert (comps["shared_features"] >= MIN_SHARED).all()
    for _, g in comps.groupby("player_key"):
        assert len(g) <= k
        assert g["rank"].tolist() == list(range(1, len(g) + 1))
        assert g["distance"].is_monotonic_increasing
    # earliest classes have no eligible comps at all
    early = info.index[info["draft_year"] < 2014 + LAG]
    assert not set(early) & set(comps["player_key"])
