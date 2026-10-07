import numpy as np
import pandas as pd
import pytest

from src.config import cfg
from src.features.build import outcomes
from src.models import train
from src.models.features import FEATURE_SETS, POST_DRAFT, assert_no_leakage

T = cfg()["target"]
LAG = cfg()["model"]["label_lag"]


def _split(train_years=(2015, 2016), test_year=2020):
    tr = pd.DataFrame({"draft_year": list(train_years), "label_known_from_draft": [y + 3 for y in train_years]})
    return tr, pd.DataFrame({"draft_year": [test_year]})


def test_valid_split_passes():
    tr, te = _split()
    assert_no_leakage(tr, te, FEATURE_SETS["full"] + ["pos_group"])


@pytest.mark.parametrize("bad", [POST_DRAFT[0], "starter", "snap_share_3yr", "some_unknown_col"])
def test_bad_features_raise(bad):
    tr, te = _split()
    with pytest.raises(AssertionError, match="Disallowed"):
        assert_no_leakage(tr, te, ["log_pick", bad])


def test_unobservable_labels_raise():
    tr, te = _split(train_years=(2015, 2018))  # 2018 labels known from 2021 > 2020
    with pytest.raises(AssertionError, match="not yet observable"):
        assert_no_leakage(tr, te, ["log_pick"])


def test_training_rows_from_test_class_raise():
    tr, te = _split()
    tr.loc[0, ["draft_year", "label_known_from_draft"]] = [2020, 2020]  # labels fine, but same class
    with pytest.raises(AssertionError, match="test class"):
        assert_no_leakage(tr, te, ["log_pick"])


def test_feature_sets_are_clean():
    for name, feats in FEATURE_SETS.items():
        assert not set(feats) & set(POST_DRAFT), name
        for f in feats:
            assert not any(s in f for s in ("snap_share", "starter", "w_av", "dr_av", "games")), (name, f)


def test_outcomes_missing_unobserved_and_label_horizon():
    last = cfg()["years"]["last_nfl_season"]
    picks = pd.DataFrame(
        {
            "player_key": ["old", "new"],
            "pfr_player_id": ["o", "n"],
            "draft_year": [2020, last - 1],
        }
    )
    shares = pd.DataFrame(
        {"pfr_player_id": ["o", "o", "n"], "season": [2021, 2022, last - 1], "snap_share": [0.8, 0.7, 0.9]}
    )
    out = outcomes(picks, shares).set_index("player_key")
    assert out.loc["old", "snap_share_y1"] == 0  # no snaps in 2020 -> 0, not NaN
    assert np.isclose(out.loc["old", "snap_share_3yr"], 0.5) and out.loc["old", "starter"] == 1
    assert out.loc["new", "snap_share_y1"] == 0.9
    assert out.loc["new", "snap_share_y2"] == 0  # season == last_nfl_season, observed
    assert np.isnan(out.loc["new", "snap_share_y3"])  # season after last_nfl_season
    assert np.isnan(out.loc["new", "snap_share_3yr"]) and np.isnan(out.loc["new", "starter"])
    assert (out["label_known_from_draft"] == [2020 + T["horizon_seasons"], last - 1 + T["horizon_seasons"]]).all()


def test_walk_forward_trains_only_on_classes_known_before_draft(monkeypatch):
    seen = {}

    def fake_fit_predict(tr, te, spec, temporal_check=True):
        seen.setdefault(int(te["draft_year"].iloc[0]), []).append(tr)
        return np.zeros(len(te))

    monkeypatch.setattr(train, "fit_predict", fake_fit_predict)
    monkeypatch.setattr(train, "fit_quantiles", lambda tr, feats: None)
    monkeypatch.setattr(train, "predict_quantiles", lambda m, df, feats: np.zeros((len(df), 3)))
    rows = [(f"{y}-{i}", y, float(i % 2) if y <= 2018 else np.nan, "WR") for y in range(2013, 2022) for i in range(4)]
    pros = pd.DataFrame(rows, columns=["player_key", "draft_year", "starter", "pos_group"])
    years = [2017, 2019, 2021]
    train.walk_forward(pros, years)
    assert set(seen) == set(years)
    for year, trains in seen.items():
        assert len(trains) == len(train.SPECS)
        for tr in trains:
            assert len(tr) and tr["draft_year"].max() <= year - LAG
            assert tr["starter"].notna().all()
