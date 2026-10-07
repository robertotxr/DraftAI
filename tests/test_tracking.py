"""Tracking metric tests on a tiny synthetic dataset; real BDB data is never touched."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.config import cfg
from src.tracking import data, metric, run, validate
from src.tracking.data import FIELD_WIDTH, KEYS

N_FRAMES = 30


def synthetic(n_games: int = 6, n_plays: int = 8, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Carrier runs +x at 5 yd/s while 11 defenders converge; defender 100 always makes the tackle."""
    rng = np.random.default_rng(seed)
    rows, plays, tackles, games = [], [], [], []
    for g in range(1, n_games + 1):
        games.append({"gameId": g, "week": 1 + (g % 2)})
        for p in range(1, n_plays + 1):
            left = (g + p) % 2 == 0
            start = np.array([30.0, 26.0])
            end = start + [14.5, 0]
            plays.append(
                {"gameId": g, "playId": p, "ballCarrierId": 200, "possessionTeam": "OFF", "defensiveTeam": "DEF"}
            )
            tackles.append({"gameId": g, "playId": p, "nflId": 100, "tackle": 1, "assist": 0, "pffMissedTackle": 0})
            tackles.append(
                {"gameId": g, "playId": p, "nflId": 101, "tackle": 0, "assist": int(p % 2), "pffMissedTackle": 0}
            )
            tackles.append(
                {"gameId": g, "playId": p, "nflId": 102, "tackle": 0, "assist": 0, "pffMissedTackle": int(p % 3 == 0)}
            )
            actors = {200: (start, end, "OFF")}
            for i in range(10):  # blockers ahead of the carrier
                a = start + [3 + i, rng.uniform(-8, 8)]
                actors[201 + i] = (a, a + [14.5, 0], "OFF")
            for i in range(11):
                a = start + [rng.uniform(4, 14), rng.uniform(-14, 14)]
                target = end if i == 0 else end + rng.normal(0, 2 + i / 2, 2)
                actors[100 + i] = (a, target, "DEF")
            actors[None] = (start, end, "football")
            for f in range(N_FRAMES):
                for nfl, (a, b, club) in actors.items():
                    pos = a + (b - a) * f / (N_FRAMES - 1)
                    v = (b - a) / ((N_FRAMES - 1) / 10)
                    x, y, d = pos[0], pos[1], np.degrees(np.arctan2(v[0], v[1])) % 360
                    if left:  # raw data for left-moving plays is mirrored
                        x, y, d = 120 - x, FIELD_WIDTH - y, (d + 180) % 360
                    rows.append({
                        "gameId": g, "playId": p, "nflId": nfl, "displayName": f"P{nfl}", "frameId": f + 1,
                        "club": club, "playDirection": "left" if left else "right", "x": x, "y": y,
                        "s": float(np.hypot(*v)), "a": 1.0,
                        "dir": d, "o": d, "event": "handoff" if f == 0 else "tackle" if f == N_FRAMES - 1 else None,
                    })  # fmt: skip
    players = pd.DataFrame({"nflId": range(100, 111), "displayName": [f"D{i}" for i in range(11)], "position": "LB"})
    return {
        "tracking": pd.DataFrame(rows), "plays": pd.DataFrame(plays), "tackles": pd.DataFrame(tackles),
        "games": pd.DataFrame(games), "players": players,
    }  # fmt: skip


@pytest.fixture
def paths(tmp_path, monkeypatch):
    def fake(key: str):
        p = tmp_path / key
        p.mkdir(parents=True, exist_ok=True)
        return p

    for mod in (data, run, validate):
        monkeypatch.setattr(mod, "path", fake)
    monkeypatch.setitem(cfg()["tracking"], "min_opportunities", 4)
    monkeypatch.setitem(cfg()["tracking"]["lgbm"], "min_child_samples", 5)
    return fake


def test_standardize_flips_left_plays():
    df = pd.DataFrame({"playDirection": ["left", "right"], "x": [100.0, 20.0], "y": [10.0, 10.0], "s": [5.0, 5.0],
                       "dir": [270.0, 90.0], "o": [270.0, 90.0]})  # fmt: skip
    out = data.standardize(df)
    assert out["x"].tolist() == [20.0, 20.0]
    assert out["y"].iloc[0] == pytest.approx(FIELD_WIDTH - 10)
    assert out["dir"].tolist() == [90.0, 90.0]
    assert out["vx"].tolist() == pytest.approx([5.0, 5.0])
    assert out["vy"].abs().max() < 1e-9


def _geom(**kw):
    base = dict(x=0.0, y=0.0, s=5.0, a=1.0, vx=5.0, vy=0.0, cx=10.0, cy=0.0, cs=0.0, cvx=0.0, cvy=0.0)
    return metric.add_features(pd.DataFrame([{**base, **kw}])).iloc[0]


def test_features_on_hand_built_geometry():
    r = _geom()  # defender runs straight at a stationary carrier
    assert r["dist"] == pytest.approx(10) and r["closing_speed"] == pytest.approx(5)
    assert r["pursuit_angle"] == pytest.approx(0, abs=1e-3)
    r = _geom(cs=5.0, cvy=5.0)  # carrier moving up the field: aim point is (10, 5) after 1 s
    assert r["pursuit_angle"] == pytest.approx(np.degrees(np.arctan(0.5)), abs=1e-3)
    assert r["closing_speed"] == pytest.approx(5) and r["rel_speed"] == pytest.approx(np.hypot(5, 5))
    r = _geom(vx=-5.0)  # running away
    assert r["closing_speed"] == pytest.approx(-5) and r["pursuit_angle"] == pytest.approx(180)
    assert _geom(cy=3.0)["sideline_dist"] == pytest.approx(3.0)


def _frames(dists: dict[int, list[float]], made: dict[int, int]) -> pd.DataFrame:
    k = {"gameId": 1, "playId": 1}
    return pd.DataFrame(
        [
            {**k, "nflId": n, "frameId": i + 1, "dist": v, "made": made[n]}
            for n, ds in dists.items()
            for i, v in enumerate(ds)
        ]
    )


def test_decision_point_precedes_contact_and_ignores_later_frames():
    far = [20.0] * 6
    dists = {1: [*far, 8.0, 4.9, 1.0, 0.0], 2: [3.0, 2.0, 1.0, 0.0], 3: [*far, 15.0, 12.0]}
    frames = _frames(dists, {1: 1, 2: 0, 3: 0})
    dec = metric.decision_points(frames).set_index("nflId")
    assert dec.index.tolist() == [1, 2]  # defender 3 never got within the opportunity radius
    assert dec.loc[1, "frameId"] == 8 and dec.loc[1, "dist"] == pytest.approx(
        4.9
    )  # first frame inside 5 yd, pre-contact
    assert dec.loc[2, "frameId"] == 1  # already inside when the carrier is defined
    assert dec.loc[1, "label"] == 1 and dec.loc[2, "label"] == 0
    dists[1] = [*dists[1][:8], 30.0, 30.0]  # rewrite everything after the decision
    again = metric.decision_points(_frames(dists, {1: 1, 2: 0, 3: 0})).set_index("nflId")
    assert again.loc[1, ["frameId", "dist", "label"]].tolist() == dec.loc[1, ["frameId", "dist", "label"]].tolist()


def test_made_requires_credit():
    k = {"gameId": 1, "playId": 1}
    tackles = pd.DataFrame([{**k, "nflId": 1, "tackle": 0, "assist": 1}, {**k, "nflId": 2, "tackle": 0, "assist": 0}])
    out = metric.add_made(_frames({1: [1.0], 2: [1.0]}, {1: 0, 2: 0}).drop(columns="made"), tackles)
    assert out.set_index("nflId")["made"].to_dict() == {1: 1, 2: 0}


def test_standing_defender_has_no_pursuit_angle():
    r = _geom(s=0.0, vx=0.0)
    assert np.isnan(r["pursuit_cos"]) and np.isnan(r["pursuit_angle"])


def test_group_folds_never_share_a_game():
    groups = pd.Series(np.repeat(np.arange(7), 13))
    seen = []
    for tr, te in metric.group_folds(groups, 3):
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
        seen.extend(te)
    assert sorted(seen) == list(range(len(groups)))


def test_run_without_data_warns_and_returns_none(paths, caplog):
    with caplog.at_level(logging.WARNING):
        assert run.run() is None
    assert data.DATA_URL in caplog.text
    with pytest.raises(FileNotFoundError, match="README"):
        data.check_data()


def test_end_to_end_on_synthetic_data(paths, tmp_path):
    s = synthetic()
    raw = paths("bdb")
    for name in ("plays", "tackles", "players", "games"):
        s[name].to_csv(raw / f"{name}.csv", index=False)
    track = s["tracking"].merge(s["games"], on="gameId")
    for w, g in track.groupby("week"):
        g.drop(columns="week").to_csv(raw / f"tracking_week_{w}.csv", index=False)
    report = run.run()
    out = tmp_path / "artifacts" / "tracking"
    assert report["decision_model"]["brier"] < 0.25
    assert json.loads((out / "validation.json").read_text())["n_decisions"] == report["n_decisions"]
    lb = pd.read_parquet(out / "leaderboard.parquet")
    assert lb["toe"].abs().sum() > 0 and lb.loc[lb["nflId"] == 100, "toe"].iloc[0] > 0  # the tackler beats expectation
    frames = pd.read_parquet(out / "animation_frames.parquet")
    assert {"carrier", "offense", "defense", "ball"} <= set(frames["role"])
    g, p = frames[KEYS].iloc[0]
    from src.tracking.animate import animate_play

    fig = animate_play(frames, g, p)
    assert len(fig.frames) == N_FRAMES and len(fig.data) == 4
    assert (tmp_path / "figures" / "tracking_calibration.png").exists()
