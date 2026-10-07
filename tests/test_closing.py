"""Closing Over Expected tests on small synthetic frames; real BDB data is never touched."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.config import cfg
from src.tracking import closing, closing_validate, data

W = data.FIELD_WIDTH
SUPP_ROW = {"possession_team": "OFF", "defensive_team": "DEF", "pass_result": "C", "pass_length": 10,
            "team_coverage_man_zone": "MAN_COVERAGE", "expected_points_added": 0.1,
            "play_nullified_by_penalty": "N", "play_description": "pass", "week": 1}  # fmt: skip


def play_frames(g=1, p=1, direction="right", d_end=(30.0, 30.0), n_in=3, n_out=4, rng=None):
    """Receiver at (40, 20), defender 11 at (30, 25) moving +x at 5 yd/s, ball lands at (50, 20). Two pre-throw frames."""
    rows = []
    roles = {1: ("Targeted Receiver", "WR", False), 2: ("Passer", "QB", False), 11: ("Defensive Coverage", "CB", True)}
    pos = {1: (40.0, 20.0), 2: (20.0, 26.0), 11: (30.0, 25.0)}
    for f in range(1, n_in + 1):
        for nid, (role, position, pred) in roles.items():
            x, y = pos[nid]
            if f < n_in:  # earlier frames differ from the throw frame
                x -= 2 * (n_in - f)
            rows.append(dict(game_id=g, play_id=p, player_to_predict=pred or nid == 1, nfl_id=nid, frame_id=f,
                             play_direction=direction, player_name=f"P{nid}", player_height="6-0", player_weight=200,
                             player_birth_date="1999-01-01", player_position=position, player_side="x",
                             player_role=role, x=x, y=y, s=5.0 if nid == 11 else 1.0, a=1.0, dir=90.0, o=90.0,
                             num_frames_output=n_out, ball_land_x=50.0, ball_land_y=20.0))  # fmt: skip
    inp = pd.DataFrame(rows)
    if direction == "left":  # same play mirrored: left-moving data is the 180 degree rotation
        inp = inp.assign(x=data.FIELD_LENGTH - inp["x"], y=W - inp["y"], dir=(inp["dir"] + 180) % 360,
                         o=(inp["o"] + 180) % 360, ball_land_x=data.FIELD_LENGTH - 50.0, ball_land_y=W - 20.0)  # fmt: skip
    out = pd.DataFrame([dict(game_id=g, play_id=p, nfl_id=n, frame_id=f, x=x, y=y)
                        for f in range(1, n_out + 1) for n, (x, y) in {1: (50.0, 20.0), 11: (d_end[0] + f * 0, d_end[1])}.items()])  # fmt: skip
    if direction == "left":
        out = out.assign(x=data.FIELD_LENGTH - out["x"], y=W - out["y"])
    return inp, out


def supp_for(*keys):
    return pd.DataFrame([{"game_id": g, "play_id": p, **SUPP_ROW} for g, p in keys])


def test_standardize_flips_left_plays_to_the_same_geometry():
    right, _ = play_frames(direction="right")
    left, _ = play_frames(direction="left")
    a = closing.build_units(closing.standardize(right), supp_for((1, 1)))
    b = closing.build_units(closing.standardize(left), supp_for((1, 1)))
    assert a[closing.FEATURES].iloc[0].to_dict() == pytest.approx(b[closing.FEATURES].iloc[0].to_dict(), nan_ok=True)
    s = closing.standardize(left)
    assert s["ball_land_x"].iloc[0] == pytest.approx(50.0) and s["ball_land_y"].iloc[0] == pytest.approx(20.0)


def test_features_use_only_the_throw_frame():
    inp, out = play_frames()
    u = closing.build_units(closing.standardize(inp), supp_for((1, 1)))
    r = u.iloc[0]
    assert r["start_dist"] == pytest.approx(np.hypot(10, 5)) and r["dist_land"] == pytest.approx(np.hypot(20, 5))
    assert r["air_time"] == pytest.approx(0.4) and r["def_speed"] == pytest.approx(5.0) and r["man"] == 1.0
    assert r["def_vel_land"] == pytest.approx(5 * 20 / np.hypot(20, 5))
    early = inp[inp["frame_id"] < 3].assign(x=lambda t: t["x"] + 7.0)  # rewrite pre-throw history
    inp2 = pd.concat([early, inp[inp["frame_id"] == 3]])
    u2 = closing.build_units(closing.standardize(inp2), supp_for((1, 1)))
    pd.testing.assert_frame_equal(u[closing.FEATURES], u2[closing.FEATURES])
    # post-throw positions only feed the target
    t1 = closing.add_target(u, out)
    t2 = closing.add_target(u, out.assign(x=out["x"] + 3.0 * (out["nfl_id"] == 11)))
    pd.testing.assert_frame_equal(t1[closing.FEATURES], t2[closing.FEATURES])
    assert t1["end_dist"].iloc[0] != pytest.approx(t2["end_dist"].iloc[0])


def test_target_is_distance_at_the_final_output_frame():
    inp, out = play_frames(d_end=(46.0, 20.0), n_out=4)
    out.loc[(out["nfl_id"] == 11) & (out["frame_id"] < 4), ["x", "y"]] = [0.0, 0.0]  # earlier frames are irrelevant
    u = closing.add_target(closing.build_units(closing.standardize(inp), supp_for((1, 1))), out)
    assert u["end_dist"].iloc[0] == pytest.approx(4.0)
    assert u["actual_closing"].iloc[0] == pytest.approx(np.hypot(10, 5) - 4.0)
    left_inp, left_out = play_frames(direction="left", d_end=(46.0, 20.0), n_out=4)
    left_out.loc[(left_out["nfl_id"] == 11) & (left_out["frame_id"] < 4), ["x", "y"]] = [0.0, 0.0]
    ul = closing.add_target(closing.build_units(closing.standardize(left_inp), supp_for((1, 1))), left_out)
    assert ul["end_dist"].iloc[0] == pytest.approx(4.0)


def test_coe_sign_positive_when_defender_ends_closer_than_expected(monkeypatch):
    monkeypatch.setitem(cfg()["closing"], "n_splits", 3)
    monkeypatch.setitem(cfg()["closing"]["lgbm"], "min_child_samples", 5)
    monkeypatch.setitem(cfg()["closing"]["lgbm"], "n_estimators", 40)
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame(rng.normal(size=(n, len(closing.FEATURES))), columns=closing.FEATURES)
    df["start_dist"] = rng.uniform(2, 15, n)
    df["air_time"] = rng.uniform(0.5, 3, n)
    df["end_dist"] = 0.6 * df["start_dist"] + rng.normal(0, 0.3, n)
    df["game_id"] = np.repeat(np.arange(12), 20)
    df.loc[0, "end_dist"] = 0.0  # a defender who ends on top of the receiver
    df.loc[1, "end_dist"] = df.loc[1, "start_dist"] * 3  # one who gets beaten badly
    r = closing.fit_oof(df)
    assert r["coe"].iloc[0] > 0 > r["coe"].iloc[1]
    np.testing.assert_allclose(r["coe"], r["expected_end"] - r["end_dist"])
    assert np.sqrt(((r["end_dist"] - r["expected_end"]) ** 2).mean()) < np.sqrt(
        ((r["end_dist"] - r["start_dist"]) ** 2).mean()
    )


def test_run_without_data_warns_and_returns_none(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(closing, "path", lambda key: tmp_path / key)
    with caplog.at_level(logging.WARNING):
        assert closing.run() is None
    assert "make bdb" in caplog.text


def test_match_prospects_by_birthdate_then_unique_position():
    players = pd.DataFrame({
        "nfl_id": [1, 2, 3, 4], "player_name": ["Jalen Ramsey Jr.", "Sam Smith", "Sam Smith", "Al Unique"],
        "player_birth_date": ["1994-10-24", "1990-01-01", "1991-02-02", "1995-05-05"],
        "position": ["CB", "CB", "CB", "FS"],
    })  # fmt: skip
    pros = pd.DataFrame({
        "player_key": ["a", "b", "c", "d"], "player_name": ["Jalen Ramsey", "Sam Smith", "Sam Smith", "Al Unique"],
        "birth_date": pd.to_datetime(["1994-10-24", "1990-01-01", None, None]), "pos_group": ["DB"] * 4,
        "draft_year": [2016, 2012, 2013, 2017],
    })  # fmt: skip
    m = closing_validate.match_prospects(players, pros).set_index("nfl_id")
    assert m.loc[1, "player_key"] == "a" and m.loc[1, "match_method"] == "name_birthdate"
    assert m.loc[2, "player_key"] == "b"
    assert 3 not in m.index  # ambiguous name + position, no birth date match
    assert m.loc[4, "match_method"] == "name_position_group"


def synthetic_files(root, n_games=10, plays_per_game=4, seed=0):
    rng = np.random.default_rng(seed)
    supp = []
    for week in (1, 2):
        inps, outs = [], []
        for gi in range(n_games // 2):
            g = week * 100 + gi
            for p in range(1, plays_per_game + 1):
                inp, out = play_frames(g, p, "left" if p % 2 else "right", d_end=(rng.uniform(35, 50), rng.uniform(15, 25)),
                                       n_out=int(rng.integers(5, 12)))  # fmt: skip
                inp["nfl_id"] += 100 * (gi % 3)  # a few repeating defenders across games
                out["nfl_id"] += 100 * (gi % 3)
                inps.append(inp)
                outs.append(out)
                supp.append({"game_id": g, "play_id": p, **SUPP_ROW, "week": week,
                             "pass_result": rng.choice(["C", "I", "IN"]), "expected_points_added": rng.normal()})  # fmt: skip
        pd.concat(inps).to_csv(root / f"input_2023_w{week:02d}.csv", index=False)
        pd.concat(outs).to_csv(root / f"output_2023_w{week:02d}.csv", index=False)
    pd.DataFrame(supp).to_csv(root / "supplementary_data.csv", index=False)


def test_end_to_end_on_synthetic_files(tmp_path, monkeypatch):
    monkeypatch.setattr(closing, "path", lambda key: tmp_path / key)
    monkeypatch.setitem(cfg()["closing"], "min_plays", 2)
    monkeypatch.setitem(cfg()["closing"], "n_splits", 3)
    monkeypatch.setitem(cfg()["closing"]["lgbm"], "min_child_samples", 5)
    (tmp_path / "bdb26" / "train").mkdir(parents=True)
    synthetic_files(tmp_path / "bdb26" / "train")
    (tmp_path / "bdb26" / "supplementary_data.csv").write_text(
        (tmp_path / "bdb26/train/supplementary_data.csv").read_text()
    )
    report = closing.run(link=False)
    out = tmp_path / "artifacts" / "tracking_closing"
    assert report["n_defender_plays"] == 40 and report["model"]["lightgbm"]["rmse"] > 0
    assert json.loads((out / "validation.json").read_text())["n_plays"] == 40
    frames = pd.read_parquet(out / "animation_frames.parquet")
    assert {"target", "defense", "offense"} <= set(frames["role"]) and frames["t"].min() < 0 < frames["t"].max()
    from src.tracking.closing_animate import animate_closing

    g, p = frames[closing.KEYS].iloc[0]
    assert len(animate_closing(frames, g, p).data) == 6
