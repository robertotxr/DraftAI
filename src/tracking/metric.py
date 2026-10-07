"""Tackles Over Expected: a tackle-opportunity model evaluated at a pre-contact decision point per defender.

The decision point of a defender-play is the first frame (from handoff / catch / run on) at which the defender is within
`opportunity_radius` of the ball carrier. Features are measured at that frame only and the label is the play outcome
(tackle or assist), so nothing after the decision leaks into the expectation. Out-of-fold probabilities (GroupKFold by
game) are calibrated tackle probabilities; Tackles Over Expected is actual minus expected, summed per defender.
The same model applied to every frame gives the animation's "tackle probability if evaluated now".
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import cfg
from src.tracking.data import FIELD_WIDTH, KEYS

START_EVENTS = ["handoff", "pass_outcome_caught", "run"]
END_EVENTS = [
    "tackle", "out_of_bounds", "touchdown", "fumble", "qb_slide", "safety",
    "fumble_defense_recovered", "fumble_offense_recovered",
]  # fmt: skip
FEATURES = [
    "dist", "closing_speed", "rel_speed", "pursuit_angle", "carrier_speed", "def_speed", "def_accel",
    "blockers_closer", "sideline_dist",
]  # fmt: skip


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Geometry features from defender (x, y, s, a, vx, vy) and carrier (cx, cy, cs, cvx, cvy) columns."""
    t = cfg()["tracking"]
    d = d.copy()
    dx, dy = d["cx"] - d["x"], d["cy"] - d["y"]
    d["dist"] = np.hypot(dx, dy)
    ux, uy = dx / d["dist"].clip(lower=1e-6), dy / d["dist"].clip(lower=1e-6)
    rvx, rvy = d["vx"] - d["cvx"], d["vy"] - d["cvy"]
    d["closing_speed"] = rvx * ux + rvy * uy  # positive when the gap is shrinking
    d["rel_speed"] = np.hypot(rvx, rvy)
    # Pursuit angle: defender heading vs. direction to where the carrier will be in `intercept_seconds`.
    wx, wy = d["cx"] + d["cvx"] * t["intercept_seconds"] - d["x"], d["cy"] + d["cvy"] * t["intercept_seconds"] - d["y"]
    norm = d["s"] * np.hypot(wx, wy)
    d["pursuit_cos"] = np.where(
        norm > 1e-6, (d["vx"] * wx + d["vy"] * wy) / norm.clip(lower=1e-6), np.nan
    )  # NaN: standing
    d["pursuit_angle"] = np.degrees(np.arccos(d["pursuit_cos"].clip(-1, 1)))
    d["carrier_speed"], d["def_speed"], d["def_accel"] = d["cs"], d["s"], d["a"]
    d["sideline_dist"] = np.minimum(d["cy"], FIELD_WIDTH - d["cy"])
    return d


def active_window(t: pd.DataFrame) -> pd.DataFrame:
    """Keep frames from the first start event until the first end event (or last frame)."""
    ev = t.loc[t["event"].notna(), KEYS + ["frameId", "event"]]
    start = ev[ev["event"].isin(START_EVENTS)].groupby(KEYS)["frameId"].min().rename("f0")
    end = ev[ev["event"].isin(END_EVENTS)].merge(start.reset_index(), on=KEYS)
    end = end[end["frameId"] >= end["f0"]].groupby(KEYS)["frameId"].min().rename("f1")
    t = t.merge(start, on=KEYS).merge(end, on=KEYS, how="left")
    keep = (t["frameId"] >= t["f0"]) & (t["frameId"] <= t["f1"].fillna(np.inf))
    return t[keep].drop(columns=["f0", "f1"])


def add_made(d: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """Add `made`: the defender is credited with a tackle or assist on the play."""
    made = tackles[(tackles["tackle"] == 1) | (tackles["assist"] == 1)][KEYS + ["nflId"]].assign(made=1)
    d = d.merge(made, on=KEYS + ["nflId"], how="left")
    d["made"] = d["made"].fillna(0).astype(int)
    return d


def decision_points(frames: pd.DataFrame) -> pd.DataFrame:
    """First frame per defender-play inside opportunity_radius; label = play outcome (`made`), never later frames."""
    r = cfg()["tracking"]["opportunity_radius"]
    near = frames[frames["dist"] <= r].sort_values("frameId")
    dec = near.groupby(KEYS + ["nflId"], as_index=False).first()
    return dec.assign(label=dec["made"])


def defender_frames(track: pd.DataFrame, plays: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """Feature table with one row per defender-frame (plus `made`), from standardized tracking."""
    p = plays[KEYS + ["ballCarrierId", "possessionTeam", "defensiveTeam"]]
    t = active_window(track[track["nflId"].notna()].merge(p, on=KEYS))
    carrier = t[t["nflId"] == t["ballCarrierId"]]
    car = carrier[KEYS + ["frameId", "x", "y", "s", "vx", "vy"]]
    car = car.rename(columns={"x": "cx", "y": "cy", "s": "cs", "vx": "cvx", "vy": "cvy"})
    d = t[t["club"] == t["defensiveTeam"]].merge(car, on=KEYS + ["frameId"])
    d = add_features(d)
    off = t[(t["club"] == t["possessionTeam"]) & (t["nflId"] != t["ballCarrierId"])]
    off = off.merge(car, on=KEYS + ["frameId"])
    off = off.assign(od=np.hypot(off["cx"] - off["x"], off["cy"] - off["y"]))[KEYS + ["frameId", "od"]]
    pair = d[KEYS + ["frameId", "nflId", "dist"]].merge(off, on=KEYS + ["frameId"])
    blockers = (pair["od"] < pair["dist"]).groupby([pair[k] for k in KEYS + ["frameId", "nflId"]]).sum()
    d = d.merge(blockers.rename("blockers_closer").reset_index(), on=KEYS + ["frameId", "nflId"], how="left")
    d["blockers_closer"] = d["blockers_closer"].fillna(0)
    return add_made(d, tackles)


def group_folds(groups: pd.Series, n_splits: int | None = None):
    """GroupKFold index pairs; a group (game) is never in both train and test."""
    n = min(n_splits or cfg()["tracking"]["n_splits"], groups.nunique())
    yield from GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)


def fit_oof(dec: pd.DataFrame, frames: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold P(tackle) for decision points (and for every frame of the held-out games), GroupKFold by game."""
    params = {**cfg()["tracking"]["lgbm"], "random_state": cfg()["seed"]}
    X, y = dec[FEATURES], dec["label"].to_numpy()
    oof = np.zeros(len(dec))
    oof_frames = None if frames is None else np.zeros(len(frames))
    for tr, te in group_folds(dec["gameId"]):
        model = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        if frames is not None:
            held_out = frames["gameId"].isin(dec["gameId"].iloc[te]).to_numpy()
            oof_frames[held_out] = model.predict_proba(frames.loc[held_out, FEATURES])[:, 1]
    return oof, oof_frames


def defender_plays(dec: pd.DataFrame, frames: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """One row per defender-play opportunity: expected (decision-point probability) vs. actual, pursuit inputs."""
    r = cfg()["tracking"]["opportunity_radius"]
    ok = (frames["dist"] <= r) & frames["pursuit_cos"].notna()  # standing defenders have no heading
    f = frames.assign(
        radial=np.where(ok, frames["def_speed"] * frames["pursuit_cos"], 0.0),
        speed_near=np.where(ok, frames["def_speed"], 0.0),
    )
    pursuit = f.groupby(KEYS + ["nflId"], as_index=False)[["radial", "speed_near"]].sum()
    g = dec[KEYS + ["nflId", "week", "proba", "made"]].rename(columns={"proba": "expected"})
    g = g.merge(pursuit, on=KEYS + ["nflId"], how="left")
    missed = tackles.rename(columns={"pffMissedTackle": "missed"})[KEYS + ["nflId", "missed"]]
    g = g.merge(missed, on=KEYS + ["nflId"], how="left").fillna({"missed": 0})
    return g.assign(toe=g["made"] - g["expected"])


def leaderboard(dp: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Player metric: Tackles Over Expected plus pursuit efficiency (share of speed aimed at the intercept point)."""
    lb = dp.groupby("nflId", as_index=False).agg(
        opportunities=("toe", "size"), tackles=("made", "sum"), missed=("missed", "sum"), expected=("expected", "sum"),
        toe=("toe", "sum"), radial=("radial", "sum"), speed_near=("speed_near", "sum"),
    )  # fmt: skip
    lb["toe_per_100"] = 100 * lb["toe"] / lb["opportunities"]
    lb["missed_rate"] = lb["missed"] / (lb["tackles"] + lb["missed"]).replace(0, np.nan)
    lb["pursuit_efficiency"] = lb["radial"] / lb["speed_near"].replace(0, np.nan)
    lb = lb.drop(columns=["radial", "speed_near"]).merge(
        players[["nflId", "displayName", "position"]], on="nflId", how="left"
    )
    lb = lb[lb["opportunities"] >= cfg()["tracking"]["min_opportunities"]]
    return lb.sort_values("toe", ascending=False).reset_index(drop=True)
