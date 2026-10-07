"""Tackles Over Expected: a frame-level tackle-opportunity model aggregated to defenders.

For every defender-frame between the moment the ball carrier is defined (handoff / catch / run) and the end of the
play we model P(this defender records a tackle or assist AND gets within `contact_radius` of the carrier within the
next `frames_ahead` frames). Out-of-fold probabilities (GroupKFold by game) are collapsed to one expected-tackle value
per defender-play, and compared with what the defender actually did.
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
    d["pursuit_cos"] = np.where(norm > 1e-6, (d["vx"] * wx + d["vy"] * wy) / norm.clip(lower=1e-6), 0.0)
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


def add_labels(d: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """Label = defender is credited with a tackle/assist and touches the carrier within the next frames_ahead frames."""
    t = cfg()["tracking"]
    d = d.sort_values(KEYS + ["nflId", "frameId"]).copy()
    by = d.groupby(KEYS + ["nflId"])["dist"]
    future = np.full(len(d), np.nan)
    for k in range(1, t["frames_ahead"] + 1):
        future = np.fmin(future, by.shift(-k).to_numpy())
    made = tackles[(tackles["tackle"] == 1) | (tackles["assist"] == 1)][KEYS + ["nflId"]].assign(made=1)
    d = d.merge(made, on=KEYS + ["nflId"], how="left")
    d["made"] = d["made"].fillna(0).astype(int)
    d["label"] = ((d["made"] == 1) & (future <= t["contact_radius"])).astype(int).to_numpy()
    return d[~np.isnan(future)]  # the last frame of a play has no future to label


def defender_frames(track: pd.DataFrame, plays: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """Feature + label table with one row per defender-frame, from standardized tracking."""
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
    return add_labels(d, tackles)


def group_folds(groups: pd.Series, n_splits: int | None = None):
    """GroupKFold index pairs; a group (game) is never in both train and test."""
    n = min(n_splits or cfg()["tracking"]["n_splits"], groups.nunique())
    yield from GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)


def fit_oof(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold P(tackle soon) for every defender-frame, GroupKFold by game."""
    params = {**cfg()["tracking"]["lgbm"], "random_state": cfg()["seed"]}
    X, y = df[FEATURES], df["label"].to_numpy()
    oof = np.zeros(len(df))
    for tr, te in group_folds(df["gameId"]):
        model = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def defender_plays(df: pd.DataFrame, tackles: pd.DataFrame) -> pd.DataFrame:
    """One row per defender-play with an opportunity: expected vs. actual tackle, plus pursuit efficiency inputs."""
    r = cfg()["tracking"]["opportunity_radius"]
    near = df["dist"] <= r
    d = df.assign(
        radial=np.where(near, df["def_speed"] * df["pursuit_cos"], 0.0), speed_near=np.where(near, df["def_speed"], 0.0)
    )
    g = d.groupby(KEYS + ["nflId"], as_index=False).agg(
        week=("week", "first"), expected=("proba", "max"), min_dist=("dist", "min"),
        radial=("radial", "sum"), speed_near=("speed_near", "sum"),
    )  # fmt: skip
    g = g[g["min_dist"] <= r]
    t = tackles.assign(made=((tackles["tackle"] == 1) | (tackles["assist"] == 1)).astype(int))
    t = t.rename(columns={"pffMissedTackle": "missed"})[KEYS + ["nflId", "made", "missed"]]
    g = g.merge(t, on=KEYS + ["nflId"], how="left").fillna({"made": 0, "missed": 0})
    # ponytail: the peak frame probability is not a calibrated play-level probability; rescale so league TOE sums to 0.
    g["expected"] *= g["made"].sum() / g["expected"].sum()
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
