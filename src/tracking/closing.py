"""Closing Over Expected (COE): how much better (or worse) than expected a coverage defender closes on the receiver
while the ball is in the air. Built on NFL Big Data Bowl 2026 Analytics data (2023 pass plays, 10 Hz tracking).

Definition
  Unit: one (play, defender) for every Defensive Coverage player flagged `player_to_predict` (the defenders the
  competition tracks after the throw). Targeted receiver = the play's `Targeted Receiver`.
  Target: defender-to-receiver distance (yards) at the final output frame, i.e. when the ball arrives.
  Expected: out-of-fold LightGBM prediction of that distance (GroupKFold by game, so a game never trains its own
  predictions). COE = expected end distance - actual end distance; positive = the defender ended closer to the
  receiver than players in the same spot usually do. Actual closing = start distance - end distance.

Leakage control
  Features come from the throw frame (last input frame) only. They use the receiver's and defender's position and motion
  at release, the ball landing point, air time (num_frames_output / 10) and pass length (all known at release), and
  the pre-snap coverage call (man vs. zone). The landing point and air time describe where and how the throw goes, so the expectation answers "given
  this throw, how close do defenders end up?"; it does not know what the defender did afterwards. Nothing from the
  output frames (the post-throw movement) enters the features. Pass outcome, yards and EPA are never features.

Standardization
  Plays moving left are rotated 180 degrees (x, y, dir, o and the landing point) so offense always moves to +x,
  reusing `src.tracking.data.standardize`.

Limitations: one defender-play is one noisy sample (the target is a single distance); defenders who play off a man or
a zone landmark are not told apart beyond man/zone; only 2023 pass plays and only the defenders BDB flags.
"""

from __future__ import annotations

import json
import logging
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import cfg, path
from src.tracking import data
from src.tracking.metric import group_folds

log = logging.getLogger(__name__)

KEYS = ["game_id", "play_id"]
HZ = 10.0
DATA_URL = "https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data"
POSITIONS = ["CB", "S", "FS", "SS", "LB", "ILB", "MLB", "OLB", "DE", "DT", "NT"]
FEATURES = [
    "start_dist", "dx_rec", "dy_rec", "dist_land", "rec_dist_land", "air_time", "def_speed", "def_accel",
    "def_vel_land", "def_angle_land", "def_vel_rec", "rec_speed", "rec_vel_land", "rec_vel_x", "rec_vel_y",
    "land_dx", "land_dy", "man", "pass_length", "pos_code",
]  # fmt: skip
SUPP_COLS = KEYS + [
    "week", "possession_team", "defensive_team", "pass_result", "pass_length", "team_coverage_man_zone",
    "expected_points_added", "play_nullified_by_penalty", "play_description",
]  # fmt: skip
OUT_DIR = "tracking_closing"


def find_files() -> tuple[dict[int, tuple], object]:
    """{week: (input_csv, output_csv)} and the supplementary csv; FileNotFoundError says how to get the data."""
    root = path("bdb26")
    inputs = {int(re.search(r"_w(\d+)\.csv", p.name).group(1)): p for p in root.rglob("input_2023_w*.csv")}
    weeks = {w: (p, p.with_name(p.name.replace("input_", "output_"))) for w, p in sorted(inputs.items())}
    supp = next(root.rglob("supplementary_data.csv"), None)
    weeks = {w: ab for w, ab in weeks.items() if ab[1].exists()}
    if not weeks or supp is None:
        raise FileNotFoundError(
            f"BDB 2026 Analytics data not found in {root} (need train/input_2023_w*.csv, output_2023_w*.csv and "
            f"supplementary_data.csv). Run `make bdb` ({DATA_URL}, requires accepting the competition rules)."
        )
    return weeks, supp


def flip_xy(df: pd.DataFrame, left: np.ndarray, cols: tuple[str, str] = ("x", "y")) -> pd.DataFrame:
    """Mirror positions of left-moving plays (180 degree rotation of the field)."""
    df = df.copy()
    df[cols[0]] = np.where(left, data.FIELD_LENGTH - df[cols[0]], df[cols[0]])
    df[cols[1]] = np.where(left, data.FIELD_WIDTH - df[cols[1]], df[cols[1]])
    return df


def standardize(inp: pd.DataFrame) -> pd.DataFrame:
    """Input frames with offense moving to +x, plus velocity components and the flipped landing point."""
    out = data.standardize(inp.rename(columns={"play_direction": "playDirection"}))
    out = out.rename(columns={"playDirection": "play_direction"})
    return flip_xy(out, (out["play_direction"] == "left").to_numpy(), ("ball_land_x", "ball_land_y"))


def _unit(vx, vy):
    n = np.hypot(vx, vy)
    return vx / n, vy / n


def build_units(inp: pd.DataFrame, supp: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, tracked coverage defender) with throw-frame features. Uses no post-throw data.

    `inp` is standardized input tracking; `supp` the supplementary play table (pre-throw scheme columns only are
    used as features, the rest is carried for the outcome analysis).
    """
    last = inp[inp["frame_id"] == inp.groupby(KEYS)["frame_id"].transform("max")]
    rc = {"nfl_id": "rec_id", "x": "rec_x", "y": "rec_y", "s": "rec_speed", "vx": "rec_vx", "vy": "rec_vy",
          "dir": "rec_dir"}  # fmt: skip
    rec = last[last["player_role"] == "Targeted Receiver"].drop_duplicates(KEYS)
    rec = rec[[*KEYS, *rc]].rename(columns=rc)
    d = last[(last["player_role"] == "Defensive Coverage") & last["player_to_predict"].astype(bool)]
    d = d.merge(rec, on=KEYS)
    s = supp[SUPP_COLS].drop_duplicates(KEYS).rename(columns={"pass_length": "pass_length_yds"})
    d = d.merge(s, on=KEYS, how="left")

    dx, dy = d["rec_x"] - d["x"], d["rec_y"] - d["y"]
    lx, ly = d["ball_land_x"] - d["x"], d["ball_land_y"] - d["y"]
    rlx, rly = d["ball_land_x"] - d["rec_x"], d["ball_land_y"] - d["rec_y"]
    ux, uy = np.sin(np.deg2rad(d["dir"])), np.cos(np.deg2rad(d["dir"]))  # heading unit vector
    dist_land = np.hypot(lx, ly)
    cos_land = (ux * lx + uy * ly) / dist_land.replace(0, np.nan)
    to_rec_x, to_rec_y = _unit(dx, dy)
    to_rl_x, to_rl_y = _unit(rlx, rly)
    d = d.assign(
        start_dist=np.hypot(dx, dy), dx_rec=dx, dy_rec=dy, dist_land=dist_land, rec_dist_land=np.hypot(rlx, rly),
        air_time=d["num_frames_output"] / HZ, def_speed=d["s"], def_accel=d["a"],
        def_vel_land=d["s"] * cos_land, def_angle_land=np.degrees(np.arccos(cos_land.clip(-1, 1))),
        def_vel_rec=d["vx"] * to_rec_x + d["vy"] * to_rec_y, rec_vel_land=d["rec_vx"] * to_rl_x + d["rec_vy"] * to_rl_y,
        rec_vel_x=d["rec_vx"], rec_vel_y=d["rec_vy"], land_dx=lx, land_dy=ly,
        man=d["team_coverage_man_zone"].map({"MAN_COVERAGE": 1.0, "ZONE_COVERAGE": 0.0}),
        pass_length=d["pass_length_yds"], pos_code=d["player_position"].map({p: i for i, p in enumerate(POSITIONS)}),
    )  # fmt: skip
    keep = [*KEYS, "nfl_id", "rec_id", "player_name", "player_position", "player_birth_date", "num_frames_output",
            "play_direction", "rec_x", "rec_y", "x", "y", "week", "possession_team", "defensive_team", "pass_result",
            "expected_points_added", "play_nullified_by_penalty", "play_description", *FEATURES]  # fmt: skip
    return d[keep].rename(columns={"x": "def_x", "y": "def_y"}).reset_index(drop=True)


def add_target(units: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Attach the end distance (at the final output frame, ball arrival) and the actual closing; drops rows without it."""
    last = out.merge(units[[*KEYS, "num_frames_output", "play_direction"]].drop_duplicates(KEYS), on=KEYS)
    last = last[last["frame_id"] == last["num_frames_output"]]
    last = flip_xy(last, (last["play_direction"] == "left").to_numpy())[[*KEYS, "nfl_id", "x", "y"]]
    d = units.merge(last.rename(columns={"x": "end_x", "y": "end_y"}), on=[*KEYS, "nfl_id"])
    r = last.rename(columns={"nfl_id": "rec_id", "x": "rec_end_x", "y": "rec_end_y"})
    d = d.merge(r, on=[*KEYS, "rec_id"])
    d["end_dist"] = np.hypot(d["end_x"] - d["rec_end_x"], d["end_y"] - d["rec_end_y"])
    d["actual_closing"] = d["start_dist"] - d["end_dist"]
    return d.drop(columns=["end_x", "end_y", "rec_end_x", "rec_end_y"]).reset_index(drop=True)


def fit_oof(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold expected end distance (LightGBM) plus naive baselines (end = start; linear on start + air time; linear on throw geometry)."""
    c = cfg()["closing"]
    params = {**c["lgbm"], "random_state": cfg()["seed"]}
    X, y = df[FEATURES], df["end_dist"].to_numpy()
    lin = df[["start_dist", "air_time"]].fillna(0)
    geo = df[["start_dist", "air_time", "dist_land", "rec_dist_land"]].fillna(0)
    oof, base_lin, base_geo = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for tr, te in group_folds(df["game_id"], c["n_splits"]):
        oof[te] = lgb.LGBMRegressor(**params).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        base_lin[te] = LinearRegression().fit(lin.iloc[tr], y[tr]).predict(lin.iloc[te])
        base_geo[te] = LinearRegression().fit(geo.iloc[tr], y[tr]).predict(geo.iloc[te])
    oof = np.clip(oof, 0, None)
    return df.assign(expected_end=oof, base_linear=base_lin, base_geometry=base_geo, coe=oof - y)


def leaderboard(df: pd.DataFrame, min_plays: int | None = None) -> pd.DataFrame:
    """Per-defender COE table (all players when `min_plays` is None)."""
    mode = lambda s: s.mode().iat[0]  # noqa: E731
    g = df.groupby("nfl_id")
    lb = g.agg(
        player_name=("player_name", "first"), position=("player_position", mode), team=("defensive_team", mode),
        plays=("coe", "size"), mean_coe=("coe", "mean"), total_coe=("coe", "sum"), coe_sd=("coe", "std"),
        mean_expected=("expected_end", "mean"), mean_actual=("end_dist", "mean"), mean_closing=("actual_closing", "mean"),
    ).reset_index()  # fmt: skip
    lb["coe_se"] = lb["coe_sd"] / np.sqrt(lb["plays"])
    lb = lb.drop(columns="coe_sd")
    if min_plays is not None:
        lb = lb[lb["plays"] >= min_plays]
    return lb.sort_values("mean_coe", ascending=False).reset_index(drop=True)


def animation_frames(inp: pd.DataFrame, out: pd.DataFrame, coe: pd.DataFrame) -> pd.DataFrame:
    """All players per frame for chosen plays: t <= 0 pre-throw (0 = throw), t >= 1 ball in the air.

    Players without output tracking hold their throw-frame position. `coe` has [game_id, play_id, nfl_id, coe].
    """
    inp = inp.copy()
    inp["t"] = inp["frame_id"] - inp.groupby(KEYS)["frame_id"].transform("max")
    inp["predicted"] = inp["player_to_predict"].astype(bool)
    role = {"Targeted Receiver": "target", "Defensive Coverage": "defense"}
    inp["role"] = inp["player_role"].map(role).fillna("offense")
    cols = [*KEYS, "t", "nfl_id", "player_name", "role", "predicted", "x", "y", "ball_land_x", "ball_land_y"]
    pre = inp[cols]
    thr = inp[inp["t"] == 0]
    rep = thr.loc[thr.index.repeat(thr["num_frames_output"])].copy()
    rep["t"] = rep.groupby([*KEYS, "nfl_id"]).cumcount() + 1
    o = out.merge(thr[[*KEYS, "play_direction"]].drop_duplicates(KEYS), on=KEYS)
    o = flip_xy(o, (o["play_direction"] == "left").to_numpy()).rename(columns={"frame_id": "t"})
    rep = rep[cols].merge(
        o[[*KEYS, "nfl_id", "t", "x", "y"]], on=[*KEYS, "nfl_id", "t"], how="left", suffixes=("", "_o")
    )
    rep["x"], rep["y"] = rep["x_o"].fillna(rep["x"]), rep["y_o"].fillna(rep["y"])
    frames = pd.concat([pre, rep.drop(columns=["x_o", "y_o"])], ignore_index=True)
    frames = frames.merge(coe[[*KEYS, "nfl_id", "coe"]], on=[*KEYS, "nfl_id"], how="left")
    desc = frames.rename(columns={"ball_land_x": "land_x", "ball_land_y": "land_y"})
    return desc.sort_values([*KEYS, "t", "nfl_id"]).reset_index(drop=True)


def run(weeks: list[int] | None = None, link: bool = True) -> dict | None:
    """Compute COE, validate, link to the draft warehouse and save artifacts/tracking_closing/. None if no data."""
    from src.tracking import closing_validate as cv

    try:
        files, supp_path = find_files()
    except FileNotFoundError as e:
        log.warning("Skipping Closing Over Expected: %s", e)
        return None
    supp = pd.read_csv(supp_path, low_memory=False)
    rng = np.random.default_rng(cfg()["seed"])
    c = cfg()["closing"]
    wanted = [w for w in files if weeks is None or w in weeks]
    per_week = -(-c["animation_plays"] // len(wanted))
    units, samples = [], []
    for w in wanted:
        inp = pd.read_csv(files[w][0], low_memory=False)
        out = pd.read_csv(files[w][1])
        inp = standardize(inp)
        u = add_target(build_units(inp, supp), out).assign(week=w)
        units.append(u)
        short = u[u["num_frames_output"] <= c["max_animation_air_frames"]][KEYS].drop_duplicates()
        pick = short.iloc[rng.permutation(len(short))[:per_week]]
        samples.append((inp.merge(pick, on=KEYS), out.merge(pick, on=KEYS)))
        log.info("week %d: %d defender-plays", w, len(u))
    units = fit_oof(pd.concat(units, ignore_index=True))
    allp = leaderboard(units)
    lb = allp[allp["plays"] >= c["min_plays"]].reset_index(drop=True)
    report = cv.validate(units, lb)
    outdir = path("artifacts") / OUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    if link:
        matched, report["draft_link"] = cv.draft_link(units, allp)
        if matched is not None:
            matched.to_parquet(outdir / "draft_link.parquet", index=False)
    frames = pd.concat([animation_frames(i, o, units) for i, o in samples], ignore_index=True)
    frames.to_parquet(outdir / "animation_frames.parquet", index=False)
    units.drop(columns=["play_description"]).to_parquet(outdir / "defender_plays.parquet", index=False)
    lb.to_parquet(outdir / "leaderboard.parquet", index=False)
    plays = units[[*KEYS, "play_description"]].drop_duplicates(KEYS)
    plays.merge(frames[KEYS].drop_duplicates(), on=KEYS).to_parquet(outdir / "animation_plays.parquet", index=False)
    (outdir / "validation.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("closing metric saved to %s", outdir)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
