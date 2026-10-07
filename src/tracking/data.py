"""Load and standardize NFL Big Data Bowl 2024 tracking data (offense always moves left to right)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import path

DATA_URL = "https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data"
STATIC_FILES = ("games.csv", "plays.csv", "players.csv", "tackles.csv")
KEYS = ["gameId", "playId"]
FIELD_LENGTH, FIELD_WIDTH = 120.0, 160 / 3
TRACKING_COLS = [
    "gameId",
    "playId",
    "nflId",
    "displayName",
    "frameId",
    "club",
    "playDirection",
    "x",
    "y",
    "s",
    "a",
    "dir",
    "event",
]


def available_weeks() -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in path("bdb").glob("tracking_week_*.csv"))


def check_data(weeks: list[int] | None = None) -> list[int]:
    """Return the usable weeks or raise a FileNotFoundError that says how to get the data."""
    have = available_weeks()
    wanted = weeks or have
    missing = [f for f in STATIC_FILES if not (path("bdb") / f).exists()]
    missing += [f"tracking_week_{w}.csv" for w in wanted if w not in have]
    if missing or not wanted:
        raise FileNotFoundError(
            f"BDB 2024 data not found in {path('bdb')} (missing: {missing or 'all tracking weeks'}). "
            f"The files at {DATA_URL} were replaced by a README in August 2025; "
            "copy a local download of the CSVs into that folder."
        )
    return wanted


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Flip plays moving left so offense always goes +x, and add velocity components (x right, y up)."""
    df = df.copy()
    left = (df["playDirection"] == "left").to_numpy()
    df["x"] = np.where(left, FIELD_LENGTH - df["x"], df["x"])
    df["y"] = np.where(left, FIELD_WIDTH - df["y"], df["y"])
    for col in ("dir", "o"):
        if col in df:
            df[col] = np.where(left, (df[col] + 180) % 360, df[col])
    rad = np.deg2rad(df["dir"])  # degrees clockwise from the +y axis
    df["vx"], df["vy"] = df["s"] * np.sin(rad), df["s"] * np.cos(rad)
    return df


def load_static() -> dict[str, pd.DataFrame]:
    return {f.removesuffix(".csv"): pd.read_csv(path("bdb") / f) for f in STATIC_FILES}


def load_tracking(week: int) -> pd.DataFrame:
    """One week of standardized tracking (weeks are loaded lazily, one file at a time)."""
    cols = TRACKING_COLS + ["o"]
    df = pd.read_csv(path("bdb") / f"tracking_week_{week}.csv", usecols=lambda c: c in cols)
    df = standardize(df)
    df["week"] = week
    return df
