"""Entrypoint: fit the tackle model on BDB 2024 tracking, score defenders, validate and save artifacts."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import cfg, path
from src.tracking import data
from src.tracking.animate import animation_frames
from src.tracking.data import KEYS
from src.tracking.metric import decision_points, defender_frames, defender_plays, fit_oof, leaderboard
from src.tracking.validate import validate

log = logging.getLogger(__name__)


def run(weeks: list[int] | None = None) -> dict | None:
    """Compute everything, save parquet/json under artifacts/tracking/, return the validation dict (None if no data)."""
    try:
        weeks = data.check_data(weeks)
    except FileNotFoundError as e:
        log.warning("Skipping tracking metric: %s", e)
        return None
    static = data.load_static()
    plays, tackles = static["plays"], static["tackles"]
    rng = np.random.default_rng(cfg()["seed"])
    per_week = -(-cfg()["tracking"]["animation_plays"] // len(weeks))
    frames, samples = [], []
    for w in weeks:
        track = data.load_tracking(w)
        df = defender_frames(track, plays, tackles).assign(week=w)
        frames.append(df)
        keys = df[KEYS].drop_duplicates()
        pick = keys.iloc[rng.permutation(len(keys))[:per_week]]
        samples.append(track.merge(pick, on=KEYS))
        log.info("week %d: %d defender-frames", w, len(df))
    frames = pd.concat(frames, ignore_index=True)
    dec = decision_points(frames)
    dec["proba"], frames["proba"] = fit_oof(dec, frames)
    dp = defender_plays(dec, frames, tackles)
    lb = leaderboard(dp, static["players"])
    report = validate(dec, dp, lb)

    out = path("artifacts") / "tracking"
    out.mkdir(parents=True, exist_ok=True)
    sample = pd.concat(samples, ignore_index=True)
    animation_frames(sample, plays, frames).to_parquet(out / "animation_frames.parquet", index=False)
    lb.to_parquet(out / "leaderboard.parquet", index=False)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    log.info("tracking metric saved to %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
