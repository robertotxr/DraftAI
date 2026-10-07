"""nflverse pulls via nfl_data_py, cached as parquet so reruns are offline and idempotent."""

from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from src.config import cfg, path

log = logging.getLogger(__name__)


def _cached(name: str, loader) -> pd.DataFrame:
    file = path("nflverse_cache") / f"{name}.parquet"
    if file.exists():
        return pd.read_parquet(file)
    df = loader()
    df.to_parquet(file, index=False)
    log.info("nflverse %s -> %d rows", name, len(df))
    return df


def draft_picks() -> pd.DataFrame:
    y = cfg()["years"]
    return _cached("draft_picks", lambda: nfl.import_draft_picks(list(range(y["draft_first"], y["draft_last"] + 1))))


def combine() -> pd.DataFrame:
    # All combine years since 2000 so position norms use a deep reference population.
    y = cfg()["years"]
    return _cached("combine", lambda: nfl.import_combine_data(list(range(2000, y["draft_last"] + 1))))


def snap_counts() -> pd.DataFrame:
    y = cfg()["years"]
    return _cached(
        "snap_counts", lambda: nfl.import_snap_counts(list(range(y["draft_first"], y["last_nfl_season"] + 1)))
    )


def rosters() -> pd.DataFrame:
    y = cfg()["years"]
    cols = ["season", "team", "position", "player_name", "birth_date", "pfr_id", "player_id", "entry_year"]
    return _cached(
        "rosters",
        lambda: nfl.import_seasonal_rosters(list(range(y["draft_first"], y["last_nfl_season"] + 1)), columns=cols),
    )


def pull_all() -> dict[str, pd.DataFrame]:
    return {
        "nfl_draft_picks": draft_picks(),
        "nfl_combine": combine(),
        "nfl_snap_counts": snap_counts(),
        "nfl_rosters": rosters(),
    }
