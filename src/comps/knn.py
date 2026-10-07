"""Historical player comps: k nearest neighbours on position-specific, standardized profiles.

Comps for a prospect drafted in year Y come only from classes whose three-year outcome was known
before the Y draft (<= Y - label_lag), so every comp shows a real outcome and nothing from the future.
Distances are NaN-aware: the mean squared z-difference over the features both players have, and a
pair needs at least `MIN_SHARED` shared features to be compared. Nothing is imputed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import cfg
from src.db import Warehouse

log = logging.getLogger(__name__)

MIN_SHARED = 4
BODY = ["ht_in_score", "wt_lb_score", "speed_score", "explosion_score", "agility_score", "draft_age"]
PROFILE = {
    "QB": ["pass_ypa_final", "pass_td_rate_career", "pass_int_rate_career", "pass_cmp_pct_final", "rush_yds_pg_adj_final"],
    "RB": ["scrimmage_share_adj_final", "rush_yds_pg_adj_final", "yds_per_carry_career", "rec_yds_pg_adj_final",
           "breakout_age_scrim"],
    "WR": ["dominator_adj_final", "dominator_adj_best", "rec_yds_pg_adj_final", "breakout_age_rec", "yds_per_rec_career"],
    "TE": ["dominator_adj_final", "dominator_adj_best", "rec_yds_pg_adj_final", "breakout_age_rec", "yds_per_rec_career"],
    "OL": ["bench_size_score"],
    "DL": ["def_playmaking_share_adj_final", "def_playmaking_share_adj_best", "pressure_pg_adj_final", "tackle_share_final"],
    "LB": ["def_playmaking_share_adj_final", "pressure_pg_adj_final", "tackle_share_final", "pd_share_final"],
    "DB": ["pd_share_final", "pd_share_best", "int_pg_career", "tackle_share_final"],
}  # fmt: skip


def comp_features(group: str) -> list[str]:
    return BODY + PROFILE[group]


def nan_distance(a: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared difference over shared non-NaN features; returns (distance, n_shared)."""
    diff = B - a
    shared = ~np.isnan(diff)
    n = shared.sum(axis=1)
    d = np.where(shared, diff**2, 0).sum(axis=1) / np.maximum(n, 1)
    return np.sqrt(d), n


def find_comps(pros: pd.DataFrame, k: int, lag: int, ref_max_year: int) -> pd.DataFrame:
    rows = []
    for group, g in pros.groupby("pos_group"):
        feats = comp_features(group)
        X = g[feats].astype(float)
        # Scale from the reference classes only, so later classes never shape the distance metric.
        ref = X[g["draft_year"] <= ref_max_year]
        Z = ((X - ref.mean()) / ref.std()).to_numpy()
        years, labeled = g["draft_year"].to_numpy(), g["starter"].notna().to_numpy()
        keys = g["player_key"].to_numpy()
        for i in range(len(g)):
            pool = np.where(labeled & (years <= years[i] - lag))[0]
            if not len(pool):
                continue
            d, n = nan_distance(Z[i], Z[pool])
            ok = n >= MIN_SHARED
            order = np.argsort(np.where(ok, d, np.inf))[:k]
            for rank, j in enumerate(order[ok[order]], start=1):
                rows.append((keys[i], rank, keys[pool[j]], float(d[j]), int(n[j])))
    return pd.DataFrame(rows, columns=["player_key", "rank", "comp_key", "distance", "shared_features"])


def run() -> None:
    wh = Warehouse()
    pros = wh.table("marts.prospects")
    comps = find_comps(pros, cfg()["comps"]["k"], cfg()["model"]["label_lag"], cfg()["athletic"]["norm_max_year"])
    # Similarity on a 0-100 scale for display: 100 = identical profile.
    comps["similarity"] = (100 * np.exp(-comps["distance"])).round(1)
    wh.write("marts.comps", comps)
    log.info("marts.comps: %d rows for %d players", len(comps), comps["player_key"].nunique())
    wh.close()
