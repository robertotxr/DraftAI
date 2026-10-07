"""Position-normalized athletic score (RAS-style, 0-10 scale).

For each drill and position group we score a player two ways against a reference population of
combine participants:
  * raw score:  percentile of the raw result within the position group (0-10)
  * size score: percentile of the residual after regressing the drill on height and weight within
                the group, so a 4.55 forty at 250 lb outranks a 4.55 at 190 lb
Lower-is-better drills (forty, cone, shuttle) are flipped so 10 is always elite.

Missing measurements are never imputed: the drill score stays NaN, `<drill>_measured` records it,
and the composite is only produced when at least `athletic.min_drills` drills were measured.
The public combine data has no 10-yard split, so it is not part of the score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import cfg, position_group

COMBINE_POS_ALIASES = {"EDGE": "DE", "DL": "DT", "OLB": "OLB", "SAF": "S"}


def combine_group(pos: str) -> str | None:
    return position_group(COMBINE_POS_ALIASES.get(pos, pos))


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[~np.isnan(reference)])
    out = np.full(values.shape, np.nan)
    ok = ~np.isnan(values)
    if len(ref) == 0:
        return out
    # Mid-rank percentile so ties land in the middle of their block.
    lo = np.searchsorted(ref, values[ok], side="left")
    hi = np.searchsorted(ref, values[ok], side="right")
    out[ok] = (lo + hi) / 2 / len(ref)
    return out


def _size_residual_fit(ref: pd.DataFrame, drill: str):
    """OLS drill ~ 1 + height + weight on reference rows; returns predictor or None."""
    d = ref[[drill, "ht_in", "wt_lb"]].dropna()
    if len(d) < 30:
        return None
    X = np.c_[np.ones(len(d)), d[["ht_in", "wt_lb"]].to_numpy()]
    beta, *_ = np.linalg.lstsq(X, d[drill].to_numpy(), rcond=None)
    return lambda df: np.c_[np.ones(len(df)), df[["ht_in", "wt_lb"]].to_numpy()] @ beta


def score(combine: pd.DataFrame, group_col: str = "pos_group", reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add drill scores and composites to `combine`.

    `reference` is the population that defines the norms (defaults to `combine` itself).
    Both frames need: group_col, ht_in, wt_lb and the drill columns.
    """
    a = cfg()["athletic"]
    drills, lower = a["drills"], set(a["lower_is_better"])
    ref_all = combine if reference is None else reference
    out = combine.copy()
    for d in drills:
        out[f"{d}_measured"] = out[d].notna().astype(int)
        out[f"{d}_raw_score"] = np.nan
        out[f"{d}_size_score"] = np.nan
    for g, idx in out.groupby(group_col).groups.items():
        ref = ref_all[ref_all[group_col] == g]
        sub = out.loc[idx]
        for d in drills:
            sign = -1.0 if d in lower else 1.0
            out.loc[idx, f"{d}_raw_score"] = 10 * _percentile(
                sign * ref[d].to_numpy(float), sign * sub[d].to_numpy(float)
            )
            fit = _size_residual_fit(ref, d)
            if fit is None:
                continue
            ref_ok = ref[[d, "ht_in", "wt_lb"]].dropna()
            ref_res = sign * (ref_ok[d].to_numpy() - fit(ref_ok))
            has_size = sub[["ht_in", "wt_lb"]].notna().all(axis=1).to_numpy()
            res = np.full(len(sub), np.nan)
            res[has_size] = sign * (sub[d].to_numpy(float)[has_size] - fit(sub[has_size]))
            out.loc[idx, f"{d}_size_score"] = 10 * _percentile(ref_res, res)
        for m in ("ht_in", "wt_lb"):
            out.loc[idx, f"{m}_score"] = 10 * _percentile(ref[m].to_numpy(float), sub[m].to_numpy(float))

    size_cols = [f"{d}_size_score" for d in drills]
    out["drills_measured"] = out[[f"{d}_measured" for d in drills]].sum(axis=1)
    enough = out["drills_measured"] >= a["min_drills"]
    out["athletic_score"] = out[size_cols].mean(axis=1).where(enough)
    out["speed_score"] = out[["forty_size_score"]].mean(axis=1)
    out["explosion_score"] = out[["vertical_size_score", "broad_jump_size_score"]].mean(axis=1)
    out["agility_score"] = out[["cone_size_score", "shuttle_size_score"]].mean(axis=1)
    return out


def build(combine: pd.DataFrame, norm_max_year: int) -> pd.DataFrame:
    """Score every combine participant against norms from combines up to `norm_max_year`."""
    c = combine.copy()
    c["pos_group"] = c["combine_pos"].map(combine_group)
    c = c[c["pos_group"].notna()]
    reference = c[c["combine_year"] <= norm_max_year]
    return score(c, reference=reference)
