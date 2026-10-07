"""Validation and draft link for Closing Over Expected: model vs. naive baselines, split-half stability, outcome
relevance (completion and EPA by COE quintile, logistic regression) and a link to pre-draft features.

Outcome relevance uses ONE defender per play, the one closest to the targeted receiver at the throw (a pre-throw
choice, so it is not selected on the result). Completion is regressed on COE controlling for the starting distance and
the expected end distance; the defender's actual end distance is COE's complement, so the COE coefficient is the
effect of finishing one yard closer than expected. Standard errors are Wald and treat plays as independent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.config import cfg
from src.ingest.identity import norm_name

log = logging.getLogger(__name__)
KEYS = ["game_id", "play_id"]
LINK_FEATURES = [
    "speed_score",
    "athletic_score",
    "agility_score",
    "explosion_score",
    "forty",
    "pick",
    "round",
    "p_starter",
]


def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    v = a.corr(b, method=method) if len(a) >= 3 else np.nan
    return None if np.isnan(v) else float(v)


def model_errors(df: pd.DataFrame) -> dict:
    y = df["end_dist"]
    preds = {
        "lightgbm": df["expected_end"],
        "linear_start_air": df["base_linear"],
        "linear_geometry": df["base_geometry"],
        "end_equals_start": df["start_dist"],
    }
    return {
        k: {"rmse": float(np.sqrt(((y - p) ** 2).mean())), "mae": float((y - p).abs().mean())} for k, p in preds.items()
    } | {"end_dist_sd": float(y.std()), "n": int(len(df))}


def split_half(df: pd.DataFrame, min_half: int) -> dict:
    """Mean COE per player in odd vs. even weeks, players with at least `min_half` plays in both."""
    g = df.assign(half=np.where(df["week"] % 2 == 1, "odd", "even")).groupby(["nfl_id", "half"])["coe"]
    t = g.agg(["size", "mean"]).unstack("half").dropna()
    t = t[(t[("size", "odd")] >= min_half) & (t[("size", "even")] >= min_half)]
    r = _corr(t[("mean", "odd")], t[("mean", "even")])
    return {"n_players": int(len(t)), "min_plays_per_half": min_half, "split_half_r": r,
            "spearman_brown": None if r is None else 2 * r / (1 + r),
            "spearman": _corr(t[("mean", "odd")], t[("mean", "even")], "spearman")}  # fmt: skip


def _logit(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    m = LogisticRegression(C=np.inf, max_iter=1000).fit(X, y)
    Xi = np.column_stack([np.ones(len(X)), X])
    beta = np.r_[m.intercept_, m.coef_[0]]
    p = 1 / (1 + np.exp(-Xi @ beta))
    se = np.sqrt(np.clip(np.diag(np.linalg.inv((Xi * (p * (1 - p))[:, None]).T @ Xi)), 0, None))[1:]
    z = beta[1:] / se
    return {n: {"coef": float(b), "se": float(s), "p": float(2 * stats.norm.sf(abs(zz)))}
            for n, b, s, zz in zip(names, beta[1:], se, z)}  # fmt: skip


def outcome_relevance(df: pd.DataFrame) -> dict:
    """Completion / EPA vs. COE for the defender closest to the receiver at the throw."""
    d = df[df["play_nullified_by_penalty"] != "Y"].dropna(subset=["pass_result"])
    d = d.loc[d.groupby(KEYS)["start_dist"].idxmin()].copy()
    d["complete"] = (d["pass_result"] == "C").astype(int)
    d["quintile"] = pd.qcut(d["coe"], 5, labels=False) + 1
    q = d.groupby("quintile").agg(n=("coe", "size"), mean_coe=("coe", "mean"), completion_rate=("complete", "mean"),
                                  mean_epa=("expected_points_added", "mean"))  # fmt: skip
    fit = _logit(d[["coe", "start_dist", "expected_end"]].to_numpy(), d["complete"].to_numpy(),
                 ["coe", "start_dist", "expected_end"])  # fmt: skip
    return {
        "defender": "closest to the targeted receiver at the throw",
        "n_plays": int(len(d)), "completion_rate": float(d["complete"].mean()),
        "by_quintile": q.reset_index().to_dict("records"),
        "q5_minus_q1_completion": float(q["completion_rate"].iloc[-1] - q["completion_rate"].iloc[0]),
        "q5_minus_q1_epa": float(q["mean_epa"].iloc[-1] - q["mean_epa"].iloc[0]),
        "logit_completion": fit,
        "epa_corr_spearman": _corr(d["coe"], d["expected_points_added"], "spearman"),
    }  # fmt: skip


def validate(df: pd.DataFrame, lb: pd.DataFrame) -> dict:
    c = cfg()["closing"]
    return {
        "n_defender_plays": int(len(df)), "n_plays": int(df[KEYS].drop_duplicates().shape[0]),
        "n_players": int(df["nfl_id"].nunique()), "n_players_leaderboard": int(len(lb)),
        "min_plays": c["min_plays"], "coe_sd": float(df["coe"].std()),
        "model": model_errors(df),
        "stability": split_half(df, max(c["min_plays"] // 2, 1)),
        "outcome": outcome_relevance(df),
    }  # fmt: skip


def match_prospects(players: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """BDB players -> prospects: normalized name + birth date, else normalized name + position group when unique."""
    from src.config import position_group

    pl = players.assign(key=players["player_name"].map(norm_name),
                        bd=pd.to_datetime(players["player_birth_date"], errors="coerce").dt.normalize(),
                        pg=players["position"].map(lambda p: position_group(p) or ("LB" if p == "MLB" else None)))  # fmt: skip
    pr = prospects.assign(key=prospects["player_name"].map(norm_name),
                          bd=pd.to_datetime(prospects["birth_date"], errors="coerce").dt.normalize())  # fmt: skip
    pr = pr[pr["draft_year"] <= 2023]

    def unique(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        f = frame.dropna(subset=cols)
        return f[~f.duplicated(cols, keep=False)]

    a = unique(pl, ["key", "bd"]).merge(unique(pr, ["key", "bd"]), on=["key", "bd"], suffixes=("", "_p"))
    a["match_method"] = "name_birthdate"
    # uniqueness is judged on the full lists, so a namesake already matched by birth date keeps the name ambiguous
    rest = unique(pl, ["key", "pg"])
    rest = rest[~rest["nfl_id"].isin(a["nfl_id"])]
    pool = unique(pr, ["key", "pos_group"])
    pool = pool[~pool["player_key"].isin(a["player_key"])]
    b = rest.merge(pool.rename(columns={"pos_group": "pg"}),
                                          on=["key", "pg"], suffixes=("", "_p"))  # fmt: skip
    b["match_method"] = "name_position_group"
    b["pos_group"] = b["pg"]
    return pd.concat([a, b], ignore_index=True)


def link_correlations(m: pd.DataFrame, min_plays: int) -> dict:
    """Spearman between mean COE and pre-draft features (pairwise complete), overall and for DBs only."""

    def block(t: pd.DataFrame) -> dict:
        res = {}
        for f in LINK_FEATURES:
            if f not in t:
                continue
            ok = t[["mean_coe", f]].dropna()
            rho, p = stats.spearmanr(ok["mean_coe"], ok[f]) if len(ok) >= 5 else (np.nan, np.nan)
            res[f] = {"n": int(len(ok)), "spearman": None if np.isnan(rho) else float(rho),
                      "p": None if np.isnan(p) else float(p)}  # fmt: skip
        return res

    t = m[m["plays"] >= min_plays]
    return {"min_plays": min_plays, "n_players": int(len(t)), "all": block(t),
            "db_only": {"n_players": int((t["pos_group"] == "DB").sum()), **block(t[t["pos_group"] == "DB"])}}  # fmt: skip


def draft_link(df: pd.DataFrame, allp: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    """Match BDB defenders to the warehouse prospects; returns (matched table, summary) or (None, reason)."""
    from src.db import Warehouse

    try:
        wh = Warehouse(read_only=True)
        try:
            pros, pred = wh.table("marts.prospects"), wh.table("marts.predictions")
        finally:
            wh.close()
    except Exception as e:  # missing warehouse must not break the tracking step
        log.warning("Skipping draft link: %s", e)
        return None, {"skipped": str(e)}
    bd = df.drop_duplicates("nfl_id").set_index("nfl_id")["player_birth_date"]
    players = allp.assign(player_birth_date=allp["nfl_id"].map(bd))
    pros = pros.merge(pred[["player_key", "p_starter"]], on="player_key", how="left")
    m = match_prospects(players, pros)
    keep = [
        "nfl_id",
        "player_name",
        "position",
        "team",
        "plays",
        "mean_coe",
        "coe_se",
        "mean_closing",
        "match_method",
        "player_key",
        "pos_group",
        "draft_year",
        "round",
        "pick",
        "undrafted",
        *LINK_FEATURES,
    ]
    m = m[[c for c in dict.fromkeys(keep) if c in m]].assign(pick=lambda t: t["pick"].astype(float))
    lb_ok = allp[allp["plays"] >= cfg()["closing"]["min_plays"]]
    ok = m[m["plays"] >= cfg()["closing"]["min_plays"]]
    summary = {
        "n_players": int(len(allp)), "n_matched": int(len(m)), "match_rate": float(len(m) / len(allp)),
        "n_players_min_plays": int(len(lb_ok)), "n_matched_min_plays": int(len(ok)),
        "match_rate_min_plays": float(len(ok) / max(len(lb_ok), 1)),
        "by_method": m["match_method"].value_counts().to_dict(),
        "correlations": link_correlations(m, cfg()["closing"]["min_plays"]),
    }  # fmt: skip
    return m, summary
