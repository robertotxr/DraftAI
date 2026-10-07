"""Validation of the tackle model and the player metric: calibration, split-half stability, missed-tackle link."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import cfg, path
from src.models.evaluate import INK, MODEL, MUTED, _style, ece, plt, score


def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    v = a.corr(b, method=method) if len(a) >= 3 else np.nan
    return None if np.isnan(v) else float(v)


def split_half(dp: pd.DataFrame) -> pd.DataFrame:
    """TOE per 100 opportunities in odd vs. even weeks, per player."""
    half = dp.assign(half=np.where(dp["week"] % 2 == 1, "odd", "even")).groupby(["nflId", "half"])
    g = half.agg(n=("toe", "size"), toe=("toe", "sum")).reset_index()
    g = g[g["n"] >= max(cfg()["tracking"]["min_opportunities"] // 2, 1)]
    g["rate"] = 100 * g["toe"] / g["n"]
    return g.pivot(index="nflId", columns="half", values="rate").dropna()


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[list[float], list[float]]:
    chunks = np.array_split(np.argsort(p), bins)
    return [float(p[c].mean()) for c in chunks if len(c)], [float(y[c].mean()) for c in chunks if len(c)]


def validate(frames: pd.DataFrame, dp: pd.DataFrame, lb: pd.DataFrame, figures: bool = True) -> dict:
    """frames: defender-frames with `label` and `proba`; dp: defender_plays output; lb: leaderboard."""
    y, p = frames["label"].to_numpy(), frames["proba"].to_numpy()
    out: dict = {"n_frames": int(len(y)), "base_rate": float(y.mean()), "n_players": int(len(lb))}
    out["frame_model"] = (
        score(y, p) if 0 < y.sum() < len(y) else {"brier": float(((p - y) ** 2).mean()), "ece": ece(y, p)}
    )
    sh = split_half(dp)
    r = _corr(sh.get("odd", pd.Series(dtype=float)), sh.get("even", pd.Series(dtype=float)))
    out["stability"] = {
        "n_players": int(len(sh)), "split_half_r": r, "spearman_brown": None if r is None else 2 * r / (1 + r),
    }  # fmt: skip
    ok = lb.dropna(subset=["missed_rate"])
    out["missed_tackle_link"] = {
        "n_players": int(len(ok)),
        "pearson": _corr(ok["toe_per_100"], ok["missed_rate"]),
        "spearman": _corr(ok["toe_per_100"], ok["missed_rate"], "spearman"),
    }
    if figures:
        _figures(y, p, sh, ok)
    return out


def _figures(y: np.ndarray, p: np.ndarray, sh: pd.DataFrame, ok: pd.DataFrame) -> None:
    fig_dir = path("figures")
    xs, ys = calibration_bins(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, max(xs + ys)], [0, max(xs + ys)], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.plot(xs, ys, color=MODEL, linewidth=2, marker="o", markeredgecolor="white", label="Tackle model (out of fold)")
    ax.set(xlabel="Predicted P(tackle soon)", ylabel="Observed rate")
    ax.set_title("Tackle model calibration on held-out games", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "tracking_calibration.png", dpi=160)
    plt.close(fig)
    for name, a, b, xl, yl, title in [
        ("tracking_stability", *(sh.get(c, pd.Series(dtype=float)) for c in ("odd", "even")),
         "TOE per 100, odd weeks", "TOE per 100, even weeks", "Split-half stability of Tackles Over Expected"),
        ("tracking_missed_tackles", ok["toe_per_100"], ok["missed_rate"],
         "TOE per 100 opportunities", "PFF missed-tackle rate", "Tackles Over Expected vs. missed tackles"),
    ]:  # fmt: skip
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(a, b, s=18, color=MODEL, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=160)
        plt.close(fig)
