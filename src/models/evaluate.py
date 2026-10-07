"""Backtest metrics and figures: proper scoring rules, paired bootstrap, calibration, interval coverage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from src.config import cfg  # noqa: E402

# Reference categorical palette (validated light-mode steps): model, market baseline, athletic baseline.
MODEL, MARKET, ATHLETIC, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#e6e5e1"
LABELS = {
    "pick_only": "Draft slot only",
    "athletic_only": "Athleticism only",
    "scouting_single": "Scouting (no slot), single",
    "scouting_per_group": "Scouting (no slot), per position",
    "full_single": "Full model, single",
    "full_per_group": "Full model, per position",
    "blend": "Market + model blend",
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(err)


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = np.full_like(p, y.mean())
    brier = brier_score_loss(y, p)
    return {
        "brier": float(brier),
        "brier_skill": float(1 - brier / brier_score_loss(y, base)),
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ece": ece(y, p),
    }


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000) -> dict:
    """Bootstrap the Brier difference (b - a); positive means model a is better."""
    rng = np.random.default_rng(cfg()["seed"])
    la, lb = (p_a - y) ** 2, (p_b - y) ** 2
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = lb[idx].mean(axis=1) - la[idx].mean(axis=1)
    return {
        "brier_gain": float(lb.mean() - la.mean()),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_better": float((diffs > 0).mean()),
    }


def summarize(bt: pd.DataFrame, specs: list[str]) -> dict:
    y = bt["starter"].to_numpy(int)
    out = {
        "n_test": int(len(bt)),
        "classes": sorted(int(c) for c in bt["draft_year"].unique()),
        "base_rate": float(y.mean()),
        "models": {s: score(y, bt[f"p_{s}"].to_numpy()) for s in specs},
        "by_class": {
            int(yr): {s: score(g["starter"].to_numpy(int), g[f"p_{s}"].to_numpy()) for s in ("pick_only", "blend")}
            for yr, g in bt.groupby("draft_year")
        },
        "by_position": {
            g: {s: score(d["starter"].to_numpy(int), d[f"p_{s}"].to_numpy())["brier"] for s in specs}
            for g, d in bt.groupby("pos_group")
            if d["starter"].nunique() == 2
        },
    }
    for s in ("blend", "full_single", "full_per_group", "scouting_single", "athletic_only"):
        out[f"bootstrap_{s}_vs_pick"] = paired_bootstrap(y, bt[f"p_{s}"].to_numpy(), bt["p_pick_only"].to_numpy())
    share = bt["snap_share_3yr"].to_numpy()
    out["interval_80_coverage"] = float(((share >= bt["q10"]) & (share <= bt["q90"])).mean())
    out["interval_median_abs_error"] = float(np.abs(share - bt["q50"]).mean())
    return out


def summarize_undrafted(df: pd.DataFrame, specs: list[str]) -> dict:
    """Undrafted combine invitees: the slot says nothing within this group, so does the profile rank them?

    With ~1% starters, AUC on the starter label is noisy; Spearman correlation with the 3-year snap
    share and the snap share of the model's top decile use the full outcome instead.
    """
    share = df["snap_share_3yr"]
    out = {"n": int(len(df)), "classes": sorted(int(c) for c in df["draft_year"].unique()),
           "starters": int(df["starter"].sum()), "mean_snap_share": float(share.mean()), "models": {}}  # fmt: skip
    for s in specs:
        p = df[f"p_{s}"]
        top = p >= p.quantile(0.9)
        out["models"][s] = {
            "auc": float(roc_auc_score(df["starter"].astype(int), p)),
            "spearman_snap_share": float(p.corr(share, method="spearman")),
            "top_decile_snap_share": float(share[top].mean()),
            "rest_snap_share": float(share[~top].mean()),
        }
    return out


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability(bt: pd.DataFrame, best: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    for spec, color in [("pick_only", MARKET), (best, MODEL)]:
        p, y = bt[f"p_{spec}"].to_numpy(), bt["starter"].to_numpy()
        order = np.argsort(p)
        pts = [(p[c].mean(), y[c].mean()) for c in np.array_split(order, 10)]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=LABELS[spec],
                markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(starter)", ylabel="Observed starter rate")
    ax.set_title("Calibration on held-out draft classes", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=160)
    plt.close(fig)


def skill_bars(bt: pd.DataFrame, specs: list[str], out: Path) -> None:
    y = bt["starter"].to_numpy(int)
    rng = np.random.default_rng(cfg()["seed"])
    idx = rng.integers(0, len(y), size=(1000, len(y)))
    rows = []
    for s in specs:
        p = bt[f"p_{s}"].to_numpy()
        loss, base = (p - y) ** 2, (y.mean() - y) ** 2
        boots = 1 - loss[idx].mean(1) / base[idx].mean(1)
        rows.append((LABELS[s], 1 - loss.mean() / base.mean(), *np.quantile(boots, [0.025, 0.975])))
    df = pd.DataFrame(rows, columns=["model", "skill", "lo", "hi"]).sort_values("skill")
    colors = [MARKET if m == LABELS["pick_only"] else ATHLETIC if m == LABELS["athletic_only"] else
              MODEL if m == LABELS["blend"] else MUTED for m in df["model"]]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(df["model"], df["skill"], color=colors, height=0.55)
    ax.errorbar(df["skill"], df["model"], xerr=[df["skill"] - df["lo"], df["hi"] - df["skill"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)  # fmt: skip
    for i, v in enumerate(df["skill"]):
        ax.text(max(v, 0) + 0.005, i + 0.3, f"{v:.3f}", fontsize=8, color="#52514e", va="center")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Brier skill score vs. base rate (higher is better, 95% bootstrap CI)")
    ax.set_title("Held-out draft classes: who predicts starters best?", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "model_skill.png", dpi=160)
    plt.close(fig)


def plots(bt: pd.DataFrame, specs: list[str], best: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reliability(bt, best, out)
    skill_bars(bt, specs, out)
