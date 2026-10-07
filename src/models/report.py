"""Figures and tables for reports/backtest_2021_2023.md (as-of-draft-night predictions only)."""

from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import path  # noqa: E402
from src.db import Warehouse  # noqa: E402
from src.models.evaluate import INK, MARKET, MODEL, MUTED, _style  # noqa: E402

CLASSES = [2021, 2022, 2023]


def load() -> pd.DataFrame:
    wh = Warehouse(read_only=True)
    pred = wh.table("marts.predictions")
    pros = wh.table("marts.prospects")[["player_key", "player_name", "position", "pick", "round", "team", "college",
                                         "snap_share_3yr", "starter", "undrafted"]]  # fmt: skip
    wh.close()
    df = pred.merge(pros, on="player_key")
    return df[df["starter"].notna() & (df["prediction_type"] == "walk_forward")]


def undrafted(base_model: str) -> dict:
    """Undrafted starters and where the model ranked them among their class's undrafted invitees.

    Ranked by the out-of-sample base model: the blend for 2017-2019 is fitted in sample.
    """
    wh = Warehouse(read_only=True)
    u = wh.table("marts.backtest")
    wh.close()
    u = u[(u["undrafted"] == 1) & u["starter"].notna()].copy()
    u["p_starter"] = u[f"p_{base_model}"]
    u["model_pct_rank"] = u.groupby("draft_year")["p_starter"].rank(pct=True)
    cols = ["draft_year", "player_name", "position", "college", "p_starter", "model_pct_rank", "snap_share_3yr"]
    hits = u[u["starter"] == 1].sort_values("draft_year")[cols]
    top = u["model_pct_rank"] > 0.9
    return {"n": len(u), "starters": hits.round(3).to_dict("records"),
            "top_decile_snap_share": u.loc[top, "snap_share_3yr"].mean(),
            "rest_snap_share": u.loc[~top, "snap_share_3yr"].mean(),
            "top_decile_any_snaps": (u.loc[top, "snap_share_3yr"] >= 0.1).mean(),
            "rest_any_snaps": (u.loc[~top, "snap_share_3yr"] >= 0.1).mean()}  # fmt: skip


def disagreement_figure(df: pd.DataFrame) -> pd.DataFrame:
    """Starter rate by how much the model disagreed with the draft slot."""
    df = df.assign(bucket=pd.qcut(df["value_over_slot"], 5, labels=["Much lower", "Lower", "Agree", "Higher",
                                                                       "Much higher"]))  # fmt: skip
    t = df.groupby("bucket", observed=True).agg(n=("starter", "size"), slot=("p_pick_only", "mean"),
                                                model=("p_starter", "mean"), actual=("starter", "mean"))  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(t))
    ax.plot(x, t["slot"], color=MARKET, marker="o", linewidth=2, markersize=7, label="Draft slot implied",
            markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.plot(x, t["model"], color=MODEL, marker="o", linewidth=2, markersize=7, label="Model (blend)",
            markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.bar(x, t["actual"], color=MUTED, alpha=0.35, width=0.5, label="Actual starter rate")
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(t.index, t["n"])])
    ax.set_ylabel("P(starter over first 3 seasons)")
    ax.set_xlabel("Model vs. draft slot")
    ax.set_title("2021-2023 classes: when the model disagreed with the slot", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path("figures") / "disagreement.png", dpi=160)
    plt.close(fig)
    return t


def by_round(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("round").agg(n=("starter", "size"), slot=("p_pick_only", "mean"), model=("p_starter", "mean"),
                                   actual=("starter", "mean"))  # fmt: skip


def calls(df: pd.DataFrame, n: int = 8) -> dict[str, pd.DataFrame]:
    cols = ["draft_year", "pick", "player_name", "position", "college", "p_pick_only", "p_starter", "snap_share_3yr"]
    up, down = df.nlargest(40, "value_over_slot"), df.nsmallest(40, "value_over_slot")
    return {
        "upgrade_hits": up[up["starter"] == 1].head(n)[cols],
        "upgrade_misses": up[up["starter"] == 0].head(n)[cols],
        "downgrade_hits": down[down["starter"] == 0].head(n)[cols],
        "downgrade_misses": down[down["starter"] == 1].head(n)[cols],
    }


def class_figure(df: pd.DataFrame, year: int, n: int = 40) -> dict:
    """Slot-implied vs model vs actual starter rate for one class's top-n upgrades and downgrades."""
    d = df[(df["draft_year"] == year) & (df["undrafted"] == 0)]
    groups = {"Model's top 40 upgrades": d.nlargest(n, "value_over_slot"),
              "Model's top 40 downgrades": d.nsmallest(n, "value_over_slot")}  # fmt: skip
    t = pd.DataFrame({k: {"slot": g["p_pick_only"].mean(), "model": g["p_starter"].mean(), "actual": g["starter"].mean()}
                      for k, g in groups.items()}).T  # fmt: skip
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x, w = np.arange(len(t)), 0.26
    for i, (col, color, label) in enumerate([("slot", MARKET, "Draft slot implied"), ("model", MODEL, "Model"),
                                              ("actual", MUTED, "Actually became starters")]):  # fmt: skip
        bars = ax.bar(x + (i - 1) * w, t[col], w * 0.92, color=color, label=label)
        ax.bar_label(bars, [f"{v:.0%}" for v in t[col]], fontsize=9, color=INK, padding=2)
    ax.set_xticks(x, t.index)
    ax.set_ylim(0, max(0.7, t.to_numpy().max() + 0.1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"{year} draft class: did the model's calls hold up?", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path("figures") / f"class_{year}.png", dpi=160)
    plt.close(fig)
    return t.round(3).to_dict("index")


def run() -> dict:
    allwf = load()
    df = allwf[allwf["draft_year"].isin(CLASSES) & (allwf["undrafted"] == 0)]
    out = {"n": len(df), "disagreement": disagreement_figure(df).round(3).reset_index().to_dict("records"),
           "by_round": by_round(df).round(3).reset_index().to_dict("records")}  # fmt: skip
    for k, t in calls(df).items():
        out[k] = t.round(3).to_dict("records")
    up, down = df.nlargest(40, "value_over_slot"), df.nsmallest(40, "value_over_slot")
    out["top40_upgrades"] = {"slot": up["p_pick_only"].mean(), "model": up["p_starter"].mean(),
                             "actual": up["starter"].mean()}  # fmt: skip
    out["top40_downgrades"] = {"slot": down["p_pick_only"].mean(), "model": down["p_starter"].mean(),
                               "actual": down["starter"].mean()}  # fmt: skip
    base = json.loads((path("artifacts") / "metrics.json").read_text())["base_model"]
    out["undrafted"] = undrafted(base)
    out["class_2023"] = class_figure(allwf, 2023)
    (path("artifacts") / "report_2021_2023.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
