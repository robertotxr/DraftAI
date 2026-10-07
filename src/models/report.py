"""Figures and tables for reports/backtest_2021_2023.md (as-of-draft-night predictions only)."""

from __future__ import annotations

import json

import matplotlib

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
                                         "snap_share_3yr", "starter"]]  # fmt: skip
    wh.close()
    df = pred.merge(pros, on="player_key")
    return df[df["draft_year"].isin(CLASSES) & df["starter"].notna() & (df["prediction_type"] == "walk_forward")]


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


def run() -> dict:
    df = load()
    out = {"n": len(df), "disagreement": disagreement_figure(df).round(3).reset_index().to_dict("records"),
           "by_round": by_round(df).round(3).reset_index().to_dict("records")}  # fmt: skip
    for k, t in calls(df).items():
        out[k] = t.round(3).to_dict("records")
    up, down = df.nlargest(40, "value_over_slot"), df.nsmallest(40, "value_over_slot")
    out["top40_upgrades"] = {"slot": up["p_pick_only"].mean(), "model": up["p_starter"].mean(),
                             "actual": up["starter"].mean()}  # fmt: skip
    out["top40_downgrades"] = {"slot": down["p_pick_only"].mean(), "model": down["p_starter"].mean(),
                               "actual": down["starter"].mean()}  # fmt: skip
    (path("artifacts") / "report_2021_2023.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
