"""Feature allowlists. Anything not listed here never reaches a model.

Every feature must be knowable on draft night. Post-draft columns (NFL outcomes, career AV,
games played) are listed in POST_DRAFT and checked by `assert_no_leakage`.
"""

from __future__ import annotations

import pandas as pd

from src.features.production import CAREER_METRICS

ATHLETIC = [
    "ht_in", "wt_lb", "ht_in_score", "wt_lb_score",
    "forty_size_score", "vertical_size_score", "broad_jump_size_score", "cone_size_score",
    "shuttle_size_score", "bench_size_score",
    "athletic_score", "speed_score", "explosion_score", "agility_score",
    "drills_measured", "attended_combine",
]  # fmt: skip

PRODUCTION = (
    [f"{m}_{agg}" for m in CAREER_METRICS for agg in ("final", "best", "career")]
    + ["breakout_age_rec", "breakout_age_scrim", "breakout_age_def",
       "never_broke_out_rec", "never_broke_out_scrim", "never_broke_out_def",
       "college_seasons", "has_def_stats", "final_season_age"]
)  # fmt: skip

CONTEXT = ["draft_age", "power_conf"]
MARKET = ["log_pick", "undrafted"]  # undrafted players get log_pick of the slot just past the draft

FEATURE_SETS = {
    "pick_only": MARKET,
    "athletic_only": ATHLETIC,
    "scouting": ATHLETIC + PRODUCTION + CONTEXT,  # what the model sees without the draft slot
    "full": ATHLETIC + PRODUCTION + CONTEXT + MARKET,  # does the model add value on top of the market?
}

POST_DRAFT = [
    "w_av", "dr_av", "games", "snap_share_y1", "snap_share_y2", "snap_share_y3", "snap_share_3yr", "starter",
]  # fmt: skip


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> None:
    """Fail loudly if a model could see post-draft information."""
    allowed = set(ATHLETIC + PRODUCTION + CONTEXT + MARKET + ["pos_group"])
    bad = [f for f in features if f in POST_DRAFT or f not in allowed]
    if bad:
        raise AssertionError(f"Disallowed features: {bad}")
    if len(test):
        test_year = test["draft_year"].min()
        if (train["label_known_from_draft"] > test_year).any():
            raise AssertionError("Training labels that were not yet observable at the test draft")
        if (train["draft_year"] >= test_year).any():
            raise AssertionError("Training rows from the test class or later")
