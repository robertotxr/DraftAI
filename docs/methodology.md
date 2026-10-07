# DraftAI methodology

Technical detail behind the [README](../README.md). The backtest write-up is in [reports/backtest_2021_2023.md](../reports/backtest_2021_2023.md).

## Architecture

```
src/ingest/    nflverse (nfl_data_py) + CFBD clients, identity resolution, raw -> staging ETL
src/features/  athletic score, opponent-adjusted production, marts.prospects
src/models/    feature allowlists + leakage guard, walk-forward training, evaluation, SHAP, report
src/comps/     kNN comps on position-specific profiles
src/tracking/  Closing Over Expected (BDB 2026), Tackles Over Expected (BDB 2024), play animations
src/db.py      warehouse access layer (DuckDB by default, Postgres via the same interface)
app/           Streamlit: Big Board, Player Card, Tracking, Methodology
tests/         feature transforms, identity matching, leakage checks, comps, tracking
```

The warehouse has three layers. `raw.*` holds the source payloads, `staging.*` holds typed tables with one grain each, and `marts.*` holds model-ready tables. Changing `paths.warehouse_url` to a `postgresql://` URL moves the whole pipeline to Postgres.

## Full backtest table

| Held-out classes 2020–2023 (n = 1,014) | Brier ↓ | Skill vs. base rate ↑ | AUC ↑ | ECE ↓ |
|---|---|---|---|---|
| Draft slot only (baseline) | 0.1294 | 0.271 | 0.828 | 0.036 |
| Athleticism only (baseline) | 0.1683 | 0.052 | 0.664 | 0.038 |
| Scouting profile, no slot | 0.1578 | 0.111 | 0.721 | 0.029 |
| Full model, single (with position interactions) | 0.1301 | 0.267 | 0.824 | 0.025 |
| Full model, one per position group | 0.1377 | 0.224 | 0.798 | 0.036 |
| **Market + model blend (production)** | **0.1283** | **0.277** | **0.833** | 0.042 |


## Modeling decisions

**Target: three-year snap share, not Approximate Value.**
- nflverse exposes AV only as a career total (`w_av`, `dr_av`), so a first-three-years AV cannot be built without leaking later seasons.
- Snap counts are observed per season (2013+), are comparable across positions, and measure what a GM buys with a pick: a player the coaching staff trusts on the field.
- *Starter* = average share of team snaps ≥ 50% over seasons 1–3. Missed seasons count as zero. A second model predicts the 10th/50th/90th percentiles of the share itself.
- Caveat: snap share has a draft-capital feedback loop, since high picks get playing time partly because they are high picks. That favors the slot baseline, which makes beating it a conservative test.

**Strict temporal validation.**
- A class's three-year outcome is only known three drafts later, so class Y is scored by models trained on classes ≤ Y − 3.
- Every downstream step follows the same rule: base-model selection, blend weights and conformal widening.
- [src/models/features.py](../src/models/features.py) holds explicit feature allowlists. `assert_no_leakage` runs before every fit and rejects post-draft columns, unobservable labels, and same- or later-class training rows. Tests cover it.
- Athletic norms and comp scaling come from pre-2017 combines, so every walk-forward class is scored against a population that existed before its draft.

**Models.**
- Draft slot only: per-position logistic regression on log(pick).
- Athleticism only, scouting profile (no slot) and full profile + slot: LightGBM with sigmoid calibration on internal CV folds of the training classes.
- Each LightGBM setup is fit two ways, one model per position group vs. a single model with a position feature. The single model wins because per-group samples are too small.
- The production model is a stacked logistic blend of the slot and full-model log-odds, fitted on earlier out-of-sample predictions only.
- The classifiers learn from drafted players only. Mixing in the ~1%-starter undrafted rows distorted their calibration on drafted players. The blend carries an undrafted indicator, fitted on earlier out-of-sample predictions, which sets the probability level for undrafted invitees.
- Intervals use cross-conformal quantile regression. Leave-classes-out residuals within the training classes set the widening, separately for round 1, rounds 2–3, rounds 4–7 and undrafted players. An earlier version calibrated on residuals of earlier walk-forward models, which were trained on far fewer classes, and over-covered (86%).
- SHAP values come from a LightGBM fitted on all labeled drafted classes.

**Features.**
- *Athletic* ([athletic.py](../src/features/athletic.py)): RAS-style 0–10 percentiles within position group for 40, vertical, broad, 3-cone, shuttle and bench. Each drill is size-adjusted by regressing it on height and weight. Missing drills stay missing, with explicit `_measured` flags. No composite is produced from fewer than 3 drills. The public combine data has no 10-yard split, so it is not used.
- *Production* ([production.py](../src/features/production.py)):
  - Within-team market shares: receiving yards/TD share, dominator rating, scrimmage share, sack + TFL share, pass-defensed share.
  - QB efficiency.
  - Pass-play usage share, the public proxy for target share.
  - Pressure proxy: (sacks + hurries) per game.
  - Opponent adjustment from the points allowed and scored by the opponents faced (score-based, so not confounded by the player's own talent).
  - Age on September 1 of each season, and breakout age with explicit "never broke out" flags.
  - Final, best and career aggregates, built strictly from pre-draft seasons.
- *Identity* ([identity.py](../src/ingest/identity.py)):
  - An nflverse pick (class, overall pick) joins the CFBD draft table, which carries the CFBD college athlete ID. The ID is accepted only if the names agree. Otherwise the code falls back to normalized name + school.
  - Results: 2,832 ID matches, 77 name-based matches, 5 ambiguous cases left unmatched on purpose, and 612 unmatched.
  - Match rate is 82.5% of picks overall and 91% excluding offensive linemen, who have no box-score stats.
  - Undrafted combine invitees go through the name + school path (80.5% matched). When the combine feed lacks their PFR id (needed for NFL snaps), it is recovered from rosters. Roster birth dates are dropped for them: a birth date only exists if the player later made a roster, which is post-draft information.

**Comps.** NaN-aware kNN on standardized, position-specific profiles (body + athletic + production). Comps for class Y come only from classes whose outcome was known by then, so every comp shows a real result.

**Tracking metric.** Closing Over Expected (COE) on BDB 2026: how close a coverage defender gets to the targeted receiver by ball arrival versus an out-of-fold LightGBM expectation built from throw-frame geometry (RMSE 2.08 yd vs. 4.09 for a linear baseline; split-half r 0.30; completion is 72-74% in the two lowest COE quintiles and 60% in the highest; no detectable link to combine athleticism, n about 200). The older Tackles Over Expected on BDB 2024 is kept but cannot run without a local copy.

**Tackles Over Expected (BDB 2024).** For each defender-play, a LightGBM model (GroupKFold by game) scores the tackle probability at the first frame the defender closes within 5 yards of the ball carrier. Features include distance, closing speed, pursuit angle to the projected intercept point, blockers in the lane and sideline leverage. TOE is tackles + assists minus that expectation. Validation covers odd/even-week stability and correlation with PFF missed-tackle rate. See [src/tracking/README.md](../src/tracking/README.md).

## Known limitations

- Undrafted players are covered only if they were invited to the combine. With 6 undrafted starters in seven classes, their probabilities are rough and they have no slot baseline.
- CFBD defensive season stats begin in 2016, so defensive production is missing for most early training classes. Offensive linemen have no production stats.
- No pro-day data, 10-yard splits, route or target data (YPRR), medicals or interviews.
- The market + model gain is not statistically significant on four held-out classes. The blend is slightly less calibrated (ECE 0.042) than the full model (0.025).
- Closing Over Expected runs on real BDB 2026 data (2023 only, one season, modest split-half stability). Tackles Over Expected is unit-tested on synthetic plays only: the BDB 2024 files are no longer downloadable.

## Data and caching

- `make bdb` downloads the Big Data Bowl 2026 Analytics data into `data/raw/bdb26/` (Kaggle credentials and accepted rules required). The 2024 files behind Tackles Over Expected were taken down in August 2025; with a local copy in `data/raw/bdb/` that metric still runs, otherwise it logs a warning and is skipped.
- The pipeline is idempotent and cached. nflverse pulls are stored as parquet and every CFBD response as JSON under `data/raw/`, so a rerun makes zero API calls.
- CFBD usage is logged to `data/raw/cfbd/_call_log.csv` from the `X-CallLimit-Remaining` header. A full build uses about 66 calls of the 1,000/month free tier, and the client refuses to call when fewer than 50 remain.
- Settings live in [config/config.yaml](../config/config.yaml): paths, seasons, thresholds, model parameters, seed.

