# DraftAI: NFL draft prospect evaluation

An end-to-end prospect evaluation system: data ingestion → athletic and production features → calibrated outcome models → historical comps → a tracking-data tackling metric → an interactive big board.

**Headline result (2020–2023 held-out classes, scored as of draft night).** A market + model blend predicts which picks become starters slightly better than draft slot alone: Brier 0.1284 vs. 0.1294, AUC 0.833 vs. 0.828. The 95% CI on that gain includes zero, so the honest claim is "matches the market and adds a small edge." The model's disagreements with the slot carry signal. Its 40 biggest upgrades in 2021–2023 had a 52% slot-implied starter rate, and 68% became starters. Full write-up: [reports/backtest_2021_2023.md](reports/backtest_2021_2023.md).

![Big board](docs/big_board.png)

| Held-out classes 2020–2023 (n = 1,014) | Brier ↓ | Skill vs. base rate ↑ | AUC ↑ | ECE ↓ |
|---|---|---|---|---|
| Draft slot only (baseline) | 0.1294 | 0.271 | 0.828 | 0.036 |
| Athleticism only (baseline) | 0.1664 | 0.063 | 0.679 | 0.036 |
| Scouting profile, no slot | 0.1575 | 0.113 | 0.721 | 0.024 |
| Full model, single (with position interactions) | 0.1298 | 0.269 | 0.824 | 0.019 |
| Full model, one per position group | 0.1379 | 0.223 | 0.797 | 0.033 |
| **Market + model blend (production)** | **0.1284** | **0.277** | **0.833** | 0.044 |

The 80% conformal intervals for three-year snap share achieved 86% coverage.

## Reproduce

```bash
make setup                 # Python 3.12 venv + pinned requirements (macOS: brew install libomp for LightGBM)
cp .env.example .env       # fill CFBD_API_KEY and KAGGLE_API_TOKEN
make all                   # ingest -> features -> train -> comps -> tracking -> report
make test lint
make app                   # streamlit big board at localhost:8501
```

- `make bdb` downloads the Big Data Bowl 2024 data. Kaggle returns 403 until you accept the [competition rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules). Until then the tracking step logs a warning and the app shows a notice.
- The pipeline is idempotent and cached. nflverse pulls are stored as parquet and every CFBD response as JSON under `data/raw/`, so a rerun makes zero API calls.
- CFBD usage is logged to `data/raw/cfbd/_call_log.csv` from the `X-CallLimit-Remaining` header. A full build uses about 66 calls of the 1,000/month free tier, and the client refuses to call when fewer than 50 remain.
- Settings live in [config/config.yaml](config/config.yaml): paths, seasons, thresholds, model parameters, seed.

## Architecture

```
src/ingest/    nflverse (nfl_data_py) + CFBD clients, identity resolution, raw -> staging ETL
src/features/  athletic score, opponent-adjusted production, marts.prospects
src/models/    feature allowlists + leakage guard, walk-forward training, evaluation, SHAP, report
src/comps/     kNN comps on position-specific profiles
src/tracking/  BDB 2024 tackle-opportunity model, Tackles Over Expected, play animation
src/db.py      warehouse access layer (DuckDB by default, Postgres via the same interface)
app/           Streamlit: Big Board, Player Card, Tracking, Methodology
tests/         feature transforms, identity matching, leakage checks, comps, tracking
```

The warehouse has three layers. `raw.*` holds the source payloads, `staging.*` holds typed tables with one grain each, and `marts.*` holds model-ready tables. Changing `paths.warehouse_url` to a `postgresql://` URL moves the whole pipeline to Postgres.

## Modeling decisions

**Target: three-year snap share, not Approximate Value.**
- nflverse exposes AV only as a career total (`w_av`, `dr_av`), so a first-three-years AV cannot be built without leaking later seasons.
- Snap counts are observed per season (2013+), are comparable across positions, and measure what a GM buys with a pick: a player the coaching staff trusts on the field.
- *Starter* = average share of team snaps ≥ 50% over seasons 1–3. Missed seasons count as zero. A second model predicts the 10th/50th/90th percentiles of the share itself.
- Caveat: snap share has a draft-capital feedback loop, since high picks get playing time partly because they are high picks. That favors the slot baseline, which makes beating it a conservative test.

**Strict temporal validation.**
- A class's three-year outcome is only known three drafts later, so class Y is scored by models trained on classes ≤ Y − 3.
- Every downstream step follows the same rule: base-model selection, blend weights and conformal widening.
- [src/models/features.py](src/models/features.py) holds explicit feature allowlists. `assert_no_leakage` runs before every fit and rejects post-draft columns, unobservable labels, and same- or later-class training rows. Tests cover it.
- Athletic norms and comp scaling come from pre-2017 combines, so every walk-forward class is scored against a population that existed before its draft.

**Models.**
- Draft slot only: per-position logistic regression on log(pick).
- Athleticism only, scouting profile (no slot) and full profile + slot: LightGBM with sigmoid calibration on internal CV folds of the training classes.
- Each LightGBM setup is fit two ways, one model per position group vs. a single model with a position feature. The single model wins because per-group samples are too small.
- The production model is a stacked logistic blend of the slot and full-model log-odds, fitted on earlier out-of-sample predictions only.
- Intervals use conformalized quantile regression. SHAP values come from a LightGBM fitted on all labeled classes.

**Features.**
- *Athletic* ([athletic.py](src/features/athletic.py)): RAS-style 0–10 percentiles within position group for 40, vertical, broad, 3-cone, shuttle and bench. Each drill is size-adjusted by regressing it on height and weight. Missing drills stay missing, with explicit `_measured` flags. No composite is produced from fewer than 3 drills. The public combine data has no 10-yard split, so it is not used.
- *Production* ([production.py](src/features/production.py)):
  - Within-team market shares: receiving yards/TD share, dominator rating, scrimmage share, sack + TFL share, pass-defensed share.
  - QB efficiency.
  - Pass-play usage share, the public proxy for target share.
  - Pressure proxy: (sacks + hurries) per game.
  - Opponent adjustment from the points allowed and scored by the opponents faced (score-based, so not confounded by the player's own talent).
  - Age on September 1 of each season, and breakout age with explicit "never broke out" flags.
  - Final, best and career aggregates, built strictly from pre-draft seasons.
- *Identity* ([identity.py](src/ingest/identity.py)):
  - An nflverse pick (class, overall pick) joins the CFBD draft table, which carries the CFBD college athlete ID. The ID is accepted only if the names agree. Otherwise the code falls back to normalized name + school.
  - Results: 2,832 ID matches, 77 name-based matches, 5 ambiguous cases left unmatched on purpose, and 612 unmatched.
  - Match rate is 82.5% of picks overall and 91% excluding offensive linemen, who have no box-score stats.

**Comps.** NaN-aware kNN on standardized, position-specific profiles (body + athletic + production). Comps for class Y come only from classes whose outcome was known by then, so every comp shows a real result.

**Tracking metric.** Tackles Over Expected on BDB 2024. For each defender-play, a LightGBM model (GroupKFold by game) scores the tackle probability at the first frame the defender closes within 5 yards of the ball carrier. Features include distance, closing speed, pursuit angle to the projected intercept point, blockers in the lane and sideline leverage. TOE is tackles + assists minus that expectation. Validation covers odd/even-week stability and correlation with PFF missed-tackle rate. See [src/tracking/README.md](src/tracking/README.md).

## Known limitations

- Drafted players only. UDFAs have no slot baseline.
- CFBD defensive season stats begin in 2016, so defensive production is missing for most early training classes. Offensive linemen have no production stats.
- No pro-day data, 10-yard splits, route or target data (YPRR), medicals or interviews.
- The market + model gain is not statistically significant on four held-out classes. The blend is slightly less calibrated (ECE 0.044) than the full model (0.019).
- The tracking metric is implemented and unit-tested on synthetic plays, but it has not been run on the real BDB data until the Kaggle rules are accepted.

## Next steps

1. Accept the BDB rules, run `make bdb tracking`, then link TOE to college defenders who later appear in the tracking data.
2. Add pro-day results and PFF-style charting data (targets, routes, pressures) when a licensed source is available.
3. Model second-contract or AV-per-season outcomes once enough seasons accumulate. Add UDFAs with a "pick 260+" slot proxy.
4. Use a cross-conformal calibration set to tighten intervals toward nominal coverage.
