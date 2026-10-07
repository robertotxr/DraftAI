# Tackles Over Expected (NFL Big Data Bowl 2024 tracking)

Original tackling metric on BDB 2024 tracking data, built so it can become a BDB-style submission.

## Data note

The BDB 2024 data is not in the repo (competition rules forbid redistribution). In August 2025 the NFL
replaced the competition files on Kaggle with a 78-byte README, so they can no longer be downloaded.
With a local copy, put the CSVs (games, plays, players, tackles, tracking_week_1..9) into `data/raw/bdb/`.
`python -m src.tracking.run` logs a warning and exits cleanly while the files are missing.

## Metric

- Standardize: flip left-moving plays so offense always goes left to right.
- Window: frames from handoff / catch / run until tackle, out of bounds, touchdown or fumble.
- Decision point: per defender-play, the first frame inside `opportunity_radius` of the carrier (or the first window
  frame if already inside). Defenders who never get that close are not opportunities.
- Model (LightGBM, GroupKFold by game) trained on decision-point rows only: features as of that frame (distance,
  closing speed, relative speed, pursuit angle vs. the carrier position projected `intercept_seconds` ahead, carrier
  speed, defender speed and acceleration, blockers closer to the carrier, distance to sideline), label = tackle or
  assist on the play. Nothing after the decision enters the features, and expected is a calibrated out-of-fold probability.
- The same model applied to every frame is the animation's "tackle probability if evaluated now".
- Tackles Over Expected (TOE) = tackles + assists minus expected, per player. Reported per 100 opportunities.
- Pursuit angle efficiency = share of defender speed aimed at the projected intercept point, while near the carrier.

## Validation plan

- Calibration of the decision-point model on held-out games (Brier, ECE, reliability plot).
- Stability: TOE per 100 in odd weeks vs. even weeks, per player (Pearson and Spearman-Brown).
- Validity: TOE per 100 vs. PFF missed-tackle rate (expected negative).
- Figures go to `reports/figures/tracking_*.png`, numbers to `artifacts/tracking/validation.json`.

## Mapping to a BDB submission

- Metric definition and tackle-opportunity model: the core of the write-up.
- Out-of-fold predictions guarantee no leakage across games.
- `animate_play` gives the play-level visuals; `leaderboard.parquet` gives the player table.
- Still to add for a submission: the 2024 competition framing (tackle-focused, week 1-9 only), a lateral-aware
  carrier definition, and uncertainty intervals per player.

## Outputs (`artifacts/tracking/`)

`animation_frames.parquet`, `leaderboard.parquet`, `validation.json`.

# Closing Over Expected (NFL Big Data Bowl 2026 Analytics)

The BDB 2024 files are gone from Kaggle, so the metric that actually runs is Closing Over Expected (COE) on BDB 2026
Analytics: 2023 regular-season pass plays, tracking before the throw (input) and while the ball is in the air (output).
Data lives in `data/raw/bdb26/` (`make bdb`, needs Kaggle credentials and accepting the rules); `make tracking` runs both
metrics, and each skips with a warning when its data is missing. Code: `closing.py` (pipeline), `closing_validate.py`
(validation, draft link), `closing_animate.py` (play viewer). Artifacts: `artifacts/tracking_closing/`
(`defender_plays`, `leaderboard`, `draft_link`, `animation_frames`, `animation_plays` parquet and `validation.json`).

## Method

- Standardize: left-moving plays are rotated 180 degrees (x, y, dir, o, landing point), via `data.standardize`.
- Unit: one (play, tracked coverage defender), 31,937 defender-plays on 12,966 plays, 716 defenders.
- Features: throw frame only. Defender-receiver offsets and distance, distance to the landing point, receiver distance to
  it, air time, defender speed, acceleration, velocity toward the landing point and the receiver, heading angle to the
  landing point, receiver velocity, man/zone, pass length, position. No output-frame, outcome or EPA data.
- Target: defender-to-receiver distance at ball arrival (final output frame). Expected = out-of-fold LightGBM
  (GroupKFold by game). COE = expected minus actual end distance (yards, positive = closed more than expected).
- Leaderboard: mean and total COE per defender with a standard error; at least `closing.min_plays` (40) plays.

## Validation (all 18 weeks)

- Model error on end distance (RMSE / MAE, yards): LightGBM 2.08 / 1.40; linear on start distance, air time and landing
  geometry 4.09 / 3.02; linear on start distance and air time 4.36 / 3.18; end = start 4.90 / 3.43.
- Split-half (odd vs. even weeks, 20+ plays per half, 267 players): r = 0.30, Spearman-Brown 0.47. Real but modest
  signal: one season is a noisy read on a single defender.
- Outcome relevance (defender closest to the receiver at the throw, 12,966 plays): completion rate by COE quintile
  71.8%, 74.1%, 65.6%, 63.0%, 60.5%; mean EPA 0.40, 0.31, 0.15, 0.09, 0.16. Logistic regression of completion on COE with
  start and expected distance as controls: odds ratio 0.86 per +1 yd COE (coefficient -0.147, Wald p about 1e-22, plays
  treated as independent).
- Draft link (exploratory): 568 of 716 defenders matched to warehouse prospects (79%; 499 by name and birth date, 69 by
  name and unique position group; 85% of those with 40+ plays). Spearman of mean COE (40+ plays) with speed score -0.09
  (n=195), athletic score 0.00 (180), agility score +0.03 (108), 40-yard dash 0.00 (197), draft pick +0.07 (236),
  model P(starter) 0.00 (256). Nothing distinguishable from zero.

## Limitations

- The landing point and air time are known at release but describe the throw itself, so COE is conditional on the
  throw. It credits closing given the ball, not coverage that prevented the throw.
- Only defenders BDB flags (`player_to_predict`) are scored; one season; no adjustment for receiver quality or route.
- Man vs. zone is the only scheme information; assignments are not modeled.
- The outcome relevance is correlational: COE partly reflects where the receiver and ball end up, not only the defender.
- Draft-link matches rely on names and birth dates; mismatches are possible, and most matches are 2017+ draftees.
