# Tackles Over Expected (NFL Big Data Bowl 2024 tracking)

Original tackling metric on BDB 2024 tracking data, built so it can become a BDB-style submission.

## Data note

The BDB 2024 data is not in the repo. Kaggle returns 403 until the competition rules are accepted at
https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules. Then download the files
(games, plays, players, tackles, tracking_week_1..9 as CSV) into `data/raw/bdb/`.
`python -m src.tracking.run` logs a warning and exits cleanly while the files are missing.

## Metric

- Standardize: flip left-moving plays so offense always goes left to right.
- Window: frames from handoff / catch / run until tackle, out of bounds, touchdown or fumble.
- Frame model (LightGBM, GroupKFold by game): P(defender gets a tackle or assist and touches the carrier within
  `frames_ahead` frames). Features: distance, closing speed, relative speed, pursuit angle (heading vs. the carrier
  position projected `intercept_seconds` ahead), carrier speed, defender speed and acceleration, blockers closer to the
  carrier than the defender, carrier distance to sideline.
- Expected tackle per defender-play = peak frame probability, among plays where the defender got within
  `opportunity_radius` of the carrier, rescaled so league-wide expected equals actual.
- Tackles Over Expected (TOE) = tackles + assists minus expected, per player. Reported per 100 opportunities.
- Pursuit angle efficiency = share of defender speed aimed at the projected intercept point, while near the carrier.

## Validation plan

- Calibration of the frame model on held-out games (Brier, ECE, reliability plot).
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
