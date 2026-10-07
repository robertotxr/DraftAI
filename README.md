# DraftAI

**Who's going to start in the NFL? And when is the draft board wrong?**

![DraftAI home](docs/home.png)

![Python 3.12](https://img.shields.io/badge/python-3.12-2a78d6) ![Tests](https://img.shields.io/badge/tests-44%20passing-1baf7a) ![Data](https://img.shields.io/badge/data-nflverse%20·%20CFBD%20·%20Big%20Data%20Bowl-eb6834)

## What it does

- **Grades every prospect.** Every pick since 2013, plus undrafted combine guys. Each one gets a shot at becoming a starter.
- **Calls out the board.** It shows who the model likes more than his draft slot. And who it likes less.
- **Finds comps.** Similar players from past drafts, and how they actually turned out.
- **Reads tracking data.** Which defenders close on the ball faster than they should.

## The scoreboard

Every number below comes from drafts the model never saw. It only knew what teams knew on draft night.

| | |
|---|---|
| **63%** | The model's 40 favorite value picks became starters 63% of the time. Their draft slot said 52%. |
| **5 of 6** | Undrafted guys who became starters since 2017. Five were in the model's top 16%. |
| **0.833** | Ranking accuracy (AUC). Draft slot alone gets 0.828. Close. The edge is in the disagreements. |
| **83%** | Outcomes that landed inside the model's 80% range. It knows what it doesn't know. |

![When the model disagreed with the draft slot](reports/figures/disagreement.png)

## Inside the app

| Big Board | Player Card |
|---|---|
| ![Big Board](docs/big_board.png) | ![Player Card](docs/player_card.png) |
| Every class, ranked by chance to start. Next to what the slot says. | Testing, college stats, what's driving the grade, comps. |

| Tracking | |
|---|---|
| ![Closing Over Expected](docs/tracking.png) | **Closing Over Expected.** 2023 tracking data. Who closes on the receiver faster than expected while the ball is in the air? It matters. When the nearest defender closes best, 60% of passes get caught. When he closes worst, 72%. |

## How it works

1. **Collect.** Draft, combine and snap counts from nflverse. College stats from CollegeFootballData. All cached.
2. **Profile.** Athletic scores adjusted for size. College production adjusted for opponents and age.
3. **Predict.** Models trained only on past drafts. Blended with the draft slot. Plus an 80% range.
4. **Compare.** The five closest comps, with real outcomes.
5. **Track.** A coverage stat built from Big Data Bowl player tracking.

**Read the story:** [The 2023 draft, three years later](reports/draft_2023.md). One page. Hits, misses, and Puka Nacua.

Want the deep dive? Here's the [full backtest](reports/backtest_2021_2023.md) and the [methodology](docs/methodology.md).

## Run it

```bash
make setup             # Python 3.12 venv + pinned requirements (macOS: brew install libomp)
cp .env.example .env   # add CFBD_API_KEY and KAGGLE_API_TOKEN
make all               # build data, models, comps, tracking and report
make app               # open the app at localhost:8501
```

## Honest limits

- Teams play the guys they drafted high. So playing time favors early picks. That makes the slot tough to beat.
- No pro days, medicals, interviews or route data. That's where scouts still win.
- College defensive stats start in 2016. Offensive linemen have no stats at all.
- Tracking covers one NFL season.
