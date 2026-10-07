"""NFL outcome models: walk-forward backtest, final fit, intervals and SHAP.

Target: `starter` = average share of team snaps over the first three NFL seasons >= 50%.
Secondary: quantiles of `snap_share_3yr` for an outcome range.

Temporal validation: to score draft class Y we train only on classes whose three-season outcome
was observable before the Y draft (class <= Y - 3). No random splits anywhere.
"""

from __future__ import annotations

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.config import cfg, path
from src.db import Warehouse
from src.models import evaluate
from src.models.features import FEATURE_SETS, assert_no_leakage

log = logging.getLogger(__name__)

# name -> (feature set, learner, per position group?)
SPECS = {
    "pick_only": ("pick_only", "logit", True),
    "athletic_only": ("athletic_only", "lgbm", False),
    "scouting_single": ("scouting", "lgbm", False),
    "scouting_per_group": ("scouting", "lgbm", True),
    "full_single": ("full", "lgbm", False),
    "full_per_group": ("full", "lgbm", True),
}


def _lgbm(**overrides):
    params = {**cfg()["model"]["lgbm"], "random_state": cfg()["seed"], **overrides}
    return lgb.LGBMClassifier(**params)


def _design(df: pd.DataFrame, features: list[str], with_group: bool) -> pd.DataFrame:
    X = df[features].astype(float).copy()
    if with_group:
        X["pos_group"] = pd.Categorical(df["pos_group"], categories=sorted(cfg()["positions"]["groups"]))
    return X


def _fit_one(train: pd.DataFrame, features: list[str], learner: str, with_group: bool):
    y = train["starter"].astype(int)
    if learner == "logit":
        X = train[features].astype(float).fillna(train[features].median())
        return LogisticRegression(C=1.0).fit(X, y)
    X = _design(train, features, with_group)
    # Sigmoid calibration on internal CV folds of the training classes only. Small position groups
    # may have few starters, so folds shrink with the minority class (and calibration is skipped below 2).
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        return _lgbm().fit(X, y)
    return CalibratedClassifierCV(_lgbm(), method="sigmoid", cv=folds).fit(X, y)


def _predict_one(model, df: pd.DataFrame, features: list[str], learner: str, with_group: bool) -> np.ndarray:
    if learner == "logit":
        X = df[features].astype(float).fillna(df[features].median())
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(_design(df, features, with_group))[:, 1]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec: str, temporal_check: bool = True) -> np.ndarray:
    """Fit `spec` on train, score test. temporal_check=False only for explicitly in-sample scoring."""
    fset, learner, per_group = SPECS[spec]
    features = FEATURE_SETS[fset]
    assert_no_leakage(train, test if temporal_check else test.iloc[:0], features)
    if not per_group:
        model = _fit_one(train, features, learner, with_group=True)
        return _predict_one(model, test, features, learner, with_group=True)
    out = np.full(len(test), np.nan)
    for g, idx in test.groupby("pos_group").groups.items():
        tr = train[train["pos_group"] == g]
        model = _fit_one(tr, features, learner, with_group=False)
        mask = test.index.isin(idx)
        out[mask] = _predict_one(model, test.loc[idx], features, learner, with_group=False)
    return out


def fit_quantiles(train: pd.DataFrame, features: list[str]) -> dict[float, lgb.LGBMRegressor]:
    X = _design(train, features, True)
    models = {}
    for q in cfg()["target"]["quantiles"]:
        params = {**cfg()["model"]["lgbm"], "random_state": cfg()["seed"], "objective": "quantile", "alpha": q}
        models[q] = lgb.LGBMRegressor(**params).fit(X, train["snap_share_3yr"])
    return models


def predict_quantiles(models: dict, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = _design(df, features, True)
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return np.sort(np.clip(preds, 0, 1), axis=1)  # enforce non-crossing quantiles


def walk_forward(pros: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """As-of-draft-night predictions: class Y is scored by models trained on classes <= Y - label_lag."""
    lag = cfg()["model"]["label_lag"]
    labeled = pros[pros["starter"].notna()]
    rows = []
    for year in years:
        train = labeled[labeled["draft_year"] <= year - lag]
        test = pros[pros["draft_year"] == year].copy()
        log.info("class %d: train %d rows (%d-%d), score %d", year, len(train), train.draft_year.min(),
                 train.draft_year.max(), len(test))  # fmt: skip
        for spec in SPECS:
            test[f"p_{spec}"] = fit_predict(train, test, spec)
        q = fit_quantiles(train, FEATURE_SETS["full"])
        test[["q10", "q50", "q90"]] = predict_quantiles(q, test, FEATURE_SETS["full"])
        rows.append(test)
    return pd.concat(rows)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack(wf: pd.DataFrame, model_spec: str) -> tuple[pd.Series, dict]:
    """Market + model blend: a logistic regression on the logits of the slot-only and model probabilities.

    For class Y it is fitted only on earlier walk-forward (out-of-sample) predictions whose labels were
    observable before the Y draft, so the blend weights are as honest as the predictions they combine.
    """
    lag = cfg()["model"]["label_lag"]
    cols = ["p_pick_only", f"p_{model_spec}"]
    out, weights = pd.Series(np.nan, index=wf.index), {}
    for year, te in wf.groupby("draft_year"):
        tr = wf[(wf["draft_year"] <= year - lag) & wf["starter"].notna()]
        if tr.empty:
            continue
        m = LogisticRegression().fit(_logit(tr[cols]), tr["starter"].astype(int))
        out[te.index] = m.predict_proba(_logit(te[cols]))[:, 1]
        weights[int(year)] = {"slot": round(float(m.coef_[0][0]), 3), "model": round(float(m.coef_[0][1]), 3)}
    return out, weights


def confidence_grade(df: pd.DataFrame, width_cuts: tuple[float, float]) -> pd.Series:
    """High / Medium / Low from data completeness and outcome-interval width."""
    complete = (
        df["athletic_score"].notna().astype(int)
        + df["college_seasons"].notna().astype(int)
        + df["consensus_rank"].notna().astype(int)
    ) / 3
    width = df["q90"] - df["q10"]
    grade = np.where((complete >= 2 / 3) & (width <= width_cuts[0]), "High",
                     np.where((complete < 1 / 3) | (width > width_cuts[1]), "Low", "Medium"))  # fmt: skip
    return pd.Series(grade, index=df.index)


def run() -> None:
    c = cfg()
    wh = Warehouse()
    pros = wh.table("marts.prospects")
    art = path("artifacts")
    labeled = pros[pros["starter"].notna()]

    # 1) Walk-forward: every class that has at least two earlier labeled classes to learn from.
    years = list(range(c["model"]["walk_forward_from"], c["years"]["draft_last"] + 1))
    wf = walk_forward(pros, years)
    # Pick the base model on the classes before the headline backtest, then blend it with the market.
    pre = wf[(wf["draft_year"] < min(c["model"]["backtest_classes"])) & wf["starter"].notna()]
    best = min(
        ["full_single", "full_per_group"],
        key=lambda s: evaluate.score(pre["starter"].to_numpy(int), pre[f"p_{s}"].to_numpy())["log_loss"],
    )
    wf["p_blend"], blend_weights = stack(wf, best)
    bt = wf[wf["draft_year"].isin(c["model"]["backtest_classes"]) & wf["starter"].notna()]
    specs = [*SPECS, "blend"]
    metrics = evaluate.summarize(bt, specs)
    metrics["base_model"], metrics["chosen_model"], metrics["blend_weights"] = best, "blend", blend_weights
    log.info("backtest metrics:\n%s", pd.DataFrame(metrics["models"]).T.round(4).to_string())
    log.info("base model: %s, blend weights: %s", best, blend_weights)

    # 2) Final model on every labeled class: used for SHAP, and in-sample scores for the first classes.
    feats = FEATURE_SETS[SPECS[best][0]]
    early = pros[pros["draft_year"] < c["model"]["walk_forward_from"]]
    p_early = fit_predict(labeled, early, best, temporal_check=False)
    p_early_pick = fit_predict(labeled, early, "pick_only", temporal_check=False)
    q_early = predict_quantiles(fit_quantiles(labeled, FEATURE_SETS["full"]), early, FEATURE_SETS["full"])

    X_all = _design(pros, feats, True)
    explain_model = _lgbm().fit(_design(labeled, feats, True), labeled["starter"].astype(int))
    sv = shap.TreeExplainer(explain_model).shap_values(X_all)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_df = pd.DataFrame(sv, columns=X_all.columns)
    shap_df.insert(0, "player_key", pros["player_key"].to_numpy())
    wh.write("marts.shap_values", shap_df)
    global_imp = shap_df.drop(columns="player_key").abs().mean().sort_values(ascending=False)
    global_imp.rename("mean_abs_shap").rename_axis("feature").reset_index().to_csv(art / "shap_global.csv", index=False)

    # 3) Predictions table
    cols = ["player_key", "draft_year", "pos_group"]
    a = wf[cols].assign(p_starter=wf["p_blend"].fillna(wf[f"p_{best}"]).to_numpy(), p_pick_only=wf["p_pick_only"].to_numpy(),
                        q10=wf["q10"].to_numpy(), q50=wf["q50"].to_numpy(), q90=wf["q90"].to_numpy(),
                        prediction_type="walk_forward")  # fmt: skip
    b = early[cols].assign(p_starter=p_early, p_pick_only=p_early_pick, q10=q_early[:, 0], q50=q_early[:, 1],
                           q90=q_early[:, 2], prediction_type="in_sample")  # fmt: skip
    pred = pd.concat([b, a], ignore_index=True)
    w = pred["q90"] - pred["q10"]
    ctx = pred[["player_key"]].merge(pros[["player_key", "athletic_score", "college_seasons", "consensus_rank"]])
    pred["confidence"] = confidence_grade(pd.concat([ctx, pred[["q10", "q90"]]], axis=1),
                                          (w.quantile(0.4), w.quantile(0.8)))  # fmt: skip
    pred["value_over_slot"] = pred["p_starter"] - pred["p_pick_only"]
    wh.write("marts.predictions", pred)
    wh.write("marts.backtest", wf)

    joblib.dump({"model": explain_model, "features": feats}, art / "explain_model.joblib")
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    evaluate.plots(bt, specs, "blend", path("figures"))
    log.info("predictions for %d players", len(pred))
    wh.close()
