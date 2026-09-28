#python solution.py (валидация + обучение на всём train + submission.csv)
#python solution.py --no-cv (только обучение и submission.csv)

from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_features, load_data
from metric import precision_at_recall

SEEDS = [0, 1, 2, 3, 4]
PARAMS = dict(
    objective="binary", learning_rate=0.03, n_estimators=500, num_leaves=15,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, n_jobs=4, verbose=-1,
)
# тест лежит позже train по времени (20–26 апр), поэтому учимся на всех днях до начала фолда, проверяемся на следующих днях
FOLDS = [("2026-04-13", "2026-04-15"), ("2026-04-15", "2026-04-17"), ("2026-04-17", "2026-04-20")]
BASELINE_COLS = ["n_events", "item_nunique"]
POPULARITY_PREFIXES = ("item_pop", "item_shared", "max_items_shared", "n_cookies_sharing")


def fit_predict(X_fit, y_fit, X_pred, seeds=SEEDS) -> np.ndarray:
    # среднее предсказание LightGBM по нескольким сидам (снижает дисперсию скора)
    preds = [
        lgb.LGBMClassifier(**PARAMS, random_state=s).fit(X_fit, y_fit).predict_proba(X_pred)[:, 1]
        for s in seeds
    ]
    return np.mean(preds, axis=0)


def time_cv(X: pd.DataFrame, y: np.ndarray, day: pd.Series, cols: list[str], name: str) -> None:
    scores, ys, ps = [], [], []
    for start, end in FOLDS:
        fit_m = (day < start).values
        val_m = ((day >= start) & (day < end)).values
        p = fit_predict(X.loc[fit_m, cols], y[fit_m], X.loc[val_m, cols])
        scores.append(precision_at_recall(y[val_m], p))
        ys.append(y[val_m]); ps.append(p)
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    print(f"{name:34s} P@R0.7 по фолдам {np.round(scores, 3)} среднее {np.mean(scores):.4f} | "
          f"ROC-AUC {roc_auc_score(y_all, p_all):.4f} PR-AUC {average_precision_score(y_all, p_all):.4f}")


def main() -> None:
    t0 = time.time()
    train, test, events = load_data()
    meta = pd.concat([train, test], ignore_index=True)
    feats = build_features(events, meta).set_index("cookie_id")
    X_train = feats.loc[train.cookie_id].reset_index(drop=True)
    X_test = feats.loc[test.cookie_id].reset_index(drop=True)
    y = train.target.values
    cols = list(X_train.columns)
    print(f"признаки: {len(cols)}, train {X_train.shape[0]}, test {X_test.shape[0]}  ({time.time() - t0:.0f} c)")

    if "--no-cv" not in sys.argv:
        day = train.window_start_ts.dt.normalize()
        time_cv(X_train, y, day, BASELINE_COLS, "baseline (n_events, item_nunique)")
        time_cv(X_train, y, day, [c for c in cols if not c.startswith(POPULARITY_PREFIXES)],
                "поведение без популярности объявл.")
        time_cv(X_train, y, day, cols, "все признаки (итоговая модель)")

    score = fit_predict(X_train[cols], y, X_test[cols])
    sub = pd.DataFrame({"cookie_id": test.cookie_id.values, "score": score})
    assert len(sub) == len(test) and sub.cookie_id.is_unique and sub.score.between(0, 1).all()
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv: {len(sub)} строк  ({time.time() - t0:.0f} c)")


if __name__ == "__main__":
    main()
