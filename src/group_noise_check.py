"""고객 단위 분할 A~E 결과의 차이가 우연(분할 운) 수준인지 확인한다.

고객 단위 분할을 seed만 바꿔 3번 반복하고(StratifiedGroupKFold), 실험별 LogLoss 평균·표준편차와
A 대비 차이를 구한다. 같은 분할 안에서 A와 비교하므로(짝지은 비교) 분할 운의 영향이 줄어든다.
차이가 표준편차의 2배보다 작으면 '차이 없음'으로 본다.
모델은 고객 단위 분할에서 가장 좋았던 CatBoost만 쓴다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from features import CAT_COLS, TARGET, build_features
from group_experiments import EXPERIMENTS, fit_catboost

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEEDS = [0, 1, 2]
KEYS = ["A_all", "B_income_job", "C_period_age", "D_asset_family", "A-B", "A-C", "A-D", "E_all_derived"]


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, _, _ = build_features(train, test)
    exps = {k: (label, cols) for k, label, cols in EXPERIMENTS if k in KEYS}

    rows = []
    for seed in SEEDS:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=X["person_id"]))
        for key in KEYS:
            label, cols = exps[key]
            cats = [c for c in cols if c in CAT_COLS]
            oof = np.zeros((len(X), 3))
            for a, b in folds:
                oof[b] = fit_catboost(X.iloc[a][cols], y.iloc[a], X.iloc[b][cols], y.iloc[b], cats)
            rows.append(dict(seed=seed, key=key, experiment=label, logloss=log_loss(y, oof)))
            print(f"seed {seed} {label:<22} {rows[-1]['logloss']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "group_noise_check.csv", index=False)
    wide = res.pivot(index="seed", columns="key", values="logloss")[KEYS]
    diff = wide.sub(wide["A_all"], axis=0)
    summary = pd.DataFrame({
        "experiment": [exps[k][0] for k in KEYS],
        "logloss_mean": wide.mean().values, "logloss_std": wide.std().values,
        "diff_vs_A_mean": diff.mean().values, "diff_vs_A_std": diff.std().values,
    }, index=KEYS)
    summary["판단"] = np.where(summary.index == "A_all", "기준",
                             np.where(summary["diff_vs_A_mean"].abs() < 2 * summary["diff_vs_A_std"].clip(lower=1e-4),
                                      "A와 차이 없음", np.where(summary["diff_vs_A_mean"] > 0, "A보다 나쁨", "A보다 좋음")))
    summary.to_csv(OUT / "group_noise_summary.csv")
    print("\n", summary.round(4).to_string())


if __name__ == "__main__":
    main()
