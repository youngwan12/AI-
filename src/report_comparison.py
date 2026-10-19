"""중간보고서 비교군 실험 (모두 같은 Stratified 5-fold, seed 42 분할).

비교군
  0. Baseline           : DACON 공식 Baseline (baseline_dacon.py 결과)
  1. begin_month        : 카드 발급 경과 개월(begin_month)만 사용     -> RandomForest / CatBoost
  2. begin_month + 고객 ID : 1 + 같은 고객을 묶는 고객 ID            -> RandomForest / CatBoost
     - CatBoost는 고객 ID 문자열을 범주형으로 그대로 사용
     - RandomForest는 문자열을 못 받으므로, 고객 ID별 credit 비율을 fold 밖 데이터로만 계산해 넣음
  3. 최종               : 원본 변수 전부 범주형 + 파생변수 + 고객 ID, CatBoost(lr 0.03, depth 6), seed 3 고정

공통 설정
  RandomForest: 500그루, min_samples_leaf=3, max_features=0.5
  CatBoost    : lr 0.05, depth 6, 검증 fold 기준 early stopping 200 (최종만 lr 0.03)

결과: outputs/report_comparison.csv, outputs/oof/final_seed3_*.npy, outputs/submission_final_seed3.csv
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from id_search import load, run_cv
from improve import target_encode

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def rf_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1)


def cb_model(cats):
    return CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, loss_function="MultiClass",
                              random_seed=42, od_type="Iter", od_wait=200, verbose=0, cat_features=cats)


def main():
    t0 = time.time()
    X, X_te, y = load()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    rows = []

    def record(group, model, features, oof):
        r = dict(group=group, model=model, features=features, logloss=log_loss(y, oof),
                 accuracy=accuracy_score(y, oof.argmax(1)), macro_f1=f1_score(y, oof.argmax(1), average="macro"),
                 minutes=round((time.time() - t0) / 60, 1))
        rows.append(r)
        print(f"{group:<22} {model:<12} logloss {r['logloss']:.4f}  acc {r['accuracy']:.4f}  [{r['minutes']}분]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "report_comparison.csv", index=False)

    # 0. Baseline (baseline_dacon.py가 같은 분할로 저장한 OOF)
    base = np.load(OUT / "oof" / "00_dacon_baseline_oof.npy")
    record("0. Baseline", "RandomForest", "DACON 공식 전처리(구간화, 직업 삭제)", base)

    # 1. begin_month만
    bm = ["card_months"]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        oof[b] = rf_model().fit(X.iloc[a][bm], y.iloc[a]).predict_proba(X.iloc[b][bm])
    record("1. begin_month", "RandomForest", "begin_month", oof)
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        m = cb_model([]).fit(X.iloc[a][bm], y.iloc[a], eval_set=(X.iloc[b][bm], y.iloc[b]), use_best_model=True)
        oof[b] = m.predict_proba(X.iloc[b][bm])
    record("1. begin_month", "CatBoost", "begin_month", oof)

    # 2. begin_month + 고객 ID
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        ta, tb = target_encode(X.iloc[a], y.iloc[a], X.iloc[b], ["person_id"])
        Fa = pd.concat([X.iloc[a][bm].reset_index(drop=True), ta.reset_index(drop=True)], axis=1)
        Fb = pd.concat([X.iloc[b][bm].reset_index(drop=True), tb.reset_index(drop=True)], axis=1)
        oof[b] = rf_model().fit(Fa, y.iloc[a]).predict_proba(Fb)
    record("2. begin_month + 고객 ID", "RandomForest", "begin_month + 고객 ID별 credit 비율(fold 밖)", oof)
    cols = bm + ["person_id"]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        m = cb_model(["person_id"]).fit(X.iloc[a][cols], y.iloc[a], eval_set=(X.iloc[b][cols], y.iloc[b]),
                                        use_best_model=True)
        oof[b] = m.predict_proba(X.iloc[b][cols])
    record("2. begin_month + 고객 ID", "CatBoost", "begin_month + 고객 ID(범주형)", oof)

    # 3. 최종 (seed 3 고정)
    oof, te = run_cv(X, X_te, y, folds, "F_CATALL", seed=3, with_test=True, learning_rate=0.03)
    record("3. 최종 (seed 3)", "CatBoost", "원본 변수 전부 범주형 + 파생변수 + begin_month + 고객 ID", oof)
    np.save(OUT / "oof" / "final_seed3_oof.npy", oof)
    np.save(OUT / "oof" / "final_seed3_test.npy", te)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["0", "1", "2"]] = te
    sub.to_csv(OUT / "submission_final_seed3.csv", index=False)

    top2 = np.argsort(-oof, axis=1)[:, :2]
    yv = y.values
    summary = {"final_logloss": log_loss(y, oof), "final_accuracy": accuracy_score(y, oof.argmax(1)),
               "final_top2": float((top2 == yv[:, None]).any(axis=1).mean()),
               "baseline_logloss": log_loss(y, base), "baseline_accuracy": accuracy_score(y, base.argmax(1)),
               "class_recall_final": [float((oof.argmax(1)[yv == k] == k).mean()) for k in range(3)],
               "class_recall_baseline": [float((base.argmax(1)[yv == k] == k).mean()) for k in range(3)]}
    (OUT / "report_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
