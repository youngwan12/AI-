"""대회 조건(랜덤 5-fold, seed 42)에서 '고객 ID + CatBoost'를 더 끌어올린다.

1단계: 변수 구성 × CatBoost 설정 비교
  F_ID    : 실험 E 변수 + 고객 ID
  F_ID2   : E + 고객 내 카드 위치 + 고객 ID + 동일 카드 ID
  F_CATALL: 원본 변수를 전부 범주형(문자열)으로 + 고객 ID (숫자 값 자체가 고객을 가리키는 열쇠라
            CatBoost의 범주형 target 통계를 쓰게 하는 방식)
2단계: 1등 구성을 seed 5개로 돌려 평균 (같은 fold, CatBoost random_seed만 변경)
3단계: 기존 모델(09, 10, 06)과 섞기. 가중치는 4/5로 고르고 1/5로 채점하는 정직한 방식으로 평가

결과: outputs/id_search.csv, outputs/submission_final.csv (기존보다 좋을 때만)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from best_search import REL
from features import CAT_COLS, TARGET, build_features
from group_experiments import ALL, DERIVED
from improve import add_relative

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RAW = ["gender", "car", "reality", "child_num", "income_total", "income_type", "edu_type", "family_type",
       "house_type", "DAYS_BIRTH", "DAYS_EMPLOYED", "work_phone", "phone", "email", "occyp_type", "family_size",
       "begin_month"]


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, X_te, _ = build_features(train, test)
    allX = add_relative(pd.concat([X, X_te], ignore_index=True))
    raw = pd.concat([train[RAW], test[RAW]], ignore_index=True).fillna("NaN").astype(str).add_prefix("raw_")
    allX = pd.concat([allX, raw], axis=1)
    n = len(train)
    return allX.iloc[:n].reset_index(drop=True), allX.iloc[n:].reset_index(drop=True), y


FEATURES = {
    "F_ID": (ALL + DERIVED + ["person_id"], CAT_COLS + ["person_id"]),
    "F_ID2": (ALL + DERIVED + REL + ["person_id", "full_id"], CAT_COLS + ["person_id", "full_id"]),
    "F_CATALL": (["raw_" + c for c in RAW] + DERIVED + REL + ["person_id"], ["raw_" + c for c in RAW] + ["person_id"]),
}


def run_cv(X, X_te, y, folds, fs, seed=42, with_test=False, **kw):
    cols, cats = FEATURES[fs]
    p = dict(iterations=6000, learning_rate=0.05, depth=6, loss_function="MultiClass", od_type="Iter", od_wait=200,
             verbose=0, thread_count=-1, random_seed=seed)
    p.update(kw)
    oof, te = np.zeros((len(X), 3)), np.zeros((len(X_te), 3))
    for a, b in folds:
        m = CatBoostClassifier(cat_features=cats, **p)
        m.fit(X.iloc[a][cols], y.iloc[a], eval_set=(X.iloc[b][cols], y.iloc[b]), use_best_model=True)
        oof[b] = m.predict_proba(X.iloc[b][cols])
        if with_test:
            te += m.predict_proba(X_te[cols]) / len(folds)
    return oof, te


def nested_blend(P, y, step=0.1):
    """P: OOF 리스트. 가중치를 4/5로 고르고 1/5로 채점한 OOF와, 전체로 고른 가중치를 반환."""
    k = len(P)
    grid = [w for w in np.array(np.meshgrid(*[np.arange(0, 1 + 1e-9, step)] * (k - 1))).reshape(k - 1, -1).T
            if w.sum() <= 1 + 1e-9]
    grid = [np.append(w, 1 - w.sum()) for w in grid]
    mix = lambda w, idx: sum(wi * p[idx] for wi, p in zip(w, P))
    nested = np.zeros_like(P[0])
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(y, y):
        w = min(grid, key=lambda w: log_loss(y[a], mix(w, a)))
        nested[b] = mix(w, b)
    best = min(grid, key=lambda w: log_loss(y, mix(w, np.arange(len(y)))))
    return nested, best


def main():
    t0 = time.time()
    X, X_te, y = load()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    rows = []

    def record(name, oof):
        r = dict(name=name, logloss=log_loss(y, oof), accuracy=accuracy_score(y, oof.argmax(1)),
                 minutes=round((time.time() - t0) / 60, 1))
        rows.append(r)
        print(f"{name:<40} logloss {r['logloss']:.4f}  acc {r['accuracy']:.4f}  [{r['minutes']}분]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "id_search.csv", index=False)
        return r["logloss"]

    # ---------- 1단계 ----------
    configs = [
        ("F_ID lr0.05 d6", "F_ID", {}),
        ("F_ID2 lr0.05 d6", "F_ID2", {}),
        ("F_CATALL lr0.05 d6", "F_CATALL", {}),
    ]
    scores = {}
    for name, fs, kw in configs:
        scores[(fs, ())] = record(name, run_cv(X, X_te, y, folds, fs, **kw)[0])
    best_fs = min(scores, key=scores.get)[0]
    for name, kw in [("lr0.03 d6", dict(learning_rate=0.03)), ("lr0.03 d4", dict(learning_rate=0.03, depth=4)),
                     ("lr0.03 d8 l2=10", dict(learning_rate=0.03, depth=8, l2_leaf_reg=10))]:
        scores[(best_fs, tuple(kw.items()))] = record(f"{best_fs} {name}", run_cv(X, X_te, y, folds, best_fs, **kw)[0])
    (fs, kw_items) = min(scores, key=scores.get)
    kw = dict(kw_items)
    print("1단계 최고:", fs, kw, flush=True)

    # ---------- 2단계: seed 5개 평균 ----------
    oofs, tes = [], []
    for seed in range(5):
        o, t = run_cv(X, X_te, y, folds, fs, seed=seed, with_test=True, **kw)
        oofs.append(o)
        tes.append(t)
        record(f"seed {seed}", o)
    seed_oof, seed_te = np.mean(oofs, axis=0), np.mean(tes, axis=0)
    record(f"{fs} {kw} seed 5개 평균", seed_oof)
    np.save(OUT / "oof" / "12_catboost_seedavg_oof.npy", seed_oof)
    np.save(OUT / "oof" / "12_catboost_seedavg_test.npy", seed_te)

    # ---------- 3단계: 기존 모델과 섞기 ----------
    names = ["09_lgbm_te_relative", "10_catboost_fullid_relative", "06_catboost_fe_id"]
    P = [np.load(OUT / "oof" / f"{n}_oof.npy") for n in names] + [seed_oof]
    T = [np.load(OUT / "oof" / f"{n}_test.npy") for n in names] + [seed_te]
    nested, w = nested_blend(P, y.values)
    final_ll = record("최종 블렌드 (가중치 정직 평가)", nested)
    print("블렌드 가중치 (09, 10, 06, 12):", np.round(w, 2), flush=True)
    full_oof = sum(wi * p for wi, p in zip(w, P))
    print(f"전체 가중치로 섞은 OOF: {log_loss(y, full_oof):.4f}", flush=True)
    if final_ll < 0.6717:
        sub = pd.read_csv(DATA / "sample_submission.csv")
        sub[["0", "1", "2"]] = sum(wi * t for wi, t in zip(w, T))
        sub.to_csv(OUT / "submission_final.csv", index=False)
        print("saved outputs/submission_final.csv", flush=True)


if __name__ == "__main__":
    main()
