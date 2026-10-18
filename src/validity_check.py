"""지금까지 보고한 점수가 부풀려지지 않았는지 점검한다.

1) 조기 종료(early stopping) 누수: 지금까지 CatBoost/LightGBM은 검증 fold를 보면서 멈출 시점을 골랐다.
   그 fold로 점수까지 매기면 약간 낙관적이 된다. 학습 fold 안에서 따로 떼어낸 데이터로 멈출 시점을
   고르고, 검증 fold는 채점에만 쓰면 점수가 얼마나 달라지는지 본다.
2) 중복 기록 편향: 랜덤 분할에서 검증 행과 '완전히 같은 행'이 학습 쪽에 있는 비율이 test와 비슷한가.
3) 신규 고객 성능: 고객 단위로 나누는 GroupKFold에서는 점수가 어떻게 되는가.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

from features import CAT_COLS, TARGET, build_features
from group_experiments import ALL, DERIVED

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
COLS = ALL + DERIVED + ["person_id"]
CATS = [c for c in COLS if c in CAT_COLS or c == "person_id"]


def catboost(Xa, ya, Xs, ys):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=6, loss_function="MultiClass",
                           random_seed=SEED, od_type="Iter", od_wait=100, verbose=0, cat_features=CATS)
    m.fit(Xa, ya, eval_set=(Xs, ys), use_best_model=True)
    return m


def run(X, y, folds, honest, grouped=False):
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        Xa, ya = X.iloc[a], y.iloc[a]
        if honest:  # 학습 fold의 15%를 떼어 멈출 시점 선택용으로만 사용
            if grouped:  # 고객 단위 평가에서는 떼는 15%도 고객 단위로 (같은 고객이 섞이면 너무 오래 학습함)
                t, s = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(Xa, ya, groups=Xa["person_id"]))
                Xt, Xs, yt, ys = Xa.iloc[t], Xa.iloc[s], ya.iloc[t], ya.iloc[s]
            else:
                Xt, Xs, yt, ys = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
            m = catboost(Xt, yt, Xs, ys)
        else:       # 지금까지 방식: 검증 fold로 멈출 시점 선택
            m = catboost(Xa, ya, X.iloc[b], y.iloc[b])
        oof[b] = m.predict_proba(X.iloc[b])
    return log_loss(y, oof), accuracy_score(y, oof.argmax(1))


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, X_te, _ = build_features(train, test)
    X = X[COLS]
    strat = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    group = list(GroupKFold(5).split(X, y, groups=X["person_id"]))

    # 2) 중복 기록 비율
    full = X["person_id"] + "|" + X["card_months"].astype(str)
    full_te = X_te["person_id"] + "|" + X_te["card_months"].astype(str)
    same_row = np.mean([full.iloc[b].isin(set(full.iloc[a])).mean() for a, b in strat])
    same_cust = np.mean([X["person_id"].iloc[b].isin(set(X["person_id"].iloc[a])).mean() for a, b in strat])
    print("[중복 기록] 검증 행 중 학습 쪽에 같은 고객이 있는 비율: "
          f"랜덤 5-fold {same_cust:.3f} / test {X_te['person_id'].isin(set(X['person_id'])).mean():.3f}")
    print("[중복 기록] 검증 행 중 학습 쪽에 완전히 같은 행이 있는 비율: "
          f"랜덤 5-fold {same_row:.3f} / test {full_te.isin(set(full)).mean():.3f}")

    # 1) 조기 종료 누수
    for honest in [False, True]:
        ll, acc = run(X, y, strat, honest)
        tag = "멈출 시점을 학습 fold 안에서 선택(정직)" if honest else "멈출 시점을 검증 fold로 선택(기존)"
        print(f"[랜덤 5-fold] {tag}: logloss {ll:.4f}, acc {acc:.4f}", flush=True)

    # 3) 신규 고객
    ll, acc = run(X, y, group, honest=True, grouped=True)
    print(f"[고객 단위 GroupKFold, 신규 고객 가정] logloss {ll:.4f}, acc {acc:.4f}")


if __name__ == "__main__":
    main()
