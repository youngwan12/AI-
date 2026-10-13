"""정확도를 최대한 끌어올리기 위한 추가 실험.

추가 신호
- 같은 고객(person_id)의 다른 카드 credit 분포 (fold 밖 데이터로만 계산하는 target encoding)
- 신상 + 발급월까지 완전히 같은 행(full_id)의 credit 분포 (같은 방식)
- 고객 안에서 이 카드의 상대 위치 (가장 최근 카드인지, 발급월 차이 등)

검증은 train.py와 같은 StratifiedKFold(5, seed 42). target encoding은
학습 fold 안에서 다시 5-fold로 나눠 계산해, 자기 정답이 자기 변수에 섞이지 않게 한다.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from features import PERSON_COLS, TARGET, build_features, to_category

warnings.filterwarnings("ignore", message=".*eval_set.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEED = 42


def add_relative(X: pd.DataFrame) -> pd.DataFrame:
    """고객 안에서 카드의 상대적 위치."""
    X = X.copy()
    g = X.groupby("person_id")["card_months"]
    X["person_max_months"] = g.transform("max")
    X["person_min_months"] = g.transform("min")
    X["months_from_newest"] = X["card_months"] - X["person_min_months"]
    X["months_to_oldest"] = X["person_max_months"] - X["card_months"]
    X["is_newest"] = (X["card_months"] == X["person_min_months"]).astype(int)
    X["full_id"] = X["person_id"] + "|" + X["card_months"].astype(str)
    X["n_same_card"] = X.groupby("full_id")["full_id"].transform("size")
    return X


def te_table(keys, y, prior, alpha):
    """key별 클래스 비율 (prior로 smoothing). 반환: DataFrame[key -> te0, te1, te2, n]"""
    cnt = pd.crosstab(keys, y).reindex(columns=[0, 1, 2], fill_value=0)
    n = cnt.sum(axis=1)
    te = (cnt + alpha * prior).div(n + alpha, axis=0)
    te["n"] = n
    return te


def apply_te(keys, table, prior, prefix):
    t = table.reindex(keys.values)
    out = pd.DataFrame(index=keys.index)
    for c in [0, 1, 2]:
        out[f"{prefix}_te{c}"] = t[c].fillna(prior[c]).values
    out[f"{prefix}_n"] = t["n"].fillna(0).values
    return out


def target_encode(X_tr, y_tr, X_ap, cols, alpha=2.0, inner=5):
    """학습 행은 내부 5-fold OOF로, 적용 행은 학습 fold 전체로 encoding."""
    prior = y_tr.value_counts(normalize=True).reindex([0, 1, 2]).values
    tr_parts, ap_parts = [], []
    for col in cols:
        oof = pd.DataFrame(index=X_tr.index)
        skf = StratifiedKFold(inner, shuffle=True, random_state=SEED)
        pieces = []
        for a, b in skf.split(X_tr, y_tr):
            tab = te_table(X_tr[col].iloc[a], y_tr.iloc[a], prior, alpha)
            pieces.append(apply_te(X_tr[col].iloc[b], tab, prior, col))
        tr_parts.append(pd.concat(pieces).loc[X_tr.index])
        tab = te_table(X_tr[col], y_tr, prior, alpha)
        ap_parts.append(apply_te(X_ap[col], tab, prior, col))
    return pd.concat(tr_parts, axis=1), pd.concat(ap_parts, axis=1)


def report(name, y, p):
    r = dict(name=name, logloss=log_loss(y, p), accuracy=accuracy_score(y, p.argmax(1)),
             macro_f1=f1_score(y, p.argmax(1), average="macro"))
    print(f"{name:<34} logloss {r['logloss']:.4f}  acc {r['accuracy']:.4f}  macroF1 {r['macro_f1']:.4f}", flush=True)
    return r


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, X_te, cat_cols = build_features(train, test)
    allX = add_relative(pd.concat([X, X_te], ignore_index=True))
    X, X_te = allX.iloc[: len(X)].reset_index(drop=True), allX.iloc[len(X):].reset_index(drop=True)

    te_cols = ["person_id", "full_id"]
    drop = ["person_id", "full_id"]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = []

    # A) LightGBM + 고객/동일카드 target encoding + 상대 위치 변수
    oof_lgb = np.zeros((len(X), 3))
    test_lgb = np.zeros((len(X_te), 3))
    for k, (a, b) in enumerate(skf.split(X, y)):
        Xa, Xb, ya, yb = X.iloc[a], X.iloc[b], y.iloc[a], y.iloc[b]
        ta, tb = target_encode(Xa, ya, Xb, te_cols)
        _, tt = target_encode(Xa, ya, X_te, te_cols)
        Fa = to_category(pd.concat([Xa.drop(columns=drop), ta], axis=1), cat_cols, X_te)
        Fb = to_category(pd.concat([Xb.drop(columns=drop), tb], axis=1), cat_cols, X_te)
        Ft = to_category(pd.concat([X_te.drop(columns=drop), tt], axis=1), cat_cols, X)
        m = LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                           random_state=SEED, verbose=-1)
        m.fit(Fa, ya, eval_set=[(Fb, yb)], callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        oof_lgb[b] = m.predict_proba(Fb)
        test_lgb += m.predict_proba(Ft) / 5
    results.append(report("09_lgbm_te_relative", y, oof_lgb))
    np.save(OUT / "oof" / "09_lgbm_te_relative_oof.npy", oof_lgb)
    np.save(OUT / "oof" / "09_lgbm_te_relative_test.npy", test_lgb)

    # B) CatBoost + person_id, full_id 범주형 + 상대 위치 변수 (CatBoost가 내부에서 순서 기반 target 통계 사용)
    cb_cats = cat_cols + ["person_id", "full_id"]
    oof_cb = np.zeros((len(X), 3))
    test_cb = np.zeros((len(X_te), 3))
    for k, (a, b) in enumerate(skf.split(X, y)):
        m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, loss_function="MultiClass",
                               random_seed=SEED, od_type="Iter", od_wait=200, verbose=0, cat_features=cb_cats)
        m.fit(X.iloc[a], y.iloc[a], eval_set=(X.iloc[b], y.iloc[b]), use_best_model=True)
        oof_cb[b] = m.predict_proba(X.iloc[b])
        test_cb += m.predict_proba(X_te) / 5
    results.append(report("10_catboost_fullid_relative", y, oof_cb))
    np.save(OUT / "oof" / "10_catboost_fullid_relative_oof.npy", oof_cb)
    np.save(OUT / "oof" / "10_catboost_fullid_relative_test.npy", test_cb)

    # C) 기존 최고(06)와 섞기
    prev = np.load(OUT / "oof" / "06_catboost_fe_id_oof.npy")
    prev_te = np.load(OUT / "oof" / "06_catboost_fe_id_test.npy")
    best = None
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            p = w1 * oof_lgb + w2 * oof_cb + (1 - w1 - w2) * prev
            ll = log_loss(y, p)
            if best is None or ll < best[0]:
                best = (ll, w1, w2)
    _, w1, w2 = best
    blend = w1 * oof_lgb + w2 * oof_cb + (1 - w1 - w2) * prev
    blend_te = w1 * test_lgb + w2 * test_cb + (1 - w1 - w2) * prev_te
    r = report(f"11_blend(lgb{w1:.1f}/cb{w2:.1f}/prev{1 - w1 - w2:.1f})", y, blend)
    results.append(r)
    np.save(OUT / "oof" / "11_blend_oof.npy", blend)
    np.save(OUT / "oof" / "11_blend_test.npy", blend_te)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["0", "1", "2"]] = blend_te
    sub.to_csv(OUT / "submission_blend.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "improve_results.csv", index=False)


if __name__ == "__main__":
    main()
