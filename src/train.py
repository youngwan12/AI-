"""Baseline부터 단계별로 실험하고 5-fold OOF LogLoss를 기록한 뒤 제출 파일을 만든다.

실행: python src/train.py
결과: outputs/results.csv, outputs/oof/*.npy, outputs/submission_*.csv
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import TARGET, build_features, raw_features, to_category

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
(OUT / "oof").mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", message=".*eval_set.*")

SEED = 42
N_FOLDS = 5


def evaluate(y, proba):
    return {
        "logloss": log_loss(y, proba),
        "accuracy": accuracy_score(y, proba.argmax(1)),
        "macro_f1": f1_score(y, proba.argmax(1), average="macro"),
    }


def run_cv(name, fit_predict, X, y, X_test):
    """fit_predict(X_tr, y_tr, X_va, y_va, X_test) -> (va_proba, test_proba)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    fold_scores = []
    t0 = time.time()
    for k, (tr, va) in enumerate(skf.split(X, y)):
        va_p, te_p = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X_test)
        oof[va] = va_p
        test_pred += te_p / N_FOLDS
        fold_scores.append(log_loss(y.iloc[va], va_p))
    res = evaluate(y, oof)
    res.update(name=name, fold_std=float(np.std(fold_scores)), seconds=round(time.time() - t0, 1))
    np.save(OUT / "oof" / f"{name}_oof.npy", oof)
    np.save(OUT / "oof" / f"{name}_test.npy", test_pred)
    print(f"{name:<28} logloss {res['logloss']:.4f} (±{res['fold_std']:.4f})  acc {res['accuracy']:.4f}  "
          f"macroF1 {res['macro_f1']:.4f}  {res['seconds']}s", flush=True)
    return res, oof, test_pred


# ---------------- 모델 정의 ----------------
def rf_baseline(X_tr, y_tr, X_va, y_va, X_te):
    m = RandomForestClassifier(random_state=SEED, n_jobs=-1)  # 기본 설정 그대로
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_va), m.predict_proba(X_te)


def make_logreg(cat_cols, num_cols):
    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        m = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va), m.predict_proba(X_te)
    return fit_predict


def make_rf_tuned():
    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                   random_state=SEED, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va), m.predict_proba(X_te)
    return fit_predict


def make_lgbm(params=None):
    p = dict(n_estimators=5000, learning_rate=0.02, num_leaves=63, min_child_samples=10,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
             random_state=SEED, verbose=-1)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        m = LGBMClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss",
              callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        return m.predict_proba(X_va), m.predict_proba(X_te)
    return fit_predict


def make_catboost(cat_cols, params=None):
    p = dict(iterations=5000, learning_rate=0.05, depth=6, loss_function="MultiClass",
             random_seed=SEED, od_type="Iter", od_wait=200, verbose=0, thread_count=-1)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        m = CatBoostClassifier(**p, cat_features=cat_cols)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        return m.predict_proba(X_va), m.predict_proba(X_te)
    return fit_predict


def best_blend(oofs, y):
    """OOF LogLoss가 최소가 되는 가중치(합 1)를 찾는다."""
    k = len(oofs)

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return log_loss(y, sum(wi * o for wi, o in zip(w, oofs)))

    r = minimize(loss, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 2000})
    w = np.abs(r.x) / np.abs(r.x).sum()
    return w


def save_submission(test_pred, name, sample):
    sub = sample.copy()
    sub[["0", "1", "2"]] = test_pred
    sub.to_csv(OUT / f"submission_{name}.csv", index=False)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train[TARGET].astype(int)

    results = []
    prior = np.tile(y.value_counts(normalize=True).sort_index().values, (len(y), 1))
    r = evaluate(y, prior)
    r.update(name="00_prior", fold_std=0.0, seconds=0)
    results.append(r)
    print(f"{'00_prior':<28} logloss {r['logloss']:.4f}")

    # --- 0) Baseline: 원본 + 정수 인코딩 + 기본 RandomForest ---
    Xr, Xr_te = raw_features(train, test)
    r, _, te = run_cv("01_baseline_rf", rf_baseline, Xr, y, Xr_te)
    results.append(r)
    save_submission(te, "baseline_rf", sample)

    # 0-b) 원본 그대로, RandomForest 설정만 조정 (튜닝 효과만 분리)
    r, *_ = run_cv("01b_rf_tuned_raw", make_rf_tuned(), Xr, y, Xr_te)
    results.append(r)

    # --- 전처리 + 파생변수 ---
    X, X_te, cat_cols = build_features(train, test)
    feat_no_id = [c for c in X.columns if c != "person_id"]
    num_cols = [c for c in feat_no_id if c not in cat_cols]

    # 1) Logistic Regression (전처리 + 파생변수, person_id 없음)
    r, *_ = run_cv("02_logreg_fe", make_logreg(cat_cols, num_cols), X[feat_no_id], y, X_te[feat_no_id])
    results.append(r)

    # 2) RandomForest (전처리 + 파생변수, 튜닝)
    Xc = to_category(X[feat_no_id], cat_cols, X_te)
    Xc_te = to_category(X_te[feat_no_id], cat_cols, X)
    Xo, Xo_te = Xc.copy(), Xc_te.copy()
    for c in cat_cols:
        Xo[c], Xo_te[c] = Xo[c].cat.codes, Xo_te[c].cat.codes
    r, *_ = run_cv("03_rf_fe", make_rf_tuned(), Xo, y, Xo_te)
    results.append(r)

    # 2-b) 전처리 + 파생변수, RandomForest 기본 설정 (전처리 효과만 분리)
    r, *_ = run_cv("03b_rf_default_fe", rf_baseline, Xo, y, Xo_te)
    results.append(r)

    # 3) LightGBM (전처리 + 파생변수)
    r, *_ = run_cv("04_lgbm_fe", make_lgbm(), Xc, y, Xc_te)
    results.append(r)

    # 4) CatBoost (전처리 + 파생변수)
    r, *_ = run_cv("05_catboost_fe", make_catboost(cat_cols), X[feat_no_id], y, X_te[feat_no_id])
    results.append(r)

    # 5) CatBoost + 고객 식별키(person_id)
    cat_id = cat_cols + ["person_id"]
    r, oof_cb, te_cb = run_cv("06_catboost_fe_id", make_catboost(cat_id), X, y, X_te)
    results.append(r)

    # 6) LightGBM + 고객 식별키 (빈도가 낮은 범주가 많아 category 대신 정수 코드)
    Xi, Xi_te = Xc.copy(), Xc_te.copy()
    codes = pd.concat([X["person_id"], X_te["person_id"]]).astype("category").cat.codes
    Xi["person_code"], Xi_te["person_code"] = codes.iloc[:len(X)].values, codes.iloc[len(X):].values
    r, oof_lgb, te_lgb = run_cv("07_lgbm_fe_id", make_lgbm(), Xi, y, Xi_te)
    results.append(r)

    # 7) 앙상블 (가중 평균)
    rf_oof = np.load(OUT / "oof" / "03_rf_fe_oof.npy")
    rf_te = np.load(OUT / "oof" / "03_rf_fe_test.npy")
    oofs, tes = [oof_cb, oof_lgb, rf_oof], [te_cb, te_lgb, rf_te]
    w = best_blend(oofs, y)
    blend_oof = sum(wi * o for wi, o in zip(w, oofs))
    blend_te = sum(wi * t for wi, t in zip(w, tes))
    r = evaluate(y, blend_oof)
    r.update(name="08_ensemble", fold_std=float("nan"), seconds=0)
    results.append(r)
    print(f"{'08_ensemble':<28} logloss {r['logloss']:.4f}  weights cb/lgb/rf = {np.round(w, 3).tolist()}")
    np.save(OUT / "oof" / "08_ensemble_oof.npy", blend_oof)
    save_submission(blend_te, "ensemble", sample)
    save_submission(te_cb, "catboost_id", sample)

    df = pd.DataFrame(results)[["name", "logloss", "fold_std", "accuracy", "macro_f1", "seconds"]]
    df.to_csv(OUT / "results.csv", index=False)
    with open(OUT / "ensemble_weights.json", "w") as f:
        json.dump({"catboost_fe_id": w[0], "lgbm_fe_id": w[1], "rf_fe": w[2]}, f, indent=2)
    print("\n", df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
