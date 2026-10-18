"""처음 보는 고객 기준(고객 단위 분할)으로 가장 좋은 방법을 찾는다.

1단계 (seed 0): 변수 묶음 × 모델 후보를 모두 비교
  변수 묶음
    E      : 원본 18개 + 파생변수 (group_experiments.py의 E)
    E_REL  : E + 같은 고객의 다른 카드 발급 시점 정보(정답 없이 입력값만 사용)
    TIME   : 기간 관련 변수만 (카드 기간, 나이, 카드 발급 나이, 보유 카드 수, 카드 순서 + 고객 내 위치)
  모델
    CatBoost, LightGBM(Optuna 튜닝), 순서형 CatBoost(0<1<2 순서 이용), 신경망(MLP), 카드 기간별 비율표
2단계 (seed 1, 2): 1단계 상위 후보와 앙상블을 다시 돌려 차이가 진짜인지 확인
`--honest-check`: 1등 단일 모델(catboost TIME)을 조기 종료 누수 없이(학습 fold 안에서 멈출 시점 선택 후
전체 재학습) 다시 평가한다.

결과: outputs/best_search.csv, outputs/best_search_summary.csv, outputs/best_search_log.txt
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CAT_COLS, TARGET, build_features, to_category
from group_experiments import ALL, DERIVED
from improve import add_relative

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

REL = ["person_max_months", "person_min_months", "months_from_newest", "months_to_oldest", "is_newest", "n_same_card"]
FEATURE_SETS = {
    "E": ALL + DERIVED,
    "E_REL": ALL + DERIVED + REL,
    "TIME": ["card_months", "age", "card_issue_age", "n_cards", "card_rank"] + REL,
}


def folds_for(X, y, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=X["person_id"]))


# ---------------- 모델 ----------------
def cb_model(cats, **kw):
    p = dict(iterations=3000, learning_rate=0.05, depth=6, loss_function="MultiClass", random_seed=42,
             od_type="Iter", od_wait=150, verbose=0, thread_count=-1)
    p.update(kw)
    return CatBoostClassifier(cat_features=cats, **p)


def run_catboost(X, y, folds, cols, **kw):
    cats = [c for c in cols if c in CAT_COLS]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        m = cb_model(cats, **kw).fit(X.iloc[a][cols], y.iloc[a], eval_set=(X.iloc[b][cols], y.iloc[b]), use_best_model=True)
        oof[b] = m.predict_proba(X.iloc[b][cols])
    return oof


def run_ordinal(X, y, folds, cols):
    """P(credit>0), P(credit>1) 두 이진 모델로 순서를 이용해 세 클래스 확률을 만든다."""
    cats = [c for c in cols if c in CAT_COLS]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        ps = []
        for k in [0, 1]:
            ya, yb = (y.iloc[a] > k).astype(int), (y.iloc[b] > k).astype(int)
            m = cb_model(cats, loss_function="Logloss").fit(X.iloc[a][cols], ya, eval_set=(X.iloc[b][cols], yb),
                                                            use_best_model=True)
            ps.append(m.predict_proba(X.iloc[b][cols])[:, 1])
        gt0, gt1 = ps[0], np.minimum(ps[1], ps[0])
        p = np.column_stack([1 - gt0, gt0 - gt1, gt1]).clip(1e-6)
        oof[b] = p / p.sum(1, keepdims=True)
    return oof


def lgbm_fit_predict(Xa, ya, Xb, yb, params):
    m = LGBMClassifier(n_estimators=5000, random_state=42, verbose=-1, **params)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
    return m.predict_proba(Xb)


def run_lgbm(X, y, folds, cols, params):
    cats = [c for c in cols if c in CAT_COLS]
    Xc = to_category(X[cols], cats) if cats else X[cols]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        oof[b] = lgbm_fit_predict(Xc.iloc[a], y.iloc[a], Xc.iloc[b], y.iloc[b], params)
    return oof


def tune_lgbm(X, y, folds, cols, n_trials):
    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        )
        return log_loss(y, run_lgbm(X, y, folds, cols, params))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params, subsample_freq=1)
    return best, study.best_value


def run_mlp(X, y, folds, cols):
    cats = [c for c in cols if c in CAT_COLS]
    nums = [c for c in cols if c not in cats]
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
                                 ("n", StandardScaler(), nums)])
        m = make_pipeline(pre, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                             max_iter=300, random_state=42))
        oof[b] = m.fit(X.iloc[a][cols], y.iloc[a]).predict_proba(X.iloc[b][cols])
    return oof


def run_table(X, y, folds, keys, alpha=5.0):
    """카드 기간(등) 값별 credit 비율표. 표에 없는 값은 전체 비율로 채운다."""
    oof = np.zeros((len(X), 3))
    for a, b in folds:
        prior = y.iloc[a].value_counts(normalize=True).reindex([0, 1, 2]).values
        k_a = X.iloc[a][keys].astype(str).agg("|".join, axis=1)
        k_b = X.iloc[b][keys].astype(str).agg("|".join, axis=1)
        cnt = pd.crosstab(k_a, y.iloc[a]).reindex(columns=[0, 1, 2], fill_value=0)
        tab = (cnt + alpha * prior).div(cnt.sum(axis=1) + alpha, axis=0)
        oof[b] = tab.reindex(k_b.values).fillna(pd.Series(prior, index=[0, 1, 2])).values
    return oof


def honest_check(X, y):
    cols = FEATURE_SETS["TIME"]
    for seed in [0, 1, 2]:
        oof = np.zeros((len(X), 3))
        for a, b in folds_for(X, y, seed):
            Xa, ya = X.iloc[a][cols], y.iloc[a]
            # 멈출 시점을 고르는 데이터도 고객 단위로 떼어야 '처음 보는 고객' 조건과 맞는다
            t, s = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(Xa, ya, groups=X["person_id"].iloc[a]))
            Xt, Xs, yt, ys = Xa.iloc[t], Xa.iloc[s], ya.iloc[t], ya.iloc[s]
            n_iter = cb_model([]).fit(Xt, yt, eval_set=(Xs, ys), use_best_model=True).get_best_iteration() + 1
            m = cb_model([], iterations=int(n_iter / 0.85), od_type=None, od_wait=None).fit(Xa, ya)
            oof[b] = m.predict_proba(X.iloc[b][cols])
        print(f"[누수 없는 평가] seed {seed} catboost TIME logloss {log_loss(y, oof):.4f}  acc {accuracy_score(y, oof.argmax(1)):.4f}",
              flush=True)


def main():
    t_start = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, X_te, _ = build_features(train, test)
    allX = add_relative(pd.concat([X, X_te], ignore_index=True))
    X = allX.iloc[: len(X)].reset_index(drop=True)
    if "--honest-check" in sys.argv:
        honest_check(X, y)
        return

    rows, oofs = [], {}

    def record(name, oof, seed):
        r = dict(stage=seed, name=name, logloss=log_loss(y, oof), accuracy=accuracy_score(y, oof.argmax(1)),
                 minutes=round((time.time() - t_start) / 60, 1))
        rows.append(r)
        oofs[(name, seed)] = oof
        print(f"seed {seed} {name:<34} logloss {r['logloss']:.4f}  acc {r['accuracy']:.4f}  [{r['minutes']}분]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "best_search.csv", index=False)

    # ---------- 1단계: seed 0 ----------
    f0 = folds_for(X, y, 0)
    record("prior(찍기)", np.tile(y.value_counts(normalize=True).sort_index().values, (len(y), 1)), 0)
    record("table: card_months", run_table(X, y, f0, ["card_months"]), 0)
    record("table: card_months+is_newest", run_table(X, y, f0, ["card_months", "is_newest"]), 0)
    for fs, cols in FEATURE_SETS.items():
        record(f"catboost {fs}", run_catboost(X, y, f0, cols), 0)
    record("catboost E_REL depth4 lr0.03", run_catboost(X, y, f0, FEATURE_SETS["E_REL"], depth=4, learning_rate=0.03), 0)
    record("catboost E_REL depth8 l2=10", run_catboost(X, y, f0, FEATURE_SETS["E_REL"], depth=8, l2_leaf_reg=10), 0)
    record("ordinal catboost E_REL", run_ordinal(X, y, f0, FEATURE_SETS["E_REL"]), 0)
    record("mlp E_REL", run_mlp(X, y, f0, FEATURE_SETS["E_REL"]), 0)
    lgb_params, _ = tune_lgbm(X, y, f0, FEATURE_SETS["E_REL"], n_trials=40)
    print("LightGBM 튜닝 결과:", lgb_params, flush=True)
    record("lgbm(tuned) E_REL", run_lgbm(X, y, f0, FEATURE_SETS["E_REL"], lgb_params), 0)

    # 1단계 상위 3개 모델(찍기·표 제외) 평균 앙상블
    stage1 = pd.DataFrame([r for r in rows if r["stage"] == 0 and not r["name"].startswith(("prior", "table"))])
    top = stage1.sort_values("logloss")["name"].head(3).tolist()
    print("상위 3개:", top, flush=True)
    record("ensemble(top3 평균)", np.mean([oofs[(n, 0)] for n in top], axis=0), 0)

    # ---------- 2단계: seed 1, 2에서 확인 ----------
    runners = {
        "catboost E": lambda f: run_catboost(X, y, f, FEATURE_SETS["E"]),
        "catboost E_REL": lambda f: run_catboost(X, y, f, FEATURE_SETS["E_REL"]),
        "catboost E_REL depth4 lr0.03": lambda f: run_catboost(X, y, f, FEATURE_SETS["E_REL"], depth=4, learning_rate=0.03),
        "catboost E_REL depth8 l2=10": lambda f: run_catboost(X, y, f, FEATURE_SETS["E_REL"], depth=8, l2_leaf_reg=10),
        "catboost TIME": lambda f: run_catboost(X, y, f, FEATURE_SETS["TIME"]),
        "ordinal catboost E_REL": lambda f: run_ordinal(X, y, f, FEATURE_SETS["E_REL"]),
        "mlp E_REL": lambda f: run_mlp(X, y, f, FEATURE_SETS["E_REL"]),
        "lgbm(tuned) E_REL": lambda f: run_lgbm(X, y, f, FEATURE_SETS["E_REL"], lgb_params),
    }
    confirm = list(dict.fromkeys(["catboost E"] + top))
    for seed in [1, 2]:
        fs = folds_for(X, y, seed)
        for name in confirm:
            record(name, runners[name](fs), seed)
        record("ensemble(top3 평균)", np.mean([oofs[(n, seed)] for n in top], axis=0), seed)

    res = pd.DataFrame(rows)
    summary = res.groupby("name").agg(logloss_mean=("logloss", "mean"), logloss_std=("logloss", "std"),
                                      acc_mean=("accuracy", "mean"), runs=("logloss", "size"))
    print("\n", summary.sort_values("logloss_mean").round(4).to_string())
    summary.to_csv(OUT / "best_search_summary.csv")


if __name__ == "__main__":
    main()
