"""중간보고서 탭 3 '분석 실험' A~E를 같은 교차검증 조건에서 돌린다.

A. 모든 변수            -> 기준 성능
B. 소득·고용 변수만      -> 경제활동 정보의 기여
C. begin_month·연령만    -> 기간 정보의 기여
D. 자산·가족 구성만      -> 생활·자산 정보의 기여
E. 전체 + 파생변수       -> 최종 성능 개선
E+. E + 고객 식별키      -> 같은 고객의 다른 카드 정보 (CatBoost만)

추가로 A에서 그룹 하나씩 뺐을 때(A-B, A-C, A-D) 얼마나 나빠지는지도 본다.
그룹 하나만 쓰는 실험은 '그 그룹 혼자 얼마나 아는가', 빼는 실험은 '다른 변수로 대체가 안 되는 몫'을 보여준다.

모델: Logistic Regression, Random Forest(500그루, min_samples_leaf=3), CatBoost
검증: StratifiedKFold(5, shuffle, seed 42) - 모든 실험·모델 동일

--customer-split 을 주면 같은 고객의 카드가 학습/검증에 나뉘지 않게 GroupKFold(고객 단위)로 돌린다.
랜덤 분할에서는 '같은 고객을 알아보는 능력'까지 점수에 섞이므로, 변수 자체의 기여는 이쪽 결과로 본다.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CAT_COLS, TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEED = 42

# 원본 변수 (단위 변환과 무직 상태 분리만 한 것)
INCOME_JOB = ["income_total", "income_type", "occyp_type", "years_employed", "unemployed"]
PERIOD_AGE = ["card_months", "age"]
ASSET_FAMILY = ["reality", "car", "family_size", "child_num", "family_type", "house_type"]
OTHERS = ["gender", "edu_type", "work_phone", "phone", "email"]
ALL = INCOME_JOB + PERIOD_AGE + ASSET_FAMILY + OTHERS
DERIVED = ["log_income", "income_per_person", "employed_ratio", "age_at_employment", "card_issue_age",
           "n_cards", "card_rank"]

EXPERIMENTS = [
    ("A_all", "A. 모든 변수", ALL),
    ("B_income_job", "B. 소득·고용", INCOME_JOB),
    ("C_period_age", "C. 카드기간·연령", PERIOD_AGE),
    ("D_asset_family", "D. 자산·가족", ASSET_FAMILY),
    ("A-B", "A에서 소득·고용 제외", [c for c in ALL if c not in INCOME_JOB]),
    ("A-C", "A에서 카드기간·연령 제외", [c for c in ALL if c not in PERIOD_AGE]),
    ("A-D", "A에서 자산·가족 제외", [c for c in ALL if c not in ASSET_FAMILY]),
    ("E_all_derived", "E. 전체 + 파생변수", ALL + DERIVED),
    ("E+_customer_id", "E+. E + 고객 식별키", ALL + DERIVED + ["person_id"]),
]


def fit_logreg(Xa, ya, Xb, cats, nums):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cats),
                             ("num", StandardScaler(), nums)])
    m = make_pipeline(pre, LogisticRegression(max_iter=3000))
    return m.fit(Xa, ya).predict_proba(Xb)


def fit_rf(Xa, ya, Xb, cats, nums):
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                             ("num", "passthrough", nums)])
    m = make_pipeline(pre, RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                  random_state=SEED, n_jobs=-1))
    return m.fit(Xa, ya).predict_proba(Xb)


def fit_catboost(Xa, ya, Xb, yb, cats):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=6, loss_function="MultiClass",
                           random_seed=SEED, od_type="Iter", od_wait=100, verbose=0, thread_count=-1,
                           cat_features=cats)
    m.fit(Xa, ya, eval_set=(Xb, yb), use_best_model=True)
    return m.predict_proba(Xb)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    X, _, _ = build_features(train, test)
    customer_split = "--customer-split" in sys.argv
    if customer_split:
        folds = list(GroupKFold(5).split(X, y, groups=X["person_id"]))
        out_csv = OUT / "group_results_customer_split.csv"
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
        out_csv = OUT / "group_results.csv"

    rows = []
    for key, label, cols in EXPERIMENTS:
        cats = [c for c in cols if c in CAT_COLS or c == "person_id"]
        nums = [c for c in cols if c not in cats]
        models = ["CatBoost"] if key.startswith("E+") else ["LogReg", "RandomForest", "CatBoost"]
        for model in models:
            t0 = time.time()
            oof = np.zeros((len(X), 3))
            for a, b in folds:
                Xa, Xb = X.iloc[a][cols], X.iloc[b][cols]
                if model == "LogReg":
                    oof[b] = fit_logreg(Xa, y.iloc[a], Xb, cats, nums)
                elif model == "RandomForest":
                    oof[b] = fit_rf(Xa, y.iloc[a], Xb, cats, nums)
                else:
                    oof[b] = fit_catboost(Xa, y.iloc[a], Xb, y.iloc[b], cats)
            r = dict(key=key, experiment=label, n_features=len(cols), model=model,
                     logloss=log_loss(y, oof), accuracy=accuracy_score(y, oof.argmax(1)),
                     seconds=round(time.time() - t0, 1))
            rows.append(r)
            print(f"{label:<22} {model:<12} logloss {r['logloss']:.4f}  acc {r['accuracy']:.4f}  ({r['seconds']}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    res = pd.DataFrame(rows)
    table = res.pivot_table(index=["key", "experiment"], columns="model", values="logloss", sort=False)
    print("\nLogLoss (낮을수록 좋음)\n", table.round(4).to_string())


if __name__ == "__main__":
    main()
