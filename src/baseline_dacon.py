"""DACON 공식 Baseline 노트북을 그대로 옮긴 것.

원본과 다른 점은 실행 환경 때문에 바꾼 부분뿐이다.
- tensorflow의 to_categorical 대신 log_loss(labels=[0, 1, 2]) 사용 (계산 결과 동일)
- 최신 pandas에서 동작하도록 replace -> map, income_total의 object 변환 생략 (값 동일)
- RandomForest에 random_state를 지정해 다시 돌려도 같은 점수가 나오게 함 (원본은 매번 달라짐)

원본 그대로 둔 것 (원본의 특징이라 일부러 고치지 않음)
- occyp_type 열 삭제, 소득·나이·근속·카드기간 구간화, index 열을 입력 변수로 사용
- RandomForestClassifier 기본 설정, StratifiedKFold(5, shuffle, random_state=55)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
(OUT / "oof").mkdir(parents=True, exist_ok=True)


def preprocess():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    # 1. 결측치: occyp_type 열 삭제
    train = train.drop("occyp_type", axis=1)
    test = test.drop("occyp_type", axis=1)

    # 2. Binary variables
    for c, m in [("gender", {"F": 0, "M": 1}), ("car", {"N": 0, "Y": 1}), ("reality", {"N": 0, "Y": 1})]:
        train[c] = train[c].map(m)
        test[c] = test[c].map(m)

    # 3-(1) child_num: 2명 이상은 2
    train.loc[train["child_num"] >= 2, "child_num"] = 2
    test.loc[test["child_num"] >= 2, "child_num"] = 2

    # 3-(2) income_total: 만 단위로 나누고 train 기준 7구간
    train["income_total"] = train["income_total"] / 10000
    test["income_total"] = test["income_total"] / 10000
    _, bin_dividers = np.histogram(train["income_total"], bins=7)
    bin_names = ["소득" + str(i) for i in range(7)]
    train["income_total"] = pd.cut(x=train["income_total"], bins=bin_dividers, labels=bin_names, include_lowest=True)
    test["income_total"] = pd.cut(x=test["income_total"], bins=bin_dividers, labels=bin_names, include_lowest=True)

    # 3-(3) Type: LabelEncoder
    label_encoder = preprocessing.LabelEncoder()
    for c in ["income_type", "edu_type", "family_type", "house_type"]:
        train[c] = label_encoder.fit_transform(train[c])
        test[c] = label_encoder.transform(test[c])
    train["income_total"] = label_encoder.fit_transform(train["income_total"])
    test["income_total"] = label_encoder.fit_transform(test["income_total"])  # 원본도 test에 fit_transform

    # 4. 음수 연속형 변수: 부호 바꾸고 train 기준 구간화
    def make_bin(variable, n):
        train[variable] = -train[variable]
        test[variable] = -test[variable]
        _, bin_dividers = np.histogram(train[variable], bins=n)
        bin_names = [str(i) for i in range(n)]
        train[variable] = pd.cut(x=train[variable], bins=bin_dividers, labels=bin_names, include_lowest=True)
        test[variable] = pd.cut(x=test[variable], bins=bin_dividers, labels=bin_names, include_lowest=True)
        test[variable] = test[variable].fillna(str(0))
        train[variable] = label_encoder.fit_transform(train[variable])
        test[variable] = label_encoder.transform(test[variable])

    make_bin("DAYS_BIRTH", n=10)
    make_bin("DAYS_EMPLOYED", n=6)
    make_bin("begin_month", n=4)

    train_x = train.drop("credit", axis=1)  # index 열이 그대로 남아 있음 (원본과 동일)
    train_y = train["credit"].astype(int)
    return train_x, train_y, test


def run_kfold(train_x, train_y, test_x, fold_seed, rf_seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    oof = np.zeros((len(train_x), 3))
    sub = np.zeros((len(test_x), 3))
    for tr, va in folds.split(train_x, train_y):
        clf = RandomForestClassifier(random_state=rf_seed)
        clf.fit(train_x.iloc[tr], train_y.iloc[tr])
        oof[va] = clf.predict_proba(train_x.iloc[va])
        sub += clf.predict_proba(test_x) / folds.n_splits
    return oof, sub


def main():
    train_x, train_y, test_x = preprocess()
    print("입력 변수:", list(train_x.columns))

    # 원본 설정(fold seed 55)으로 RandomForest seed만 바꿔 5번
    scores = []
    for rf_seed in range(5):
        oof, _ = run_kfold(train_x, train_y, test_x, fold_seed=55, rf_seed=rf_seed)
        scores.append(log_loss(train_y, oof, labels=[0, 1, 2]))
    print(f"원본 분할(seed 55), RF seed 0~4: {np.round(scores, 4).tolist()} -> 평균 {np.mean(scores):.4f}")

    # 우리 실험과 같은 분할(seed 42)로 한 번: 우리 결과표와 직접 비교용
    oof, sub = run_kfold(train_x, train_y, test_x, fold_seed=42, rf_seed=42)
    ll = log_loss(train_y, oof, labels=[0, 1, 2])
    acc = accuracy_score(train_y, oof.argmax(1))
    f1 = f1_score(train_y, oof.argmax(1), average="macro")
    print(f"우리 분할(seed 42): logloss {ll:.4f}  acc {acc:.4f}  macroF1 {f1:.4f}")

    np.save(OUT / "oof" / "00_dacon_baseline_oof.npy", oof)
    submission = pd.read_csv(DATA / "sample_submission.csv")
    submission[["0", "1", "2"]] = sub
    submission.to_csv(OUT / "submission_dacon_baseline.csv", index=False)

    # train.py 결과표(outputs/results.csv)가 있으면 맨 위에 이 행을 넣는다
    res_path = OUT / "results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        res = res[res["name"] != "00_dacon_baseline"]
        row = pd.DataFrame([{"name": "00_dacon_baseline", "logloss": ll, "fold_std": np.nan,
                             "accuracy": acc, "macro_f1": f1, "seconds": np.nan}])
        pd.concat([row, res], ignore_index=True).to_csv(res_path, index=False)


if __name__ == "__main__":
    main()
