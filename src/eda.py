"""EDA: 데이터 구조, 클래스 비율, 결측치, 이상치, 중복 고객을 확인하고 그림을 저장한다."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PERSON_COLS = [
    "gender", "car", "reality", "child_num", "income_total", "income_type",
    "edu_type", "family_type", "house_type", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_MOBIL", "work_phone", "phone", "email", "occyp_type", "family_size",
]


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    print("== shape ==")
    print("train", train.shape, "test", test.shape)

    print("\n== dtypes / 결측치 ==")
    print(pd.DataFrame({"dtype": train.dtypes, "train_na": train.isna().sum(), "test_na": test.isna().sum()}))

    print("\n== credit 분포 ==")
    dist = train["credit"].value_counts().sort_index()
    print(pd.DataFrame({"count": dist, "ratio": (dist / len(train)).round(4)}))
    prior = (dist / len(train)).values
    print("prior LogLoss (클래스 비율로만 예측):", round(-(prior * np.log(prior)).sum(), 4))

    print("\n== 상수/이상값 ==")
    print("FLAG_MOBIL unique:", train["FLAG_MOBIL"].unique())
    print("DAYS_EMPLOYED > 0:", (train["DAYS_EMPLOYED"] > 0).sum(), "값:", train.loc[train["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"].unique())
    print("DAYS_EMPLOYED>0 인 행의 income_type:\n", train.loc[train["DAYS_EMPLOYED"] > 0, "income_type"].value_counts())
    print("DAYS_EMPLOYED>0 & occyp_type 결측:", (train.loc[train["DAYS_EMPLOYED"] > 0, "occyp_type"].isna()).mean().round(4))
    print("child_num 분포:\n", train["child_num"].value_counts().sort_index())
    print("family_size 분포:\n", train["family_size"].value_counts().sort_index())
    print("family_size < child_num+1 (가족 수 < 자녀+본인):", (train["family_size"] < train["child_num"] + 1).sum())
    print("income_total describe:\n", train["income_total"].describe())
    print("age(years) describe:\n", (-train["DAYS_BIRTH"] / 365.25).describe())
    print("begin_month describe:\n", train["begin_month"].describe())

    print("\n== 중복 고객 ==")
    key = train[PERSON_COLS].fillna("NaN").astype(str).agg("|".join, axis=1)
    n_person = key.nunique()
    print("train 행 수", len(train), "고유 고객 수", n_person)
    grp = train.assign(key=key).groupby("key")["credit"]
    multi = grp.size()
    print("카드 2장 이상 고객:", (multi > 1).sum(), "| 그 고객들의 행:", multi[multi > 1].sum())
    nun = grp.nunique()
    print("같은 고객인데 credit이 서로 다른 고객 수:", (nun > 1).sum())
    test_key = test[PERSON_COLS].fillna("NaN").astype(str).agg("|".join, axis=1)
    print("test 행 중 train에 같은 고객이 있는 비율:", test_key.isin(set(key)).mean().round(4))
    full_dup = train.drop(columns=["index"]).duplicated().sum()
    print("index 제외 완전 중복 행:", full_dup)

    print("\n== 범주형 변수별 credit 비율 ==")
    for c in ["income_type", "edu_type", "family_type", "house_type", "occyp_type"]:
        tab = pd.crosstab(train[c].fillna("NaN"), train["credit"], normalize="index").round(3)
        tab["n"] = train[c].fillna("NaN").value_counts()
        print(f"\n[{c}]\n", tab.sort_values("n", ascending=False))

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    dist.plot.bar(ax=axes[0, 0], color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0, 0].set_title("credit distribution")
    for i, v in enumerate(dist.values):
        axes[0, 0].text(i, v, f"{v}\n({v / len(train):.1%})", ha="center", va="bottom")

    na = train.isna().mean().sort_values(ascending=False)
    na = na[na > 0]
    na.plot.bar(ax=axes[0, 1], color="#888888")
    axes[0, 1].set_title("missing ratio")

    (-train["DAYS_BIRTH"] / 365.25).plot.hist(bins=40, ax=axes[0, 2], color="#4C72B0")
    axes[0, 2].set_title("age (years)")

    np.log1p(train["income_total"]).plot.hist(bins=40, ax=axes[1, 0], color="#4C72B0")
    axes[1, 0].set_title("log(income_total)")

    emp = train.loc[train["DAYS_EMPLOYED"] <= 0, "DAYS_EMPLOYED"] / -365.25
    emp.plot.hist(bins=40, ax=axes[1, 1], color="#4C72B0")
    axes[1, 1].set_title("years employed (excluding 365243)")

    pd.crosstab(-train["begin_month"] // 12, train["credit"], normalize="index").plot.bar(
        stacked=True, ax=axes[1, 2], color=["#4C72B0", "#55A868", "#C44E52"]
    )
    axes[1, 2].set_title("credit ratio by card age (years)")
    axes[1, 2].set_xlabel("years since card issued")

    plt.tight_layout()
    plt.savefig(FIG / "eda_overview.png", dpi=120)
    print("\nsaved", FIG / "eda_overview.png")


if __name__ == "__main__":
    main()
