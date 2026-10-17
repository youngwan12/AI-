"""중간보고서 탭 3 'train.csv 변수 분석' 항목을 하나씩 검증한다.

1) begin_month: credit 0/1/2 그룹의 평균 카드 발급 경과 기간이 다른가 (제일 먼저 검증)
2) income_total: credit 0과 2의 평균 소득 차이, 소득 구간별 credit 비율
3) DAYS_EMPLOYED: 양수 값(무직)을 별도 상태로 봤을 때 credit 비율
4) reality, car: 자산 보유 여부별 credit 비율
5) family_size, child_num: 가구 구성별 credit 비율, 소득과 결합한 1인당 소득

검정: 연속형은 Kruskal-Wallis(세 그룹 분포 차이), 범주형은 카이제곱 독립성 검정.
p < 0.05면 '통계적으로 차이 있음'. 다만 표본이 2만 6천 개라 아주 작은 차이도 유의하게 나오므로
차이의 크기(평균 차이, 비율 차이)를 같이 본다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def by_credit_mean(df, col):
    g = df.groupby("credit")[col]
    stat, p = kruskal(*[v.values for _, v in g])
    return g.mean(), g.median(), p


def crosstab_test(df, col):
    ct = pd.crosstab(df[col], df["credit"])
    _, p, _, _ = chi2_contingency(ct)
    ratio = pd.crosstab(df[col], df["credit"], normalize="index").round(3)
    ratio["n"] = ct.sum(axis=1)
    return ratio, p


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    df["credit"] = df["credit"].astype(int)
    df["card_months"] = -df["begin_month"]
    df["unemployed"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    df["years_employed"] = np.where(df["unemployed"] == 1, 0, -df["DAYS_EMPLOYED"] / 365.25)
    df["income_per_person"] = df["income_total"] / df["family_size"]

    print("=" * 60, "\n1) begin_month: credit별 카드 발급 경과 기간(개월)")
    mean, med, p = by_credit_mean(df, "card_months")
    print(pd.DataFrame({"평균": mean.round(2), "중앙값": med}).to_string(), f"\nKruskal-Wallis p = {p:.2e}")
    print("발급 경과 연차별 credit 비율:")
    print(pd.crosstab(df["card_months"] // 12, df["credit"], normalize="index").round(3).to_string())

    print("=" * 60, "\n2) income_total: credit별 연간 소득")
    mean, med, p = by_credit_mean(df, "income_total")
    print(pd.DataFrame({"평균": mean.round(0), "중앙값": med}).to_string(), f"\nKruskal-Wallis p = {p:.3f}")
    diff = (mean[2] - mean[0]) / mean[0] * 100
    print(f"credit 0 대비 credit 2 평균 소득 차이: {diff:+.2f}%")
    df["income_bin"] = pd.qcut(df["income_total"], 5, labels=["하위20%", "20-40%", "40-60%", "60-80%", "상위20%"])
    ratio, p = crosstab_test(df, "income_bin")
    print("소득 5분위별 credit 비율:\n", ratio.to_string(), f"\n카이제곱 p = {p:.3f}")

    print("=" * 60, "\n3) DAYS_EMPLOYED: 양수(365243) = 무직 상태로 분리")
    ratio, p = crosstab_test(df, "unemployed")
    print(ratio.rename(index={0: "근무 중", 1: "무직(양수 값)"}).to_string(), f"\n카이제곱 p = {p:.3f}")
    emp = df[df["unemployed"] == 0]
    mean, med, p = by_credit_mean(emp, "years_employed")
    print("근무 중인 사람의 credit별 근속연수:\n", pd.DataFrame({"평균": mean.round(2), "중앙값": med.round(2)}).to_string(),
          f"\nKruskal-Wallis p = {p:.3f}")

    print("=" * 60, "\n4) reality, car: 자산 보유 여부")
    for c in ["reality", "car"]:
        ratio, p = crosstab_test(df, c)
        print(f"[{c}]\n", ratio.to_string(), f"\n카이제곱 p = {p:.3f}")

    print("=" * 60, "\n5) family_size, child_num: 가구 구성")
    for c in ["family_size", "child_num"]:
        tmp = df.assign(**{c: df[c].clip(upper=4)})
        ratio, p = crosstab_test(tmp, c)
        print(f"[{c}] (4 이상은 4로 묶음)\n", ratio.to_string(), f"\n카이제곱 p = {p:.3f}")
    mean, med, p = by_credit_mean(df, "income_per_person")
    print("credit별 1인당 소득:\n", pd.DataFrame({"평균": mean.round(0), "중앙값": med}).to_string(),
          f"\nKruskal-Wallis p = {p:.3f}")

    # ---- 그림 ----
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    pd.crosstab(df["card_months"] // 12, df["credit"], normalize="index").plot.bar(stacked=True, ax=axes[0, 0], color=colors)
    axes[0, 0].set(title="1) credit ratio by card age", xlabel="years since card issued", ylabel="ratio")
    df.groupby("credit")["income_total"].mean().plot.bar(ax=axes[0, 1], color=colors)
    axes[0, 1].set(title="2) mean income by credit", xlabel="credit", ylabel="income_total")
    pd.crosstab(df["income_bin"], df["credit"], normalize="index").plot.bar(stacked=True, ax=axes[0, 2], color=colors)
    axes[0, 2].set(title="2) credit ratio by income quintile", xlabel="")
    axes[0, 2].set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], rotation=0)
    pd.crosstab(df["unemployed"].map({0: "employed", 1: "unemployed"}), df["credit"], normalize="index").plot.bar(
        stacked=True, ax=axes[1, 0], color=colors)
    axes[1, 0].set(title="3) credit ratio by employment status", xlabel="")
    asset = pd.concat([
        pd.crosstab("reality=" + df["reality"], df["credit"], normalize="index"),
        pd.crosstab("car=" + df["car"], df["credit"], normalize="index"),
    ])
    asset.plot.bar(stacked=True, ax=axes[1, 1], color=colors)
    axes[1, 1].set(title="4) credit ratio by asset ownership", xlabel="")
    pd.crosstab(df["family_size"].clip(upper=4), df["credit"], normalize="index").plot.bar(stacked=True, ax=axes[1, 2], color=colors)
    axes[1, 2].set(title="5) credit ratio by family size (4+ grouped)", xlabel="family_size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()
        ax.tick_params(axis="x", rotation=0)
    fig.legend(handles, labels, title="credit", loc="upper right", ncol=3)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(FIG / "variable_analysis.png", dpi=120)
    print("\nsaved", FIG / "variable_analysis.png")


if __name__ == "__main__":
    main()
