"""최종 모델(CatBoost 단독: 원본 변수 전부 범주형 + 고객 ID, lr 0.03, seed 5개 평균)의 결과 정리.

id_search.py가 저장한 outputs/oof/12_catboost_seedavg_*.npy를 사용한다.

만드는 것
- outputs/submission_catboost_final.csv : 최종 제출 파일
- outputs/final_metrics.json             : 발표용 숫자
- outputs/figures/final_1_card_age.png   : 신호 1, 카드 발급 경과 기간별 credit 비율
- outputs/figures/final_2_customer_id.png : 신호 2, 고객 ID를 넣었을 때/뺐을 때 (대회 조건 vs 처음 보는 고객)
- outputs/figures/final_3_calibration.png  : 확률이 실제 비율과 맞는지 (DACON Baseline과 비교)
- outputs/figures/final_4_confusion.png    : 등급별로 얼마나 맞히는지
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from features import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
GREY, BLUE, RED = "#9E9E9E", "#2F6FD6", "#C44E52"
BASELINE, PRIOR, TOP1 = 1.0738, 0.8824, 0.6581


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["credit"].astype(int).values
    oof = np.load(OUT / "oof" / "12_catboost_seedavg_oof.npy")
    te = np.load(OUT / "oof" / "12_catboost_seedavg_test.npy")
    base = np.load(OUT / "oof" / "00_dacon_baseline_oof.npy")

    # 제출 파일
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["0", "1", "2"]] = te
    sub.to_csv(OUT / "submission_catboost_final.csv", index=False)

    # 아는 고객 / 처음 보는 고객 (train.py와 같은 5-fold 분할 기준)
    X, X_te, _ = build_features(train, test)
    pid = X["person_id"].values
    seen = np.zeros(len(y), bool)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        seen[b] = pd.Series(pid[b]).isin(set(pid[a])).values
    test_seen = float(X_te["person_id"].isin(set(pid)).mean())
    ll_seen = log_loss(y[seen], oof[seen], labels=[0, 1, 2])
    ll_new = log_loss(y[~seen], oof[~seen], labels=[0, 1, 2])

    top2 = np.argsort(-oof, axis=1)[:, :2]
    metrics = {
        "logloss": log_loss(y, oof),
        "accuracy": accuracy_score(y, oof.argmax(1)),
        "top2_accuracy": float((top2 == y[:, None]).any(axis=1).mean()),
        "reduction_vs_baseline": 1 - log_loss(y, oof) / BASELINE,
        "logloss_known_customers": ll_seen, "logloss_new_customers": ll_new,
        "share_known_in_validation": float(seen.mean()), "share_known_in_test": test_seen,
        "logloss_reweighted_to_test": test_seen * ll_seen + (1 - test_seen) * ll_new,
        "accuracy_known": accuracy_score(y[seen], oof[seen].argmax(1)),
        "accuracy_new": accuracy_score(y[~seen], oof[~seen].argmax(1)),
    }
    (OUT / "final_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    for k, v in metrics.items():
        print(f"{k:<28} {v:.4f}")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # 1) 카드 발급 경과 기간별 credit 2 비율
    years = (-train["begin_month"] // 12).astype(int)
    share2 = pd.Series(y == 2).groupby(years.values).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(share2.index, share2.values, marker="o", color=BLUE, lw=2)
    for xv, v in share2.items():
        ax.annotate(f"{v:.1f}%", (xv, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.axhline((y == 2).mean() * 100, color=GREY, ls="--", lw=1)
    ax.text(share2.index.max(), (y == 2).mean() * 100 - 2.5, f"overall {(y == 2).mean() * 100:.1f}%",
            color=GREY, ha="right")
    ax.set_xticks(share2.index, ["<1y", "1y", "2y", "3y", "4y", "5y"])
    ax.set_xlabel("years since card issued")
    ax.set_ylabel("share of credit = 2 (%)")
    ax.set_ylim(40, 80)
    ax.set_title("Signal 1: older cards are more often credit 2")
    plt.tight_layout()
    plt.savefig(FIG / "final_1_card_age.png", dpi=130)
    plt.close()

    # 2) 고객 ID를 넣었을 때 / 뺐을 때 (같은 조건끼리 비교)
    #    랜덤 분할: results.csv의 05(ID 없음) vs 06(ID 있음)
    #    고객 단위 분할: group_results_customer_split.csv의 E vs E+ (CatBoost)
    res = pd.read_csv(OUT / "results.csv").set_index("name")["logloss"]
    grp = pd.read_csv(OUT / "group_results_customer_split.csv")
    grp = grp[grp["model"] == "CatBoost"].set_index("key")["logloss"]
    pairs = {"competition setting\n(88% of test customers are known)": (res["05_catboost_fe"], res["06_catboost_fe_id"]),
             "new customers only\n(customer-level split)": (grp["E_all_derived"], grp["E+_customer_id"])}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(pairs))
    for i, (label, (without, with_id)) in enumerate(pairs.items()):
        for dx, v, c, name in [(-0.18, without, GREY, "without customer ID"), (0.18, with_id, BLUE, "with customer ID")]:
            ax.bar(i + dx, v, width=0.34, color=c, label=name if i == 0 else None)
            ax.text(i + dx, v + 0.01, f"{v:.4f}", ha="center")
    ax.axhline(PRIOR, color=RED, ls="--", lw=1)
    ax.text(1.45, PRIOR + 0.01, f"class-ratio guess {PRIOR}", color=RED, ha="right")
    ax.set_xticks(xs, list(pairs))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("LogLoss (lower is better)")
    ax.set_title("Signal 2: the customer ID only helps customers seen before")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "final_2_customer_id.png", dpi=130)
    plt.close()

    # 3) 신뢰도 곡선
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for k, ax in enumerate(axes):
        for name, p, c in [("DACON baseline", base, RED), ("final CatBoost", oof, BLUE)]:
            t, m = calibration_curve((y == k).astype(int), p[:, k], n_bins=10, strategy="quantile")
            ax.plot(m, t, marker="o", color=c, label=name)
        ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
        ax.set(title=f"credit = {k}", xlabel="predicted probability", ylabel="observed share", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    fig.suptitle("Predicted probabilities vs observed shares (closer to the diagonal is better)")
    plt.tight_layout()
    plt.savefig(FIG / "final_3_calibration.png", dpi=130)
    plt.close()

    # 4) 혼동행렬
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay.from_predictions(y, oof.argmax(1), normalize="true", values_format=".2f", cmap="Blues",
                                            ax=ax, colorbar=False)
    ax.set_title(f"Final CatBoost: accuracy {metrics['accuracy']:.1%}, top-2 {metrics['top2_accuracy']:.1%}")
    plt.tight_layout()
    plt.savefig(FIG / "final_4_confusion.png", dpi=130)
    plt.close()
    print("saved figures to", FIG)


if __name__ == "__main__":
    main()
