"""train.py가 저장한 결과로 발표용 그림을 만든다.

- outputs/figures/results_logloss.png : 실험별 OOF LogLoss
- outputs/figures/confusion_matrix.png : 최종 앙상블 OOF 혼동행렬
- outputs/figures/calibration.png      : Baseline vs 최종 앙상블 신뢰도 곡선
- outputs/figures/feature_importance.png : LightGBM(+고객키) 변수 중요도
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from features import TARGET, build_features, to_category

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    res = pd.read_csv(OUT / "results.csv")

    # 1) 실험별 LogLoss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#bbbbbb" if n in ("00_dacon_baseline", "00_prior") else "#4C72B0" for n in res["name"]]
    ax.barh(res["name"], res["logloss"], color=colors)
    for i, v in enumerate(res["logloss"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center")
    ax.axvline(res.loc[res["name"] == "00_dacon_baseline", "logloss"].item(), color="#C44E52", ls="--", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold OOF LogLoss (lower is better)")
    ax.set_xlim(0, res["logloss"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIG / "results_logloss.png", dpi=120)
    plt.close()

    # 2) 혼동행렬 + 분류 리포트
    best = np.load(OUT / "oof" / "08_ensemble_oof.npy")
    print(classification_report(y, best.argmax(1), digits=3))
    ConfusionMatrixDisplay.from_predictions(y, best.argmax(1), normalize="true", values_format=".2f", cmap="Blues")
    plt.title("Ensemble OOF confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=120)
    plt.close()

    # 3) 신뢰도 곡선 (클래스 0 = 신용 좋은 고객의 확률)
    base = np.load(OUT / "oof" / "00_dacon_baseline_oof.npy")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axes):
        for name, p in [("DACON baseline", base), ("ensemble", best)]:
            t, m = calibration_curve((y == k).astype(int), p[:, k], n_bins=10, strategy="quantile")
            ax.plot(m, t, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"credit = {k}")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "calibration.png", dpi=120)
    plt.close()

    # 4) 변수 중요도 (전체 train으로 LightGBM 1회 학습, gain 기준)
    X, X_te, cat_cols = build_features(train, test)
    Xc = to_category(X.drop(columns=["person_id"]), cat_cols, X_te)
    codes = pd.concat([X["person_id"], X_te["person_id"]]).astype("category").cat.codes
    Xc["person_code"] = codes.iloc[: len(X)].values
    m = LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=63, min_child_samples=10,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=42,
                       verbose=-1, importance_type="gain")
    m.fit(Xc, y)
    imp = pd.Series(m.feature_importances_, index=Xc.columns).sort_values()
    imp = imp / imp.sum()
    imp.plot.barh(figsize=(8, 7), color="#4C72B0")
    plt.xlabel("share of total gain")
    plt.title("LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance.png", dpi=120)
    plt.close()
    print(imp.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
