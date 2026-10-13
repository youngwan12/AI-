"""정확도가 어디까지 올라갈 수 있는지 데이터 자체로 확인한다.

1) 입력이 완전히 같은 행끼리 정답이 얼마나 갈리는지 (어떤 모델도 이 행들은 구분 못 함)
2) 최종 모델의 Top-1 / Top-2 정확도, 확신도 순으로 잘랐을 때의 정확도
3) 학습 데이터로 채점하면 왜 정확도가 부풀려지는지 (답 외우기)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import raw_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["credit"].astype(int).values

    # 1) 입력이 완전히 같은 행
    cols = [c for c in train.columns if c not in ("index", "credit")]
    key = train[cols].fillna("NaN").astype(str).agg("|".join, axis=1)
    ct = pd.crosstab(key, train["credit"])
    multi = ct[ct.sum(axis=1) > 1]
    n_rows = int(multi.values.sum())
    same_label = (multi.gt(0).sum(axis=1) == 1).mean()
    oracle = multi.max(axis=1).sum() / n_rows
    print(f"입력이 완전히 같은 행: {n_rows}행 ({len(multi)}그룹), 그룹 안 정답이 모두 같은 비율 {same_label:.3f}")
    print(f"  -> 이 행들은 정답을 미리 알고 찍어도 최대 정확도 {oracle:.3f}")

    # 2) 최종 모델
    name = "11_blend" if (OUT / "oof" / "11_blend_oof.npy").exists() else "08_ensemble"
    p = np.load(OUT / "oof" / f"{name}_oof.npy")
    top2 = np.argsort(-p, axis=1)[:, :2]
    print(f"\n[{name}] Top-1 정확도 {(p.argmax(1) == y).mean():.4f}, Top-2 정확도 {(top2 == y[:, None]).any(axis=1).mean():.4f}")
    order = np.argsort(-p.max(1))
    cum = np.cumsum((p.argmax(1) == y)[order]) / np.arange(1, len(y) + 1)
    for t in [0.95, 0.90, 0.85, 0.80]:
        k = np.where(cum >= t)[0].max() + 1
        print(f"  확신 높은 상위 {k}명({k / len(y):.1%})까지 정확도 {t:.0%} 이상")

    # 3) 답 외우기
    Xr, _ = raw_features(train, test)
    rf = RandomForestClassifier(random_state=0, n_jobs=-1).fit(Xr, y)
    print(f"\n학습에 쓴 train으로 다시 채점(답 외우기, 의미 없음): {(rf.predict(Xr) == y).mean():.4f}")


if __name__ == "__main__":
    main()
