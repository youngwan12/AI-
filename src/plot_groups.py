"""group_experiments.py 결과(outputs/group_results.csv)를 그림으로 그린다."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

LABELS = {
    "A_all": "A. all variables", "B_income_job": "B. income & job only", "C_period_age": "C. card period & age only",
    "D_asset_family": "D. asset & family only", "A-B": "A minus income & job", "A-C": "A minus card period & age",
    "A-D": "A minus asset & family", "E_all_derived": "E. all + derived", "E+_customer_id": "E+. E + customer key",
}
STYLE = {"LogReg": ("#8C8C8C", "s"), "RandomForest": ("#55A868", "o"), "CatBoost": ("#4C72B0", "D")}
PRIOR, BASELINE = 0.8824, 1.0738


def main():
    res = pd.read_csv(OUT / "group_results.csv")
    keys = list(dict.fromkeys(res["key"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, (color, marker) in STYLE.items():
        sub = res[res["model"] == model].set_index("key")
        ys = [i for i, k in enumerate(keys) if k in sub.index]
        ax.scatter(sub.loc[[keys[i] for i in ys], "logloss"], ys, color=color, marker=marker, s=60, label=model, zorder=3)
    ax.axvline(PRIOR, color="#C44E52", ls="--", lw=1)
    ax.text(PRIOR - 0.003, -0.8, f"class-ratio guess {PRIOR}", color="#C44E52", ha="right", fontsize=9)
    for y in [3.5, 6.5]:
        ax.axhline(y, color="#dddddd", lw=1)
    ax.set_yticks(range(len(keys)), [LABELS[k] for k in keys])
    ax.invert_yaxis()
    ax.set_ylim(len(keys) - 0.5, -1.2)
    ax.set_xlabel("5-fold OOF LogLoss (lower is better)")
    ax.set_title(f"Experiments A-E (DACON baseline = {BASELINE})")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "group_experiments.png", dpi=120)
    print("saved", OUT / "figures" / "group_experiments.png")


if __name__ == "__main__":
    main()
