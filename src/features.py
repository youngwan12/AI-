"""전처리와 파생변수 생성."""
import numpy as np
import pandas as pd

TARGET = "credit"

# begin_month(카드 발급 시점)를 뺀 고객 신상 정보. 이 값이 모두 같으면 같은 고객으로 본다.
PERSON_COLS = [
    "gender", "car", "reality", "child_num", "income_total", "income_type",
    "edu_type", "family_type", "house_type", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "work_phone", "phone", "email", "occyp_type", "family_size",
]

CAT_COLS = ["income_type", "edu_type", "family_type", "house_type", "occyp_type"]


def raw_features(train: pd.DataFrame, test: pd.DataFrame):
    """Baseline용: index만 빼고 원본 그대로, 문자열 컬럼은 정수 코드로만 바꾼다."""
    n_train = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True).drop(columns=["index"])
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna("NaN").astype("category").cat.codes
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """train/test를 같이 받아 같은 규칙으로 전처리한다. (train, test, 범주형 컬럼 목록) 반환."""
    n_train = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)

    # 1) 상수 열 제거
    df = df.drop(columns=["FLAG_MOBIL"])

    # 2) 2값 변수 -> 0/1
    df["gender"] = (df["gender"] == "M").astype(int)
    df["car"] = (df["car"] == "Y").astype(int)
    df["reality"] = (df["reality"] == "Y").astype(int)

    # 3) DAYS_EMPLOYED 양수(365243) = 무직/연금 수령자 -> 플래그 + 0으로 치환
    df["unemployed"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    df.loc[df["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = 0

    # 4) 결측치: 무직이면 'Unemployed', 아니면 'Unknown'
    df["occyp_type"] = np.where(
        df["occyp_type"].isna(),
        np.where(df["unemployed"] == 1, "Unemployed", "Unknown"),
        df["occyp_type"],
    )

    # 고객 식별 키는 이상치 처리 전 원래 값으로 만든다
    df["person_id"] = df[PERSON_COLS].fillna("NaN").astype(str).agg("|".join, axis=1)

    # 5) 이상치: 자녀 수, 가족 수 상한
    df["child_num"] = df["child_num"].clip(upper=4)
    df["family_size"] = df["family_size"].clip(upper=6)

    # 6) 단위 변환
    df["age"] = -df["DAYS_BIRTH"] / 365.25
    df["years_employed"] = -df["DAYS_EMPLOYED"] / 365.25
    df["card_months"] = -df["begin_month"]

    # 7) 파생변수
    df["log_income"] = np.log1p(df["income_total"])
    df["income_per_person"] = df["income_total"] / df["family_size"]
    df["employed_ratio"] = df["years_employed"] / df["age"]
    df["age_at_employment"] = df["age"] - df["years_employed"]
    df["card_issue_age"] = df["age"] - df["card_months"] / 12
    df["n_cards"] = df.groupby("person_id")["person_id"].transform("size")
    df["card_rank"] = df.groupby("person_id")["card_months"].rank(method="dense")

    df = df.drop(columns=["index", "DAYS_BIRTH", "DAYS_EMPLOYED", "begin_month"])

    tr = df.iloc[:n_train].reset_index(drop=True)
    te = df.iloc[n_train:].reset_index(drop=True)
    return tr, te, CAT_COLS


def to_category(df: pd.DataFrame, cat_cols, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """LightGBM용: 문자열 컬럼을 pandas category로. ref가 있으면 같은 범주 목록을 쓴다."""
    out = df.copy()
    for c in cat_cols:
        cats = sorted(pd.concat([df[c], ref[c]]).unique()) if ref is not None else sorted(df[c].unique())
        out[c] = pd.Categorical(out[c], categories=cats)
    return out
