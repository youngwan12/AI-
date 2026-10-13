# 신용카드 사용자 연체 예측 (DACON 과제 11번)

고객 정보로 신용등급 `credit`(0/1/2)의 **확률**을 예측합니다. 평가지표는 다중분류 LogLoss(낮을수록 좋음)입니다.
대회 페이지: https://dacon.io/competitions/official/235713/data

## 폴더 구조

```
data/                  train.csv, test.csv, sample_submission.csv (DACON에서 받아서 넣기, git에는 안 올림)
src/eda.py             EDA: 클래스 비율, 결측치, 이상치, 중복 고객 확인 + 그림
src/features.py        전처리 + 파생변수
src/train.py           Baseline부터 단계별 실험 (5-fold) + 제출 파일 생성
src/report.py          발표용 그림 (실험 결과, 혼동행렬, 신뢰도 곡선, 변수 중요도)
outputs/results.csv    실험별 점수
outputs/submission_*.csv  DACON 제출 파일
outputs/figures/       그림
```

## 실행 방법

```bash
pip install -r requirements.txt
python src/eda.py      # 1분 이내
python src/train.py    # 약 20분 (CatBoost가 대부분, CPU 4코어 기준), 로그는 outputs/train_log.txt
python src/report.py   # 1분 이내
```

## 사전 실험 결과 (Stratified 5-fold OOF, seed 42)

| 실험 | 바꾼 것 | LogLoss | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 00_prior | 모든 행에 클래스 비율(0.122/0.237/0.641)만 예측 | 0.8824 | 0.6413 | 0.2605 |
| 01_baseline_rf | **Baseline**: 원본 + 정수 인코딩 + RandomForest 기본값 | 0.9242 | 0.7027 | 0.5589 |
| 01b_rf_tuned_raw | 원본 그대로, RF 설정만 조정 (500그루, min_samples_leaf=3) | 0.7201 | 0.7135 | 0.5021 |
| 02_logreg_fe | 전처리 + 파생변수 + 로지스틱 회귀 | 0.8575 | 0.6454 | 0.2830 |
| 03_rf_fe | 전처리 + 파생변수 + RF 설정 조정 | 0.7262 | 0.7126 | 0.5020 |
| 03b_rf_default_fe | 전처리 + 파생변수 + RF 기본값 | 0.8785 | 0.7024 | 0.5397 |
| 04_lgbm_fe | 전처리 + 파생변수 + LightGBM | 0.7153 | 0.7166 | 0.5156 |
| 05_catboost_fe | 전처리 + 파생변수 + CatBoost | 0.7383 | 0.7100 | 0.4859 |
| 06_catboost_fe_id | 05 + 고객 식별키(범주형) | **0.6750** | 0.7363 | 0.5790 |
| 07_lgbm_fe_id | 04 + 고객 식별키(정수 코드) | 0.7121 | 0.7182 | 0.5190 |
| 08_ensemble | 06×0.93 + 07×0.07 (OOF로 가중치 탐색) | **0.6748** | 0.7358 | 0.5744 |

정확한 값은 `outputs/results.csv`에 있습니다(라이브러리 버전이 다르면 소수 넷째 자리가 조금 달라질 수 있음).

### 결과에서 읽을 점

- **Baseline이 "클래스 비율만 찍기"(0.8824)보다도 나쁩니다(0.9242).** 기본 RandomForest는 잎이 1개 샘플까지 자라서 확률을 0이나 1에 가깝게 냅니다. 확신하고 틀리는 경우가 많아 LogLoss가 크게 나빠집니다.
- **가장 큰 개선 1: 확률을 부드럽게 내는 설정.** `min_samples_leaf=3`만으로 0.9242 → 0.7201(22% 감소).
- **가장 큰 개선 2: 고객 식별키.** begin_month를 뺀 신상 정보가 같으면 같은 고객으로 보면 26,457행이 고객 8,759명입니다. test의 88.4%가 train에 있는 고객이라, CatBoost가 고객별 연체 성향을 학습하면 0.7383 → 0.6750으로 좋아집니다.
- 개별 범주형 변수(직업, 학력 등)는 credit 비율 차이가 거의 없어 단독으로는 신호가 약합니다.
- 변수 중요도(LightGBM gain) 1위는 카드 보유 기간 `card_months`(24%)이고, 그다음이 고객 식별키, 나이, 근속 관련 변수입니다.
- 최종 앙상블은 credit 0(신용 가장 좋음)의 재현율이 24%로 낮습니다. 확률 자체는 잘 맞습니다(`outputs/figures/calibration.png`).
- LightGBM에 고객키를 정수 코드로 넣는 것은 효과가 작습니다(0.7153 → 0.7121). 정수 코드 순서에 의미가 없기 때문이고, OOF target encoding이 다음 개선 후보입니다.

## 제출 파일

- `outputs/submission_ensemble.csv` : 최종 후보 (OOF 0.6748)
- `outputs/submission_catboost_id.csv` : CatBoost 단독 (OOF 0.6750)
- `outputs/submission_baseline_rf.csv` : Baseline (비교용)

## 다음 할 일

1. Optuna로 CatBoost/LightGBM 튜닝 (목표 OOF 0.665 이하)
2. LightGBM에 고객키 OOF target encoding 적용
3. CatBoost 여러 seed 평균
4. 신규 고객 성능 확인용 GroupKFold(고객 단위) 참고 실험
5. DACON에 제출해서 Public/Private 점수 기록
