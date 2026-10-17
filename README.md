# 신용카드 사용자 연체 예측 (DACON 과제 11번)

고객 정보로 신용등급 `credit`(0/1/2)의 **확률**을 예측합니다. 평가지표는 다중분류 LogLoss(낮을수록 좋음)입니다.
대회 페이지: https://dacon.io/competitions/official/235713/data

## 폴더 구조

```
data/                  train.csv, test.csv, sample_submission.csv (DACON에서 받아서 넣기, git에는 안 올림)
src/eda.py             EDA: 클래스 비율, 결측치, 이상치, 중복 고객 확인 + 그림
src/baseline_dacon.py  DACON 공식 Baseline 노트북을 그대로 옮긴 것
src/features.py        전처리 + 파생변수
src/train.py           Baseline부터 단계별 실험 (5-fold) + 제출 파일 생성
src/report.py          발표용 그림 (실험 결과, 혼동행렬, 신뢰도 곡선, 변수 중요도)
src/improve.py         추가 실험: 고객 이력 target encoding, 동일 카드 정보, 고객 내 상대 위치
src/ceiling.py         정확도 상한 분석 (같은 입력의 정답 불일치, Top-2, 확신도별 정확도)
src/validity_check.py  점수 부풀림 점검 (조기 종료 누수, 중복 기록, 신규 고객 성능)
src/variable_analysis.py  중간보고서 탭 3 변수 분석 검증
src/group_experiments.py  중간보고서 탭 3 실험 A~E (LogReg / RandomForest / CatBoost)
docs/중간보고서_탭3_실험결과.md  위 두 스크립트의 결과 정리
outputs/results.csv    실험별 점수
outputs/submission_*.csv  DACON 제출 파일
outputs/figures/       그림
```

## 실행 방법

```bash
pip install -r requirements.txt
python src/eda.py      # 1분 이내
python src/train.py    # 약 20분 (CatBoost가 대부분, CPU 4코어 기준), 로그는 outputs/train_log.txt
python src/baseline_dacon.py  # 1분, 공식 Baseline 점수를 results.csv 맨 위에 추가
python src/report.py   # 1분 이내
python src/improve.py  # 약 6분, 최종 블렌드 + outputs/submission_blend.csv
python src/ceiling.py  # 1분 이내
python src/variable_analysis.py   # 1분 이내
python src/group_experiments.py   # 약 30분 (CatBoost가 대부분)
python src/plot_groups.py
```

## 사전 실험 결과 (Stratified 5-fold OOF, seed 42)

| 실험 | 바꾼 것 | LogLoss | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 00_dacon_baseline | **DACON 공식 Baseline** (직업 열 삭제, 구간화, RandomForest 기본값) | **1.0738** | 0.6169 | 0.4190 |
| 00_prior | 모든 행에 클래스 비율(0.122/0.237/0.641)만 예측 | 0.8824 | 0.6413 | 0.2605 |
| 01_rf_default_raw | 원본 + 정수 인코딩 + RandomForest 기본값 (구간화 안 함) | 0.9242 | 0.7027 | 0.5589 |
| 01b_rf_tuned_raw | 원본 그대로, RF 설정만 조정 (500그루, min_samples_leaf=3) | 0.7201 | 0.7135 | 0.5021 |
| 02_logreg_fe | 전처리 + 파생변수 + 로지스틱 회귀 | 0.8575 | 0.6454 | 0.2830 |
| 03_rf_fe | 전처리 + 파생변수 + RF 설정 조정 | 0.7262 | 0.7126 | 0.5020 |
| 03b_rf_default_fe | 전처리 + 파생변수 + RF 기본값 | 0.8785 | 0.7024 | 0.5397 |
| 04_lgbm_fe | 전처리 + 파생변수 + LightGBM | 0.7153 | 0.7166 | 0.5156 |
| 05_catboost_fe | 전처리 + 파생변수 + CatBoost | 0.7383 | 0.7100 | 0.4859 |
| 06_catboost_fe_id | 05 + 고객 식별키(범주형) | **0.6750** | 0.7363 | 0.5790 |
| 07_lgbm_fe_id | 04 + 고객 식별키(정수 코드) | 0.7121 | 0.7182 | 0.5190 |
| 08_ensemble | 06×0.93 + 07×0.07 (OOF로 가중치 탐색) | **0.6748** | 0.7358 | 0.5744 |

공식 Baseline 노트북 자체 출력은 1.0497입니다(원래 분할 seed 55, RandomForest seed 미지정). 같은 분할에서 RandomForest seed만 0~4로 바꾸면 1.0512~1.0699(평균 1.0617)이 나옵니다. 위 표의 1.0738은 다른 실험과 같은 분할(seed 42)로 돌린 값이라 이 값끼리 비교합니다.

정확한 값은 `outputs/results.csv`에 있습니다(라이브러리 버전이 다르면 소수 넷째 자리가 조금 달라질 수 있음).

### 결과에서 읽을 점

- **공식 Baseline(1.0738)은 "클래스 비율만 찍기"(0.8824)보다도 나쁩니다.** 기본 RandomForest는 잎이 1개 샘플까지 자라서 확률을 0이나 1에 가깝게 냅니다. 확신하고 틀리는 경우가 많아 LogLoss가 크게 나빠집니다. 여기에 직업 열 삭제와 구간화로 정보가 줄었고, 행 번호인 `index` 열까지 입력에 들어가 있습니다.
- 구간화하지 않고 원본 값을 쓰면 같은 RandomForest 기본값으로도 0.9242가 됩니다.
- **가장 큰 개선 1: 확률을 부드럽게 내는 설정.** `min_samples_leaf=3`만으로 0.9242 → 0.7201.
- **가장 큰 개선 2: 고객 식별키.** begin_month를 뺀 신상 정보가 같으면 같은 고객으로 보면 26,457행이 고객 8,759명입니다. test의 88.4%가 train에 있는 고객이라, CatBoost가 고객별 연체 성향을 학습하면 0.7383 → 0.6750으로 좋아집니다.
- 개별 범주형 변수(직업, 학력 등)는 credit 비율 차이가 거의 없어 단독으로는 신호가 약합니다.
- 변수 중요도(LightGBM gain) 1위는 카드 보유 기간 `card_months`(24%)이고, 그다음이 고객 식별키, 나이, 근속 관련 변수입니다.
- 최종 앙상블은 credit 0(신용 가장 좋음)의 재현율이 24%로 낮습니다. 확률 자체는 잘 맞습니다(`outputs/figures/calibration.png`).
- LightGBM에 고객키를 정수 코드로 넣는 것은 효과가 작습니다(0.7153 → 0.7121). 정수 코드 순서에 의미가 없기 때문이고, OOF target encoding이 다음 개선 후보입니다.

## 추가 실험: 정확도를 더 올릴 수 있나 (`src/improve.py`)

| 실험 | 추가한 것 | LogLoss | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 09_lgbm_te_relative | 고객·동일카드별 정답 비율(fold 밖에서만 계산) + 고객 안 카드 위치 | 0.6737 | 0.7365 | 0.5809 |
| 10_catboost_fullid_relative | CatBoost에 고객키 + 동일카드키 + 카드 위치 | 0.6741 | 0.7370 | 0.5814 |
| 11_blend | 09×0.5 + 10×0.3 + 06×0.2 | **0.6717** | **0.7372** | 0.5806 |

LogLoss는 0.6748 → 0.6717로 조금 더 좋아졌지만 정확도는 73~74%에서 더 오르지 않습니다.

## 정확도 90%는 가능한가 (`src/ceiling.py`)

이 데이터로 정확도 90%는 현실적으로 어렵습니다. 수학적으로 증명한 것은 아니고, 아래 근거로 판단한 것입니다.

- **입력이 완전히 같은데 정답이 다른 행이 많습니다.** 신상 정보와 발급월까지 모두 같은 행이 4,497행(2,144그룹)인데, 그룹 안 정답이 모두 같은 경우는 67.1%뿐입니다. 이 행들은 정답을 미리 알고 찍어도 최대 84.0%입니다. 어떤 모델도 같은 입력에 다른 답을 낼 수 없기 때문입니다. 단, 이 상한은 전체의 17%인 이 행들에만 해당합니다.
- 서로 다른 방법 6가지(RF, LightGBM, CatBoost, 고객 이력, 동일 카드, 앙상블)가 모두 73~74%에서 멈춥니다.
- 학습에 쓴 train으로 다시 채점하면 97.3%가 나오지만, 답을 외운 것이라 의미가 없습니다. test나 DACON 리더보드에서는 그대로 무너집니다.

대신 정직하게 보고할 수 있는 90%대 수치는 다음과 같습니다.

| 지표 | 값 | 뜻 |
| --- | --- | --- |
| Top-2 정확도 | **92.9%** | 정답이 모델이 꼽은 상위 2개 등급 안에 있는 비율 |
| 확신도 상위 14.9% 고객 정확도 | **90% 이상** | 모델이 확신하는 3,941명만 보면 90% 이상 맞힘 |
| 확신도 상위 47.7% 고객 정확도 | 85% 이상 | 절반 가까운 고객은 85% 이상 |

## 검증 점검: 점수가 부풀려지지 않았나 (`src/validity_check.py`)

| 점검 항목 | 기존 방식 | 정직하게 다시 잰 값 | 판단 |
| --- | --- | --- | --- |
| 조기 종료 시점을 검증 fold로 고른 영향 (CatBoost E+) | 0.6746 | 0.6755 (학습 fold 안에서 시점 선택 후 전체 재학습) | 영향 거의 없음 |
| 앙상블 가중치를 같은 OOF로 고른 영향 (11_blend) | 0.6717 | 0.6717 (가중치는 4/5로 고르고 1/5로 채점) | 영향 없음 |
| 검증 행과 같은 고객이 학습 쪽에 있는 비율 | 랜덤 5-fold 84.7% | test 88.4% | 검증이 test보다 약간 어려움 → 낙관적이지 않음 |
| 검증 행과 완전히 같은 행이 학습 쪽에 있는 비율 | 랜덤 5-fold 13.9% | test 17.2% | 위와 같음 |
| **처음 보는 고객 (고객 단위 GroupKFold)** | 0.6746 / 정확도 73.7% | **0.8088 / 정확도 69.0%** | **고객 식별 효과는 기존 고객에게만 통함** |

- DACON test는 88%가 기존 고객이라 0.67대 점수가 test 성능을 잘 대변합니다.
- 처음 보는 고객만 놓고 보면 0.81 수준입니다. 이 경우에도 찍기(0.8824)와 공식 Baseline(1.0738)보다는 낫습니다.
- `n_cards`, `card_rank`는 train과 test의 입력값(정답 제외)을 합쳐서 계산합니다. 대회에서는 허용되는 방식이지만, 실제 서비스라면 그 시점까지의 데이터로만 계산해야 합니다.

## 제출 파일

- `outputs/submission_blend.csv` : **최종 후보** (OOF LogLoss 0.6717, 정확도 73.7%)
- `outputs/submission_ensemble.csv` : 이전 최종 (OOF 0.6748)
- `outputs/submission_catboost_id.csv` : CatBoost 단독 (OOF 0.6750)
- `outputs/submission_dacon_baseline.csv` : DACON 공식 Baseline (비교용)

## 다음 할 일

1. Optuna로 CatBoost/LightGBM 튜닝 (목표 OOF 0.665 이하, 공식 Baseline 대비 38% 감소)
2. LightGBM에 고객키 OOF target encoding 적용
3. CatBoost 여러 seed 평균
4. 신규 고객 성능 확인용 GroupKFold(고객 단위) 참고 실험
5. DACON에 제출해서 Public/Private 점수 기록
