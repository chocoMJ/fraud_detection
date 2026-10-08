# Python streaming Active Learning baseline

Carcillo 관련 저장소의 `main.R`을 참고한 **오류 수정 baseline**이다.
원 논문의 성능 수치 재현 또는 R과 동일한 수치 결과를 보장하지 않는다.
원본 R 파일과 데이터는 수정하지 않는다. Handbook 공개 합성 데이터 연결은 아래 절차를 사용한다.

## 구조

```text
demo_py/
  data.py             일별 CSV 및 Handbook pickle 직접 로딩·검증
  model.py            초기 균형 결정트리 앙상블·후속 Random Forest
  strategies.py       N/R/U 고유 카드 선택
  evaluation.py       거래 예측·카드 조사 성능
  simulation.py       시간 순서·학습 윈도·라벨 공개 관리
  run.py              CLI 실행·결과 저장
  Convert_CSV.py     로컬 pickle을 확인용 CSV로 내보내기 (선택 도구)
  Generate_Scenario/
    generate_scenario1.py  시나리오 1 전용 transformed 데이터 생성
    generate_scenario3.py  시나리오 3 전용 transformed 데이터 생성
  vendor/            공식 생성·특성 가공 함수와 출처 기록
  tests/
    test_baseline.py  주요 실험 규칙 검증
    test_handbook.py  데이터 매핑·pickle·확인용 CSV 검증
  data/              입력 데이터 및 확인용 CSV
  results/           데이터셋·실험별 결과
  requirements.txt
```

## 실행

사용자가 설치한 Python으로 직접 실행한다. 프로젝트 루트에서 최초 한 번 가상환경과 패키지를 준비한다.
현재 PC에는 Python 3.12가 설치되어 있다.

```powershell
python -m venv demo_py/.venv
.\demo_py\.venv\Scripts\python.exe -m pip install -r demo_py/requirements.txt
```

이후에는 다음 명령으로 N 기준선을 실행한다. 가상환경 활성화나 PowerShell 실행 정책 변경은 필요 없다.

```powershell
.\demo_py\.venv\Scripts\python.exe demo_py/run.py --strategies N
```

VS Code/PyCharm에서는 Python 인터프리터를 `demo_py/.venv/Scripts/python.exe`로 선택하고
`run.py`를 실행한다. 인자 없이 실행하면 N/R/U를 모두 실행한다.
이하 `python` 명령은 해당 가상환경의 Python 또는 필요한 패키지가 설치된 Python을 뜻한다.
`run_local.ps1`은 선택적인 호환용 도구이며 로컬 가상환경, 없으면 PATH의 Python을 사용한다.
Codex 내장 Python이나 `.deps`는 실행에 사용하지 않는다.

## Handbook 데이터 연결

현재 기본 입력은 공식 생성 코드로 만든 `data/handbook/scenario1_only`의 전체 일별 pickle이다.
현재 2018-04-01부터 2018-09-30까지 183일분이며 사기 시나리오 1만 적용한 **합성 데이터**다.
미리 다운로드한 로컬 pickle 파일을 직접 사용한다. 코드에 다운로드 기능은 없다.

```powershell
# 연결 확인용: 첫 10일 중 7일 초기 학습 + 3일 평가, RF 20개 트리
python -m demo_py.run --max-days 10 --forest-trees 20 --output-dir demo_py/results/handbook_smoke

# 183일 전체, 기존 기본 모델 설정으로 N/R/U 비교
python -m demo_py.run
```

일반 Python 환경에서는 다음과 같다.

```powershell
python -m demo_py.run
python -m unittest demo_py.tests.test_baseline demo_py.tests.test_handbook -v
```

기본 결과 경로는 `demo_py/results/scenario1_only`다.
실행기는 Handbook 전용이며 데이터셋 선택 옵션은 없다. `--data-dir`을 지정하면 선택한 데이터 경로를 우선한다.
Handbook의 기본 입력은 `data/handbook/scenario1_only/YYYY-MM-DD.pkl`이다.
폴더의 모든 pickle을 날짜순으로 읽으며, `--max-days`를 지정할 때만 기간을 제한한다.
pickle을 직접 읽어 메모리에서 12개 특성을 선택하고 CUSTOMER_ID→CARD_ID,
TX_FRAUD→y로 매핑하므로 CSV 생성은 실행의 필수 단계가 아니다.
별도로 받은 공식 가공본은 `--data-dir path/to/pkl_directory`로 지정한다.
공식 raw 데이터가 아니라 특성이 추가된 `simulated-data-transformed` 파일을 사용한다.
pickle은 신뢰할 수 있는 공식 출처의 파일만 입력한다.
기존 변환 CSV를 사용하려면 `--input-format csv`를 지정한다.
여러 캐시 revision이 있으면 `--data-dir`로 사용할 revision 폴더를 명시한다.
기간을 확장하려면 추가로 받은 pickle을 입력 폴더에 넣는다. 공개 범위는 2018-04-01~2018-09-30이다.
기존 데이터와 설정이 다른 변환은 같은 출력 폴더를 덮어쓰지 않는다.

```text
data/handbook/
  source/<commit SHA>/2018-04-01.pkl ...   공식 원본 캐시
  daily/day001.csv ... day030.csv         baseline 입력
  daily/row_maps/day001.csv ...           각 입력 행의 원본 거래 ID·시각
  daily/manifest.json                     날짜 매핑·소스 URL·체크섬·건수·특성
```

변환 매핑:

| 원본 | baseline | 의미 |
|---|---|---|
| CUSTOMER_ID | CARD_ID | 고객을 조사 단위로 사용하는 대리 식별자. 실제 카드 ID는 아님 |
| TX_FRAUD | y | 정상 0, 사기 1 |
| TX_AMOUNT | 동일 이름 | 거래 금액 |
| TX_DURING_WEEKEND / TX_DURING_NIGHT | 동일 이름 | 주말·야간 여부 |
| CUSTOMER_ID_NB_TX / AVG_AMOUNT의 1·7·30일 특성 | 동일 이름 | 고객의 거래 횟수·평균 금액 6개 |
| TERMINAL_ID_NB_TX의 1·7·30일 특성 | 동일 이름 | 공식 전처리의 지연 창 기준 터미널 거래 횟수 3개 |

입력 특성은 총 12개다. 의미를 잃는 `x1` 등의 이름으로 바꾸지 않는다.
`TRANSACTION_ID`, `TERMINAL_ID`, 날짜·시간 식별 컬럼, `TX_FRAUD_SCENARIO`는 입력에서 제외한다.
`TERMINAL_ID_RISK_*` 3개도 제외한다. 공식 risk 특성은 지연된 전체 과거 사기 정답을 이용하지만,
현재 AL baseline은 조사한 거래의 정답만 이후 학습에 공개하므로 그대로 사용하면 가용 라벨 정책이 달라진다.
이는 Handbook 전처리가 잘못됐다는 의미가 아니라 현재 실험의 정보 가용성에 맞춘 선택이다.
사용하는 고객 통계는 현재 거래를 포함하는 과거 rolling 값이며 미조사 거래의 금액·횟수 관측은 허용한다.
완성된 공식 전처리 값을 가져오기 때문에 30일 통계를 다운로드한 구간 안에서 다시 계산하지 않는다.

원본 commit과 Git blob 해시를 검증한 뒤 공식 pickle을 읽는다. 구형 pandas 숫자 인덱스 형식은
호환 로더로 처리한다. 임의 URL이나 임의 pickle을 받는 변환기는 제공하지 않는다.
준비가 모두 끝난 뒤에만 daily 폴더가 생성되며, 원본 다운로드는 실패 후 재시도할 때 재사용한다.
run 결과의 `config.json`에도 manifest와 실제 평가에 사용한 입력 파일 체크섬을 기록한다.

출처: [공식 데이터 저장소](https://github.com/Fraud-Detection-Handbook/simulated-data-transformed),
[공식 특성 생성 설명](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html).

2026-10-05 확장: 공식 커밋의 183개 pickle을 Git blob 해시로 검증했다.
전체 1,754,155건(사기 14,681건)을 현재 로더로 읽어 날짜 연속성, 거래 ID 중복,
필수 열 및 숫자 검증을 통과했다. 결과는 `results/handbook_data_validation/summary.json`에 있다.
이는 데이터 검증이며 183일 전체 학습·평가 실행은 아니다.

과거 연결 검증 결과 (2026-09-28): 2018-04-01~04-30의 288,062건 중 사기 1,702건을 변환했다.
기존 테스트 8개와 변환 테스트 4개, 총 12개가 통과했다. 실제 데이터 30일 전체가 로더의
컬럼·라벨·유한 숫자 검증을 통과했으며, 첫 10일로 7일 초기 학습 + 3일 평가를 실행했다.
N/R/U 모두 완료했고 시드 42, RF 20개 트리의 결과는 `results/handbook_smoke`에 있다.
이는 연결 검증용 실행이다. Handbook 30일 전체를 RF 500개로 평가한 결과는 아직 생성하지 않았다.

## 일반 Python 설치와 추가 실행 옵션

Python 3.10 이상 환경에서 프로젝트 루트를 작업 경로로 사용한다.

```powershell
python -m pip install -r demo_py/requirements.txt
python -m demo_py.run
python -m unittest demo_py.tests.test_baseline -v
```

직접 `python demo_py/run.py`로 실행해도 된다. 기본 데이터·출력 경로는 스크립트 위치를 기준으로 결정된다.

```powershell
python -m demo_py.run --strategies N R U --seeds 42 43 44 --output-dir demo_py/results/multi
python -m demo_py.run --strategies U --forest-trees 20 --output-dir demo_py/results/smoke
python -m demo_py.run --input-format csv --data-dir path/to/daily_csv
```

아래 CSV 형식 설명은 `--input-format csv`로 기존 Handbook 모델 입력용 CSV를 읽을 때 적용된다.
파일명은 `day<number>.csv`이며 숫자 순으로 읽는다. 날짜 번호 차이는 실제 일수 차이로 취급한다.
`CARD_ID`는 문자열, `y`는 0/1, 입력 특성은 결측·무한대 없는 숫자여야 한다.
기본적으로 `CARD_ID`, `y`를 제외한 모든 컬럼을 입력 특성으로 사용한다.
다른 데이터에서는 `--features`로 사용할 컬럼을 명시해 거래 식별자·시각·누출 변수를 제외한다.

## 실험 규칙 및 원본과의 차이

1. **초기 7개 일별 파일은 전체 정답이 알려진 seed 데이터**다. 평가 결과에 포함하지 않는다.
2. 첫 평가일에는 이전 학습 윈도에 있는 seed 데이터로 초기 앙상블을 만든다.
   일별 사기 거래 전체와 같은 수의 정상 거래를 샘플링해 결정트리 10개씩 학습한다.
   정상 거래가 부족하면 복원 추출, 단일 클래스 날짜는 해당 클래스 데이터 전체를 사용한다.
3. 이후 매일 최근 7일의 확보된 라벨로 Random Forest를 새로 학습한다.
   트리 수 500, `max_features=sqrt`, bootstrap 사용, class weight 없음이다.
   초기 전체 라벨도 날짜가 윈도에서 빠질 때까지 유지한다.
   이는 원본의 초기 기간에서 선택한 거래만 이후에 사용하는 방식과 **명시적으로 다른 정책**이다.
4. 당일 예측 → 카드 선택 → 선택 카드의 당일 전체 거래 라벨 공개 → 다음 날 반영 순서다.
   당일·미래 정답은 예측이나 카드 선택에 사용하지 않는다. 평가는 시뮬레이터의 숨겨진 전체 정답으로 수행한다.
5. 학습 데이터는 날짜 키로 관리한다. 데이터가 없는 날도 일수는 경과한다.
   학습 윈도가 비면 직전 모델을 유지하고 `model_reused=True`로 기록한다.
   첫 평가일의 학습 데이터가 없으면 명시적으로 오류를 낸다.
6. 예측 점수를 거래에 대응시킨 상태로 카드별 점수를 집계한다.
   위험도는 카드 거래의 **최대 사기 확률**, 불확실성은 **최소 |p-0.5|**다.
   동점은 카드 ID 문자열 순으로 결정한다.
7. `N`: 위험도 상위 100개. `R`: 상위 95개 + 남은 카드 중 무작위 5개.
   `U`: 상위 95개 + 남은 카드 중 가장 불확실한 5개.
   `--budget`, `--exploration`으로 변경 가능하다. 고유 카드 수가 예산보다 적으면 모두 선택하며
   이 경우에도 가능한 범위에서 탐색 슬롯을 먼저 확보한다. 동일 카드를 중복 조사하지 않는다.
8. 사기 카드 차단은 기본적으로 끈다. `--block-cards`를 주면 평가 기간 중 조사로 사기가 확인된
   카드의 거래를 다음 날부터 제외한다. 차단된 거래는 이후 평가 모집단에서도 제외된다.
   초기 seed의 사기 카드는 자동 차단하지 않는다. 옵션을 켜면 전략별 평가 모집단이 달라질 수 있다.
9. `x1`을 포함한 명시적 특성 전체를 사용한다. R의 위치 기반 첫 두 컬럼 제거를 재현하지 않는다.
10. 의사 라벨링 전략은 포함하지 않는다. R의 라이브러리와 scikit-learn의 분할·가지치기 등은 다르며,
    초기 트리와 후속 RF의 수치적 동일성을 목표로 하지 않는다.

## 평가 및 결과

`results/<dataset>/`에 전략·시드별로 다음을 저장한다. 같은 출력 경로로 다시 실행하면 같은 이름의 파일을 덮어쓴다.

- `daily_U_seed42.csv`: 평가일별 성능, 실제 조사 수, 라벨 확보량, 학습에 사용한 날짜, 차단 제외량.
- `selected_U_seed42.csv`: 조사 카드 ID, 위험도, 불확실성, 선택 이유, 평가용 카드 정답.
- `summary.csv`: 전략·시드별 일별 지표 평균 및 조사·발견 카드-일 합계.
- `config.json`: 설정, 특성 목록, 라이브러리 버전, 입력 파일 SHA-256.

카드 정답은 **그날 해당 카드 거래 중 하나라도 사기이면 1**이다.
`normalized_card_precision`(NCP)은 발견한 사기 카드 수를
`min(설정된 하루 조사 예산, 당일 평가 대상 사기 카드 수)`으로 나눈 값이다.
위험도 기반 선정과 탐색으로 선정한 카드를 모두 포함한다.
분모가 0인 날은 NaN이며 평균에서 제외한다.
`summary.csv`에는 일별 NCP의 평균을 `mean_normalized_card_precision`으로 저장한다.
`card_precision`은 실제 조사한 고유 카드 중 사기 카드 비율, `card_recall`은
당일 평가 모집단의 사기 카드 중 발견한 비율이다.
동일 카드가 여러 날 나타날 수 있으므로 요약 합계는 전체 기간 고유 카드 수가 아닌 **카드-일** 수다.

카드(고객 ID) 단위로 `roc_auc`와 `average_precision`을 기록한다.
하루의 고객별 최대 거래 사기 확률을 점수로, 사기 거래가 하나라도 있으면 1을 정답으로 사용한다.
조사한 카드뿐 아니라 당일 평가 대상 전체 카드의 순위를 평가한다.
열 이름은 유지하며 `config.json`의 `evaluation`에 평가 단위와 집계 방식을 기록한다.
이 변경 전에 생성한 결과 CSV는 거래 단위 ROC-AUC/AP이므로 새 결과와 구분해야 한다.
**Average Precision은 원본 PRROC의 PR-AUC와 동일한 적분 방식이 아니므로 `auprc`로 표기하지 않는다.**
ROC-AUC는 카드 정답이 단일 클래스인 날, AP는 사기 카드가 없는 날에 NaN이다.
조사 카드가 없으면 precision, 사기 카드가 없으면 recall도 NaN이다.
요약 평균은 NaN을 제외한 일별 macro 평균이다. 여러 시드 결과도 시드별로 따로 보존한다.

## 검증

테스트는 다음을 확인한다: 정렬 후 점수 대응, 카드 중복 방지, 조사 예산과 작은 후보군,
카드 단위 정답 집계, 당일·미래 라벨 변경이 당일 선택에 영향을 주지 않음,
선택 라벨만 후속 학습에 포함, 학습 윈도 날짜, 시드 재현성, 단일 클래스,
빈 예측 입력, 다음 날 카드 차단, 파일 숫자 정렬과 카드 ID 앞자리 0 보존.

이는 원 논문과의 실험 설계 대조나 성능 재현 검증을 대신하지 않는다.

2026-09-28 검증: Python 3.12.14, NumPy 2.5.3, pandas 3.0.1, scikit-learn 1.9.1에서
8개 테스트 통과. 기본 설정·시드 42로 기존 `syntdf` 30일 파일을 실행했으며,
N/R/U 각각 8~30일의 23일 평가를 완료했다. 각 전략의 조사 카드-일 수는 2,300이고,
중복 카드 선택 없음, 일별 예산 100 준수, 직전 7일 학습 범위를 결과 파일에서도 확인했다.
이는 단일 시드 실행 검증이며 전략 간 우열에 대한 결론은 아니다.

## 출처

참고 코드: `../2018_activeLearning/StreamingActiveLearningStrategies/main.R`.
원본 저장소의 GPL v3 라이선스 전문을 `LICENSE`에 보존한다.
이 Python 수정본은 2026-09-28에 작성했으며 변경 사항은 위 실험 규칙에 기록했다.

## 데이터 확인용 CSV 내보내기

`Convert_CSV.py`는 로컬 `.pkl` 하나 또는 폴더 안의 `.pkl` 파일들을 CSV로 내보내는
독립 실행 도구다. 다운로드나 학습을 수행하지 않는다. 메인 실험에서는 사용하지 않는다.
원래 컬럼 이름, 전체 컬럼, 행 순서를 유지한다. 인덱스는 저장하지 않으며 CSV는 Python 자료형을 보존하지 않는다.
UTF-8 BOM으로 저장해 Excel에서도 열 수 있다. 기존 CSV를 덮어쓰려면 `--overwrite`를 지정한다.

```powershell
python -m demo_py.Convert_CSV demo_py/data/handbook/source/6e3ca5849b4681430388056d3f1dcfb41d4e8269/2018-04-01.pkl
# 일반 Python 환경: 폴더 전체 변환
python -m demo_py.Convert_CSV path/to/pkl_folder --output-dir path/to/inspection_csv
```

기본 출력 폴더는 `demo_py/data/handbook/inspection_csv`다. 이 CSV는 사람이 확인하는 용도로,
모델 입력용 `dayNNN.csv`와 형식이 다르다. pickle은 신뢰할 수 있는 출처의 파일만 사용한다.

테스트 전체 실행: `python -m unittest discover -s demo_py/tests -t . -v`.
기존 실험 결과도 `results/`의 실험별 하위 폴더로 모았으며, 기존 결과 파일 내용은 유지했다.

## 시나리오 1만 적용한 데이터 생성

공식 Handbook의 거래 생성·특성 가공 함수 8개를 `vendor/handbook.py`에 보존했다.
출처 커밋과 원본 노트북 해시는 `vendor/handbook_source.json`, 라이선스는 `LICENSE`에 있다.
`Generate_Scenario/generate_scenario1.py`가 이 함수들을 호출하며 pandas 버전 차이를 피하도록 그룹 처리를 명시적으로 수행한다.
생성 과정에 인터넷이나 기존 pickle은 필요 없다.

```powershell
# 현재 PC: 기본 183일, 고객 5,000명, 터미널 10,000개
python -m demo_py.Generate_Scenario.generate_scenario1

# 일반 Python 환경
python -m demo_py.Generate_Scenario.generate_scenario1

# 기간·규모를 지정하려면 새 출력 폴더 사용
python -m demo_py.Generate_Scenario.generate_scenario1 --days 30 --customers 5000 --terminals 10000 --output-dir demo_py/data/handbook/scenario1_30days

# 생성한 데이터로 N 기준선 실행 (전체 183일)
python -m demo_py.run --data-dir demo_py/data/handbook/scenario1_only --strategies N --output-dir demo_py/results/scenario1_N
```

- 처음부터 거래를 생성하고 `TX_AMOUNT > 220`일 때만 `TX_FRAUD=1`, `TX_FRAUD_SCENARIO=1`로 지정한다. 나머지는 모두 0이다.
- 시나리오 2/3은 호출하지 않고, 거래 금액도 사기 생성 단계에서 바꾸지 않는다.
- 공식 기본값(시작일 2018-04-01, 반경 5)과 시드(고객 0, 터미널 1, 거래는 고객 ID)를 사용한다.
- 야간·주말, 고객 통계, 터미널 통계를 새로 계산해 공식 transformed와 같은 23열로 저장한다.
- 공식 코드대로 고객 통계는 현재 거래를 포함하며, 야간은 hour <= 6이다. 터미널 사기 비율은 7일 라벨 지연을 적용한 과거 창으로 계산한다.
- 기본 출력은 `data/handbook/scenario1_only/YYYY-MM-DD.pkl`이다. `manifest.json`에 설정·시드·출처·건수·파일 해시를 기록한다.
- 기존 출력 폴더는 덮어쓰지 않는다. 오류가 나면 다른 `--output-dir`을 지정한다.
- 현재 demo 모델은 23열 중 기존 12개 입력 특성만 사용한다. 라벨 기반 터미널 위험도 3개는 생성되지만 모델 입력에는 제외된다.
- 원래 3개 시나리오 데이터는 `--data-dir demo_py/data/handbook/source`로 선택할 수 있다. 기본 실행은 `scenario1_only` 전체를 사용한다.
- 이 데이터는 고정된 금액 규칙의 대조 실험용이다. 시간에 따른 규칙 전환이나 드리프트 종류 선택 기능은 포함하지 않는다.
