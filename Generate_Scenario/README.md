# 개별 사기 시나리오 데이터 생성

프로젝트 루트에서 사용자 Python 환경으로 실행한다.

```powershell
.\demo_py\.venv\Scripts\python.exe demo_py/Generate_Scenario/generate_scenario1.py
.\demo_py\.venv\Scripts\python.exe demo_py/Generate_Scenario/generate_scenario3.py
```

기본값: 2018-04-01부터 183일, 고객 5,000명, 터미널 10,000개.
각 출력은 `demo_py/data/handbook/scenario1_only`와 `scenario3_only`다.
이미 있는 출력 폴더는 덮어쓰지 않는다. `--output-dir`로 새 폴더를 지정할 수 있다.

- 시나리오 1: 금액 220 초과 거래에만 사기 라벨을 붙인다.
- 시나리오 3: 매일 전체 고객 프로필에서 3명을 선택한다. 해당 고객들의 이후 14일 거래를 모아 1/3(소수점 버림)을 추출하고 금액을 5배로 변경해 사기로 지정한다. 정상 라벨 0, 사기 시나리오 번호 3만 생성한다.
- 시나리오 3에서는 220 초과 조건과 터미널 사기 시나리오를 적용하지 않는다.
- 공식 코드와 동일하게 마지막 관측일에는 새 피해 고객을 선택하지 않으며, 기간 말미의 14일 구간은 데이터 끝에서 잘린다. 중복 선정된 거래의 금액은 다시 5배가 될 수 있다.
- 사기 주입 후 고객·터미널 통계를 새로 계산하고, 날짜별 23열 pickle 및 manifest.json을 저장한다.
- 시나리오 3만 생성해도 시간별 피해 고객 교체가 발생하지만, 지정 시점의 별도 드리프트 전환 기능은 없다.

생성 후 모델 실행:

```powershell
.\demo_py\.venv\Scripts\python.exe demo_py/run.py --strategies N --data-dir demo_py/data/handbook/scenario3_only --output-dir demo_py/results/scenario3_only
```

공식 출처: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html#fraud-scenarios-generation
