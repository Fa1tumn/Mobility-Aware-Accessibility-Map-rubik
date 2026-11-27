# Pi replay 측정: 처리 간격의 근거

`compare_replay.py`는 녹화 사진으로 **정확도**를 PC에서 비교한다. 이 문서의
`pi_replay_benchmark.py`는 같은 녹화 사진을 **Pi에서 실시간 속도로** 다시 흘려 보내
처리 간격마다 **비용**을 잰다. 현장에 다시 나가지 않아도 되고, 두 결과를 같은 구간 번호로
맞대어 볼 수 있다.

## 재는 것

| 항목 | 내용 |
|---|---|
| 단계별 시간 | 사진 한 장의 전처리, Area, Material, 후처리, 합계(ms). 품질 검사(밝기·선명도)와 JPEG 디코딩도 따로 |
| 처리하지 못한 사진 | 구간마다 받은 사진 - 분석한 사진. 분석하는 동안 새 사진에 밀려 **읽지도 못한** 사진은 따로 |
| CPU·메모리·온도 | 1초마다 이 프로세스 CPU(코어 하나 = 100%), 시스템 CPU(전체 = 100%), RSS, 최고 온도, 최저 CPU 클럭 |
| 결과 지연 | 구간이 끝난 뒤 그 구간 결과가 나오기까지 걸린 시간 |
| PC 결과와 비교 | `four_method_replay/results.json`이 있으면 같은 구간에서 같은 사진을 골랐는지, 판정이 같은지 |

## 분석 방식 (`--policy`)

| 값 | 동작 | 대응 |
|---|---|---|
| `live` | `main.py`와 같은 `camera_inference.iter_windows`. 3장마다 가장 선명한 사진을 분석하고 구간 결과는 투표 | 현재 수집 코드 |
| `timed` | 구간마다 한 장, 구간의 마지막 사진 | `compare_replay` 방법 A/B |
| `quality` | 구간마다 한 장, 밝기 검사 + 선명도·자이로 순위로 고른 사진 | `compare_replay` 방법 C/D |

모델은 `--model-dir`로 하나만 고른다. `area-surface-v3`이면 PC 방법 A/C,
`finetuned-v2-onnx`이면 B/D와 비교한다. 두 모델은 Material 가중치만 다르고 구조가 같으므로
비용은 같다고 보고, 필요하면 두 번 돌려 확인한다.

## Pi에서 실행

```bash
cd ~/test   # 저장소 위치
# 기본: live 방식, 간격 1·2·4·8초, 간격마다 5분, 간격 사이 60초 식힘 (약 23분)
.venv/bin/python pi_replay_benchmark.py recordings/campus_20261125T075508Z

# 연구 비교용: 세 방식 × 2·4·8초 (약 55분)
.venv/bin/python pi_replay_benchmark.py recordings/campus_20261125T075508Z \
    --policy timed quality live --intervals 2 4 8

# 발열까지 보려면 녹화 끝까지(약 16분씩)
.venv/bin/python pi_replay_benchmark.py recordings/campus_20261125T075508Z --intervals 2 --duration-s 0
```

자주 쓰는 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--backend torch\|onnxruntime` | `INFERENCE_BACKEND` (torch) | 두 실행 방식을 각각 재서 비교 |
| `--torch-threads N` | 2 | PyTorch 계산 스레드 |
| `--surface-region wheel_ahead` | 없음 | Area를 건너뛰는 바퀴 앞 영역 방식 |
| `--source-fps 30` | 녹화 간격(약 5장/초) | 라이브 카메라(30장/초)의 디코딩 부하 흉내. 같은 사진이 반복된다 |
| `--start-s`, `--duration-s` | 0, 300 | 녹화의 어느 부분을 얼마나 쓸지. `--duration-s 0`은 끝까지 |
| `--cooldown-s` | 60 | 앞 측정의 발열이 다음 측정에 섞이지 않게 쉬는 시간 |

측정 중에는 `main.py`나 다른 무거운 작업을 끈다. Ctrl+C로 멈추면 끝난 측정까지 저장한다.

## 결과

`recordings/<녹화>/pi_benchmark/<UTC 시각>/`에 저장된다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 간격별 요약 표, 단계별 시간 표, PC 비교 표. 보고서에 바로 옮길 수 있다 |
| `summary.json` | 같은 내용과 기기·버전·커밋 정보 |
| `windows.csv` | 구간마다 받은 사진, 읽은 사진, 분석 수, 처리 못 한 수, 버려진 수, 처리 시간, 지연, 판정 |
| `inferences.csv` | 분석 한 번마다 단계별 ms |
| `system.csv` | 1초마다 CPU, RSS, 온도, 클럭 |

## 간격을 정하는 방법

1. **감당 여부:** `감당`이 `예`인 간격만 남긴다. 구간 처리 p95와 결과 지연 p95가 모두
   간격보다 작다는 뜻이다. 아니면 결과가 점점 밀린다.
2. **발열:** 같은 간격에서 시간이 갈수록 `inferences.csv`의 합계 ms가 늘고 `system.csv`의
   클럭이 떨어지면 발열로 느려진 것이다. 이 경우 녹화 끝까지 잰 값으로 판단한다.
3. **정확도와 위치 해상도:** 남은 간격 중에서 `compare_replay.py` 결과(재질 판정, 바뀌는
   지점 검출)와 간격당 이동 거리(약 1m/s면 2초에 약 2m)를 비교해 고른다.
4. **PC 결과와 같은지:** `같은 사진일 때 같은 판정`이 비교 구간과 거의 같아야 PC 정확도
   결과를 Pi에 그대로 적용할 수 있다. 같은 사진 선택 수가 적으면 Pi가 바빠 사진을 놓친 것이다.

## 한계

실제 수집에서 함께 도는 카메라 인코딩(GStreamer), IMU(BLE), GPS 수신, Supabase 업로드는
들어가지 않는다. 짧게 `main.py`를 실제로 돌려 콘솔의 `timing_ms`와 `frames=분석/받음`을
이 측정의 `live` 결과와 비교해 차이를 적는다.
