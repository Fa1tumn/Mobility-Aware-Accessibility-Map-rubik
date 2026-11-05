# Rubik Pi CPU 현장 수집 (Area + Material, ONNX Runtime)

NPU 변환은 보류하고, Rubik Pi의 **CPU에서 ONNX Runtime으로 온디바이스 추론**을
돌려 현장 수집을 수행하는 절차입니다. 발표 설명: "Rubik Pi CPU에서 온디바이스 추론".

## 구성 요약

| 항목 | 값 |
| --- | --- |
| 모델 | Area best **epoch 7** (7클래스) + Material **epoch 10** (6클래스), LR-ASPP MobileNetV3 Large |
| 실행 backend | ONNX Runtime **CPUExecutionProvider** (`inference.CONFIG["backend"]="onnxruntime"`) |
| 전처리 | **PIL bilinear** 640×360 리사이즈 + RGB + ImageNet 정규화 (학습과 동일; cv2 리사이즈는 'other' 붕괴 때문에 사용 안 함) |
| 결합 정책 | 같은 프레임에 Area→Material 순차 실행, Area sidewalk ≥ 0.8 **AND** Material ≥ 0.8 픽셀만 채택, 없으면 `unknown` |
| 범위 제외 | 계단·파손 head |
| 수집 주기 | 목표 2초. 처리 시간만큼 sleep을 줄이고, 초과 시 즉시 다음 프레임(최신 프레임만 처리, 쌓임 방지) |
| 관측 시각 | 추론 완료 시각이 아니라 **프레임 수집 시각**(`frame_time`)을 저장 |

## 1. Pi 환경 준비

Pi에서 한 번만:

```bash
python3 -m venv ~/rubik-venv
~/rubik-venv/bin/pip install -r requirements-adapter.txt
# 카메라/센서용 패키지(기존 환경에 맞게): opencv, bleak 등
```

`models/area-surface-v3/`는 Git LFS로 관리됩니다. Pi에서 클론 시
`git lfs install && git lfs pull`로 실제 가중치를 받으세요(포인터 파일만 받으면
`onnxruntime`이 모델을 열지 못합니다).

## 2. CPU 처리 시간 측정 (수집 전에 먼저)

전처리 → Area → Material → 후처리 전체 시간을 Pi에서 실측합니다. 전처리 비용은
입력 해상도에 비례하므로, **실제 카메라 해상도(1280×720)** 기준으로 재는 것이
중요합니다. 폰 사진(수천만 화소) 원본으로 재면 전처리가 과대평가됩니다.

```bash
# 실제 카메라 해상도에 맞춘 샘플 이미지들로 측정
~/rubik-venv/bin/python benchmark_cpu.py --dir sample_frames_1280x720 --iters 120
```

출력은 단계별 avg/p50/p95/max, 전체 지연, 2초 목표 충족 여부, 연속 실행 드리프트
(발열 추정용)를 보여줍니다. **p95 total < 2000 ms**면 2초 수집에 여유가 있습니다.

참고 측정값(개발 노트북, Apple Silicon, ONNX Runtime CPU, 1280×720):
pre ~26 ms, area ~56 ms, material ~70 ms, post ~40 ms, **total 평균 192 ms / p95 236 ms**.
Pi CPU(QCS6490 Kryo)는 이보다 수 배 느릴 수 있으므로, 실제 2초 충족 여부는
**반드시 Pi에서 위 명령으로 확인**하세요.

## 3. 현장 수집 실행

```bash
export SUPABASE_URL="https://<project>.supabase.co"
read -rsp "Supabase secret key: " SUPABASE_SECRET_KEY; export SUPABASE_SECRET_KEY; echo
~/rubik-venv/bin/python main.py
```

- `inference.CONFIG["backend"]`는 기본 `"onnxruntime"`입니다. 변경 불필요.
- 매 레코드마다 콘솔에 `infer=…ms`, `loop=…ms`, 단계별 `timing_ms`가 출력됩니다.
  `Iteration exceeded target 2s…`가 자주 보이면 CPU가 2초를 못 맞추는 신호입니다.
- GPS는 폰이 Pi LAN IP의 8000 포트 `/gps`로 전송, IMU는 BLE, 저장은 SQLite
  (`observations.sqlite3`), 업로드는 매 iteration Supabase upsert.
- `SUPABASE_URL`/키가 없으면 업로드는 비활성화되고 로컬 저장만 됩니다.

## 4. 연속 실행 중 발열·메모리·지연 확인 (Pi에서)

수집을 돌리면서 별도 터미널에서:

```bash
# 발열(throttling)
watch -n 2 'vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone*/temp'
# 메모리
watch -n 5 'free -m; ps -o pid,rss,comm -p $(pgrep -f main.py)'
```

- 온도 상승과 함께 `timing_ms total`이 계속 증가하면 thermal throttling 가능성.
  `benchmark_cpu.py`의 "Sustained-run drift"로도 사전 확인됩니다.
- `ps`의 RSS가 지속 증가하면 메모리 누수 점검(정상이면 평탄).
- `loop` 시간이 2초를 반복적으로 넘으면 수집 주기가 밀립니다.

## 5. NPU 전환 판단 기준

다음 중 하나면 NPU(QNN) 전환을 검토합니다.

- Pi에서 `p95 total ≥ 2000 ms` (2초 목표 미달).
- 연속 실행 시 발열로 total이 크게 드리프트하여 주기가 불안정.
- 더 짧은 수집 주기(예: 1초)가 필요.

전환 시: 모델을 QCS6490용 DLC/context binary로 변환 → `surface_model.py`에
`QnnBackend` 추가 → `inference.CONFIG["backend"]="qnn"`. 전처리/후처리/정책과
저장·업로드 경로는 그대로 재사용됩니다(backend만 교체).

## Supabase 테이블 스키마

새 필드를 받도록 대상 테이블에 컬럼이 있어야 합니다. 없으면 추가하세요.

```sql
ALTER TABLE public.observations
  ADD COLUMN IF NOT EXISTS area_accept_ratio      double precision,
  ADD COLUMN IF NOT EXISTS material_accept_ratio  double precision,
  ADD COLUMN IF NOT EXISTS model_version          text,
  ADD COLUMN IF NOT EXISTS policy_version         text;
```

For an existing Supabase table, the uploader detects missing surface-metadata
columns and retries with the legacy fields. Full metadata remains in SQLite;
the fallback is remembered for that endpoint until the process restarts.
Apply `supabase_surface_metadata.sql`, restart collection, then rerun
`python3 upload_supabase.py` to backfill metadata from saved local records.
Other upload errors still stop the batch and remain eligible for retry.
