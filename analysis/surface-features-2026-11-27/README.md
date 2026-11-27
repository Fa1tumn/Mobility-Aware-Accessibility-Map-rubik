# 휠체어 현장 실험 IMU 노면 특징 (2026-11-27)

2026-11-27 휠체어 현장 실험 기록으로 IMU 진동 특징을 뽑았다. 이 특징으로 두 가지를 정리했다.
붉은 노면과 블록을 가르는 특징 후보를 골랐고, 라벨 붙은 블록 길마다 진동 특성과 편한 길 안내용 임시 비용을 만들었다.
원본 기록(recordings: frames, frames_v3_2s_wheel_ahead, imu, coast)은 용량이 커서 넣지 않았다.

## 먼저 볼 파일

| 파일 | 내용 |
|---|---|
| `block_comfort_map.html` | 블록 길 지도. 2초 창 점을 편안함 비용으로 색칠하고, 아래에 라벨별 특성 표를 둔다. 브라우저로 연다 |
| `block_profiles.csv` | 라벨 20개의 진동 특성, 진동수, 편안함 순위·등급, 안내 문구 |
| `feature_candidates.md` | 붉은 노면과 블록을 가르는 특징 후보 정리 |
| `index.html` | 전체 분석 보고서 (coast 관성 시험, imu 실험, 사진 표본) |

## 올린 파일 전체

### 블록 길 특성과 편한 길 (`block_profiles.py`)

| 파일 | 내용 |
|---|---|
| `block_profiles.csv` | 라벨별 창 수, 거리, 수직 진동 RMS(중앙값·사분위), 10m당 큰 충격 수, 첨도와 진동 모양, 주 진동수 사분위, 대역 비율, 1m당 반복 수, 속도별 예상 진동수 대역, 편안함 비용·순위·등급, 안내 문구, 위치 범위 |
| `block_profiles.json` | 같은 내용과 계산 기준(기준값, 무게, 등급 경계) |
| `block_comfort.geojson` | 이동 중이고 끊김 없는 2초 창 점 2,219개. 라벨, 속도, 수직 RMS, 충격 수, 첨도, 진동수, 편안함 비용, 라벨 등급과 안내 문구 |
| `block_comfort_map.html` | 위 GeoJSON 지도와 라벨 표 |

### 특징 후보 (`feature_candidates.py`)

| 파일 | 내용 |
|---|---|
| `feature_candidates.md` | 후보 순위, 묶음별 대표, 회차마다 방향이 바뀌는 특징, 속도 영향, 중복 묶음 |
| `feature_candidates.csv` | 특징별 회차 delta, 중앙값, 회차 안정성, 속도 상관, coast 비교, 중복 전체 |
| `plots/candidates_top.png` | 상위 후보 8개의 회차별 분포 (빨강 = 붉은 노면, 파랑 = 블록) |

### 특징 추출 (`extract_features.py`)

| 파일 | 내용 |
|---|---|
| `imu_segments_features.csv` | imu 실험(회차 1·2·3) red_*·block_* 구간 12개 특징 |
| `imu_windows_2s_features.csv` | 같은 구간의 2초 창 103개 특징 |
| `imu_surface_summary.csv` | 회차·노면별 2초 창 중앙값 |
| `frames_2s_features.csv` | frames 기록 2초 창 2,351개. 진동 특징 약 100개, 카메라 예측, GPS 위치·속도, 블루투스 끊김 |
| `frames_surface_summary.csv` | frames 라벨별 특징 분포 |
| `observations_features.geojson` | frames 2초 창 점 2,351개. 진동 요약, 카메라 예측, 속도, 끊김, 수직 RMS, 첨도, 25~40Hz 파워, 큰 충격 수 |
| `coast_trials_features.csv` | 관성 시험 46회 특징과 품질 표시. 측정 오류 4회는 `excluded_measurement_error`로 표시 |
| `coast_surface_summary.csv` | 측정 오류를 뺀 42회의 노면별 요약 |
| `coast_psd.csv` | 관성 시험별 주파수 스펙트럼 (측정 오류 제외) |
| `coast_peak_candidates.csv` | 관성 시험 중 충격 후보 시각 |
| `recording_quality.csv` | 회차별 샘플 수, 끊김, 포화 등 수집 품질 |
| `feature_dictionary.json` | 특징 정의 |
| `analysis_config.json` | 필터, 창, 대역 설정과 라이브러리 버전 |
| `validation.json` | `validate_features.py` 검사 결과 |
| `plots/imu_run_*.png` | imu 실험 회차별 진동과 분석 구간 |
| `plots/coast_*.png`, `plots/frames_*.jpg` | 보고서 그림 (관성 시험 시간·주파수, 사진 표본) |

### 스크립트

| 파일 | 내용 |
|---|---|
| `extract_features.py` | 원본 recordings에서 특징 CSV, GeoJSON, 보고서, 그림을 만든다 |
| `feature_candidates.py` | imu 실험 2초 창으로 특징 후보를 고른다 |
| `block_profiles.py` | frames 2초 창으로 라벨별 블록 특성과 편안함 비용을 만든다 |
| `validate_features.py` | 계산 검사(합성 신호, 원본 패킷 단위, 대역 합, 중복)와 원본 해시 확인 |

## 분석 기준

- **imu 실험 구간:** `imu/*/marks.csv`의 표시 사이에서 앞뒤 1초를 뺀다. 회차 1은 `route_end` 표시가 없어 마지막 구간을 기록 끝까지 쓴다
- **진동:** 3축 가속도에 1~40Hz 4차 Butterworth 필터(전후방)를 건다. 샘플링 100Hz, 센서 대역폭 42Hz
- **끊김:** `imu.csv`의 `t_s`는 수신기가 붙인 값이라 `imu_ble.csv` 알림 간격으로 확인한다. 0.1초가 넘는 창은 후보·블록 특성에서 뺀다
- **수직·수평:** 장착 방향이 확인되지 않아 구간 평균 중력 방향으로 나눈다
- **coast:** 데이터 안내서의 측정 오류 시험(block 4번째, block4 3·4번째)은 요약·그림에서 뺀다. block4는 회차가 두 개라 두 회차의 trial 3을 모두 뺐다
- **블록 특성:** GPS 속도 0.3m/s 이상인 창만 쓴다. 대표값은 중앙값과 사분위다
- **편안함 비용:** `0.5 × 수직 RMS / 전체 중앙값 + 0.5 × (10m당 충격 + 1) / (전체 중앙값 + 1)`. 1이 전체 보통이고, 0.85 미만은 편함, 1.15 이상은 불편이다

## 아직 못 한 것

- **보행로 간선에 붙이기:** 보행로 graph가 없어 2초 창을 간선에 붙이지 못했다. 지금은 라벨(수집 폴더) 단위로만 정리했다. 라벨 하나가 여러 길에 걸친 경우가 있어(`block`은 약 250m 범위) graph가 생기면 간선 단위로 다시 집계한다
- **경로 엔진 반영:** 경로 비용에 진동 항을 넣으려면 graph 간선 형식에 필드를 추가해야 해서 지도 저장소와 맞춘 뒤 진행한다

## 한계

- 2초 창은 같은 구간에서 연속으로 나와 서로 독립이 아니다. imu 실험 회차는 3개, frames 라벨은 대부분 주행 한 번이다
- 측정 속도가 1.07~1.18m/s로 좁다. 다른 속도의 예상 진동수(`1m당 반복 수 × 속도`)는 검증하지 않은 가정이다
- 주 진동수가 대부분 20~30Hz이고 1m당 17~28번이다. 블록 이음매 간격과 맞지 않아 휠체어 바퀴·프레임 떨림이 섞인 것으로 보인다. 블록 고유의 진동수가 아니라 "그 길에서 느끼는 떨림 대역"으로 읽는다
- 편안함 비용의 무게와 등급 경계는 정하지 않은 값이다. 사람이 느낀 불편과 비교하지 않았다
- imu 실험에는 GPS가 없어 구간별 속도가 같았는지 알 수 없다

## 다시 만들기

```bash
pip install numpy pandas scipy PyWavelets matplotlib pillow
SURFACE_RECORDINGS=/path/to/recordings python extract_features.py
python feature_candidates.py
python block_profiles.py
python validate_features.py   # extract_features.py 결과 검사 (원본 경로가 같아야 함)
```
