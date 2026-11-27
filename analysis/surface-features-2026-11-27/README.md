# 휠체어 현장 실험 IMU 노면 특징 (2026-11-27)

2026-11-27 휠체어 현장 실험 기록으로 IMU 진동 특징을 뽑았다. **주 분석 대상은 coast(타력 주행) 8종 노면**
(block, block1, block2, block3, block4, block5, red_good, red_worst)이고, 노면마다 주행 시간·감속도·진동을 정리했다.
보조로 붉은 노면과 블록을 가르는 특징 후보(imu 실험)와 frames 라벨별 블록 길 특성을 정리했다.
원본 기록(recordings: frames, frames_v3_2s_wheel_ahead, imu, coast)은 용량이 커서 넣지 않았다.

## 올린 파일 정리

| 구분 | 파일 | 내용 |
|---|---|---|
| **먼저 볼 것** | `coast_profiles.md` | **주 분석.** coast 8종 노면(block~block5, red_good, red_worst)의 주행 시간, 감속도(기존·pitch 보정), 수직 진동, 첨도, 주 진동수, 떨림·굴림·합계 순위, 감속도 검증 |
| | `index.html` | 전체 분석 보고서 (coast 관성 시험, imu 실험, 사진 표본). 브라우저로 연다 |
| | `feature_candidates.md` | 붉은 노면과 블록을 가르는 특징 후보 (imu 실험 기준) |
| | `README.md` | 이 폴더 설명 |
| **coast 8종** | `coast_profiles.csv` | 노면 8종별 주행·회차 수, 주행 시간(중앙값·사분위·최소·최대), 감속도 기존·보정(중앙값·사분위·음수 수), 손 뗄 때 속도 추정, pitch 변화, 수직 진동, 첨도, 주 진동수, 대역 비율, 품질 표시 수, 비용과 순위 |
| | `coast_profiles.json` | 같은 내용과 기준값, 무게, 제외한 주행 4회, 감속도 검증 결과 |
| | `coast_trials_features.csv` | 주행 46회의 특징과 품질 표시. 측정 오류 4회는 `excluded_measurement_error`로 표시(행은 남김). pitch 보정 감속도와 손 뗄 때 속도 추정 포함 |
| | `coast_surface_summary.csv` | 측정 오류를 뺀 42회의 노면 8종 요약(주행 시간, 진동 RMS, 중심 주파수, 감속도 대리값) |
| | `coast_psd.csv` | 측정 오류를 뺀 42회의 주행별 주파수 스펙트럼 |
| | `coast_peak_candidates.csv` | 주행 중 충격 후보 시각 453건(주행 44회). 측정 오류 주행의 37건은 `excluded_measurement_error`로 표시 |
| | `plots/coast_profiles.png` | 굴림(주행 시간) 대 떨림(수직 RMS), 주행별 감속도 기존·보정, 손 뗄 때 속도 추정 |
| | `plots/coast_comparison.png`, `plots/coast_spectra.png` | 노면별 주행 시간·진동·감속도 비교, 노면별 스펙트럼 |
| | `plots/coast_<노면>_<회차>.png` | 회차별 진동과 주행 구간, 시간-주파수 그림 12개 |
| **imu 실험** | `imu_segments_features.csv` | 회차 1·2·3의 red_*·block_* 구간 12개 특징 |
| | `imu_windows_2s_features.csv` | 같은 구간의 2초 창 103개 특징 |
| | `imu_surface_summary.csv` | 회차·노면별 2초 창 중앙값 (6행) |
| | `plots/imu_run_*.png` | 회차별 진동과 분석 구간 3개 |
| **특징 후보** | `feature_candidates.csv` | 특징 69개의 회차별 delta, 중앙값, 회차 안정성, 속도 상관, coast 비교, 중복 |
| | `plots/candidates_top.png` | 상위 후보 8개의 회차별 분포 |
| **frames (보조)** | `frames_2s_features.csv` | 2초 창 2,351개의 진동 특징, 카메라 예측, GPS 위치·속도, 블루투스 끊김 |
| | `frames_surface_summary.csv` | frames 라벨 20개의 주요 특징 중앙값·사분위 |
| | `observations_features.geojson` | 2초 창 점 2,351개. 진동 요약, 카메라 예측, 속도, 끊김, 수직 RMS, 첨도, 25~40Hz 파워, 큰 충격 수 |
| | `plots/frames_<라벨>.jpg` | 라벨별 사진 표본 20개 |
| **블록 길 (보조)** | `block_comfort_map.html` | frames 라벨 블록 길 지도(2초 창 점을 편안함 비용으로 색칠)와 라벨 표 |
| | `block_profiles.csv` | frames 라벨 20개의 진동 특성, 진동수, 편안함 순위·등급, 안내 문구 |
| | `block_profiles.json` | 같은 내용과 계산 기준(기준값, 무게, 등급 경계) |
| | `block_comfort.geojson` | 이동 중이고 끊김 없는 2초 창 점 2,219개와 편안함 비용 |
| **품질·설정** | `recording_quality.csv` | 회차 36개(frames 20, coast 13, imu 3)의 샘플 수, 끊김, 포화 등 수집 품질 |
| | `feature_dictionary.json` | 특징 정의 35개 |
| | `analysis_config.json` | 필터, 창, 대역, 제외 규칙, 끊김 기준, 라이브러리 버전 (코드에서 생성) |
| | `validation.json` | `validate_features.py` 검사 결과 |
| **스크립트** | `extract_features.py` | 원본 recordings에서 특징 CSV, GeoJSON, 보고서, 그림, 설정 파일을 만든다 |
| | `coast_profiles.py` | coast 8종 노면의 굴림·진동 특성과 감속도 검증 |
| | `feature_candidates.py` | imu 실험 2초 창으로 특징 후보를 고른다 |
| | `block_profiles.py` | frames 라벨별 블록 특성과 편안함 비용 |
| | `validate_features.py` | 계산 검사(합성 신호, 원본 패킷 단위, 대역 합, 중복)와 원본 해시 확인 |

## 분석 기준

- **imu 실험 구간:** `imu/*/marks.csv`의 표시 사이에서 앞뒤 1초를 뺀다. 회차 1은 `route_end` 표시가 없어 마지막 구간을 기록 끝까지 쓴다
- **진동:** 3축 가속도에 1~40Hz 4차 Butterworth 필터(전후방)를 건다. 샘플링 100Hz, 센서 대역폭 42Hz
- **끊김:** `imu.csv`의 `t_s`는 수신기가 붙인 값이라 `imu_ble.csv` 알림 간격으로 확인한다. 0.1초가 넘는 창은 후보·블록 특성에서 뺀다
- **수직·수평:** 장착 방향이 확인되지 않아 구간 평균 중력 방향으로 나눈다
- **coast 제외:** 데이터 안내서의 측정 오류 주행(block 4번째, block4 3·4번째)은 요약·그림에서 뺀다. 안내서대로 폴더명과 trial로 골라, 회차가 두 개인 block4는 두 회차의 3번째를 모두 뺐다
- **coast 주행 시간:** `stop_s - release_s`. 같은 속도로 밀었다고 기록돼 노면 비교의 주 지표로 쓴다
- **coast 감속도:** 정지 후(stop+0.5~2.5초) 전방 가속도 평균에서 주행 중 평균을 뺀다(수집 프로그램과 같은 방법). 보정판은 x축에 섞인 중력 성분 `-g·sin(pitch)`를 먼저 뺀다
- **coast 비용:** 떨림 = 수직 RMS / 8종 중앙값, 굴림 = (1/주행 시간) / 8종 중앙값, 합계 = 0.5 × 떨림 + 0.5 × 굴림
- **블록 특성:** GPS 속도 0.3m/s 이상인 창만 쓴다. 대표값은 중앙값과 사분위다
- **편안함 비용:** `0.5 × 수직 RMS / 전체 중앙값 + 0.5 × (10m당 충격 + 1) / (전체 중앙값 + 1)`. 1이 전체 보통이고, 0.85 미만은 편함, 1.15 이상은 불편이다

## 아직 못 한 것

- **보행로 간선에 붙이기:** 보행로 graph가 없어 2초 창을 간선에 붙이지 못했다. 지금은 라벨(수집 폴더) 단위로만 정리했다. 라벨 하나가 여러 길에 걸친 경우가 있어(`block`은 약 250m 범위) graph가 생기면 간선 단위로 다시 집계한다
- **경로 엔진 반영:** 경로 비용에 진동 항을 넣으려면 graph 간선 형식에 필드를 추가해야 해서 지도 저장소와 맞춘 뒤 진행한다

## 한계

- coast 감속도는 보정해도 1/주행 시간과 상관이 거의 없고(Spearman -0.02), 초기 속도 추정이 회차에 따라 0.6~3.0m/s로 갈린다. 회차마다 장착 기울기가 달랐을 가능성이 있어 노면 비교에는 주행 시간을 쓰고 감속도는 참고로 둔다

- 2초 창은 같은 구간에서 연속으로 나와 서로 독립이 아니다. imu 실험 회차는 3개, frames 라벨은 대부분 주행 한 번이다
- 측정 속도가 1.07~1.18m/s로 좁다. 다른 속도의 예상 진동수(`1m당 반복 수 × 속도`)는 검증하지 않은 가정이다
- 주 진동수가 대부분 20~30Hz이고 1m당 17~28번이다. 블록 이음매 간격과 맞지 않아 휠체어 바퀴·프레임 떨림이 섞인 것으로 보인다. 블록 고유의 진동수가 아니라 "그 길에서 느끼는 떨림 대역"으로 읽는다
- 편안함 비용의 무게와 등급 경계는 정하지 않은 값이다. 사람이 느낀 불편과 비교하지 않았다
- imu 실험에는 GPS가 없어 구간별 속도가 같았는지 알 수 없다

## 다시 만들기

```bash
pip install numpy pandas scipy PyWavelets matplotlib pillow
SURFACE_RECORDINGS=/path/to/recordings python extract_features.py
python coast_profiles.py
python feature_candidates.py
python block_profiles.py
python validate_features.py   # extract_features.py 결과 검사 (원본 경로가 같아야 함)
```
