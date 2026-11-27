# 휠체어 현장 실험 IMU 노면 특징 (2026-11-27)

2026-11-27 휠체어 현장 실험 기록에서 IMU 진동 특징을 뽑고, 붉은 노면과 블록을 가르는 특징 후보를 정리한 결과다.
원본 기록(recordings: frames, frames_v3_2s_wheel_ahead, imu, coast)은 용량이 커서 이 폴더에 넣지 않았다.

## 결과

| 파일 | 내용 |
|---|---|
| `feature_candidates.md` | 특징 후보 정리. 먼저 읽을 파일 |
| `feature_candidates.csv` | 특징별 회차 delta, 중앙값, 회차 안정성, 속도 상관, coast 비교, 중복 표시 전체 |
| `imu_segments_features.csv` | imu 실험(회차 1·2·3)의 red_*·block_* 구간 12개 특징 |
| `imu_windows_2s_features.csv` | 같은 구간을 2초로 나눈 창 103개 특징 |
| `imu_surface_summary.csv` | 회차·노면별 2초 창 중앙값 |
| `plots/candidates_top.png` | 상위 후보 8개의 회차별 분포 (빨강 = 붉은 노면, 파랑 = 블록) |
| `plots/imu_run_*.png` | 회차별 진동 크기와 분석에 쓴 구간 |

## 분석 기준

- 구간: `imu/*/marks.csv`의 표시 사이에서 앞뒤 1초를 뺀다. 회차 1은 `route_end` 표시가 없어 마지막 구간을 기록 끝까지 쓴다
- 진동: 3축 가속도에 1~40Hz 4차 Butterworth(전후방) 필터를 건 뒤 특징을 계산한다. 100Hz 설정, 센서 대역폭 42Hz
- 끊김: `imu.csv`의 `t_s`는 수신기가 붙인 값이라 `imu_ble.csv` 알림 간격으로 확인한다. 0.1초를 넘는 2초 창은 후보 평가에서 뺀다
- 수직·수평: 장착 방향이 확인되지 않아 구간 평균 중력 방향으로 나눈다(`vertical_*`, `horizontal_rms_mps2`)
- coast: 데이터 안내서의 측정 오류 시험(block 4번째, block4 3·4번째)을 뺀 42회만 참고 비교에 쓴다
- 후보 순위: 회차 1·2·3 모두 같은 방향으로 갈릴 때, 회차별 |Cliff's delta| 중 가장 작은 값

## 다시 만들기

```bash
pip install numpy pandas scipy PyWavelets matplotlib pillow
SURFACE_RECORDINGS=/path/to/recordings python extract_features.py   # 특징 CSV와 그림
python feature_candidates.py                                         # 후보 순위와 정리
```

`extract_features.py`는 이 폴더에 있는 파일 외에도 coast·frames 특징 CSV, GeoJSON, HTML 보고서를 함께 만든다.

## 한계

2초 창은 같은 20초 구간에서 연속으로 나와 서로 독립이 아니고, 회차가 3개뿐이다. imu 실험에는 GPS가 없어
구간별 속도가 같았는지 알 수 없다. 다른 장소의 같은 재질에서 같은 값이 나오는지는 아직 확인하지 않았다.
