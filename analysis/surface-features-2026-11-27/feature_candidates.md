# 노면 특징 후보 정리

## 데이터 범위

- 기준: imu 실험 회차 1, 2, 3의 2초 창 103개 (붉은 노면 54, 블록 49). 블루투스 간격 0.1초 초과 4개는 뺌
- 구간: marks.csv의 red_1→block_1→red_2→block_2 사이, 표시 앞뒤 1초 제외. 회차 1은 route_end 표시가 없어 마지막 구간을 기록 끝까지 씀
- coast: 측정 오류 시험(block 4번째, block4 3·4번째)을 뺀 42회. 시험 수가 적어 방향만 참고
- frames: GPS로 이동 중인 2초 창 2303개를 속도 상관 확인에만 씀. 폴더 이름과 모델 예측은 재질 정답이 아니어서 판별력 평가에는 쓰지 않음

## 읽는 법

- delta: Cliff's delta, 블록 창이 붉은 노면 창보다 클 확률 - 작을 확률. ±1에 가까울수록 잘 갈림, 0이면 못 가름
- 점수: 세 회차 모두 같은 방향일 때 회차별 |delta| 중 가장 작은 값. 회차가 바뀌어도 유지되는 차이만 남긴다
- 노면 차이/회차 흔들림: 1보다 크면 같은 노면의 회차 간 차이보다 노면 간 차이가 크다
- 속도 상관: frames 이동 창에서 GPS 속도와의 Spearman. |값| 0.5 이상이면 속도 보정 필요
- 중복: 위 순위의 특징과 |Spearman| 0.9 이상. 같은 정보라 하나만 쓰면 된다

## 후보 (중복 제외, 점수 순)

| 특징 | 묶음 | 점수 | delta 전체 | 회차별 delta | 붉은/블록 중앙값 | 노면 차이/회차 흔들림 | 속도 상관 | coast 블록-red | coast worst-good |
|---|---|---|---|---|---|---|---|---|---|
| `raw_acc_norm_median_mps2` | 크기 | 0.81 | -0.81 | -0.85 / -0.90 / -0.81 | 10.647 / 10.209 | 1.2 | 0.20 | -0.40 | 0.80 |
| `jerk_rms_mps3` | 크기 | 0.79 | -0.80 | -0.93 / -0.94 / -0.79 | 476.247 / 327.468 | 1.0 | 0.16 | -0.41 | 1.00 |
| `z_excess_kurtosis` | 분포 모양 | 0.72 | 0.75 | 0.79 / 0.79 / 0.72 | 0.329 / 1.783 | 2.0 | -0.06 | 0.90 | 0.00 |
| `power_25_40hz_mps4` | 크기 | 0.68 | -0.77 | -0.79 / -0.85 / -0.68 | 14.128 / 6.755 | 2.9 | 0.15 | -0.73 | 1.00 |
| `power_15_25hz_mps4` | 크기 | 0.66 | -0.67 | -0.76 / -0.66 / -0.73 | 10.652 / 4.252 | 0.6 | 0.12 | -0.38 | 1.00 |
| `peak_candidates_gt10` | 충격·이벤트 | 0.58 | -0.65 | -0.77 / -0.58 / -0.75 | 3.000 / 1.000 | 1.1 | 0.07 | -0.14 | 1.00 |
| `x_excess_kurtosis` | 분포 모양 | 0.54 | 0.72 | 0.54 / 0.90 / 0.68 | 0.170 / 1.884 | 2.1 | -0.10 | 0.91 | 0.33 |
| `crest_factor` | 분포 모양 | 0.49 | 0.55 | 0.49 / 0.65 / 0.57 | 2.612 / 3.150 | 3.0 | -0.05 | 0.81 | -0.07 |
| `z_ptp_mps2` | 크기 | 0.45 | -0.53 | -0.64 / -0.45 / -0.66 | 20.769 / 15.996 | 0.5 | 0.09 | -0.25 | 1.00 |
| `z_energy_fraction` | 주파수 분포 | 0.43 | -0.54 | -0.68 / -0.43 / -0.48 | 0.415 / 0.357 | 1.8 | 0.06 | -0.55 | -0.87 |
| `spectral_spread_hz` | 주파수 분포 | 0.40 | 0.62 | 0.40 / 0.74 / 0.71 | 7.959 / 8.728 | 1.7 | 0.03 | 0.84 | 1.00 |
| `power_1_3hz_mps4` | 크기 | 0.34 | -0.46 | -0.56 / -0.34 / -0.62 | 0.075 / 0.044 | 0.5 | 0.14 | -0.21 | 1.00 |
| `power_8_15hz_mps4` | 크기 | 0.28 | -0.45 | -0.62 / -0.28 / -0.50 | 3.326 / 1.802 | 0.2 | 0.10 | -0.15 | 1.00 |
| `peak_candidates_gt5` | 충격·이벤트 | 0.27 | -0.39 | -0.60 / -0.32 / -0.27 | 5.000 / 5.000 | 0.0 | 0.16 | -0.03 | 0.93 |
| `gyro_rms_rad_s` | 회전·자세 | 0.25 | -0.43 | -0.25 / -0.45 / -0.71 | 0.250 / 0.165 | 1.1 | 0.13 | -0.35 | 1.00 |
| `peak_mps2` | 크기 | 0.24 | -0.38 | -0.33 / -0.24 / -0.63 | 14.270 / 12.097 | 0.3 | 0.07 | -0.17 | 1.00 |
| `autocorr_peak_lag_s` | 분포 모양 | 0.16 | -0.21 | -0.35 / -0.16 / -0.18 | 0.240 / 0.170 | 0.7 | -0.09 | 0.45 | 0.60 |
| `y_excess_kurtosis` | 분포 모양 | 0.13 | 0.25 | 0.13 / 0.16 / 0.39 | 0.186 / 0.395 | 0.8 | -0.07 | 0.57 | 0.47 |
| `wavelet_fraction_a4_0_3p125` | 주파수 분포 | 0.12 | 0.36 | 0.50 / 0.56 / 0.12 | 0.005 / 0.007 | 0.5 | 0.06 | 0.43 | 0.47 |
| `y_ptp_mps2` | 크기 | 0.12 | -0.54 | -0.12 / -0.78 / -0.62 | 14.714 / 11.280 | 0.6 | 0.13 | -0.32 | 1.00 |

## 묶음별 대표

- **크기:** `raw_acc_norm_median_mps2` (0.81), `jerk_rms_mps3` (0.79), `power_25_40hz_mps4` (0.68)
- **분포 모양:** `z_excess_kurtosis` (0.72), `x_excess_kurtosis` (0.54), `crest_factor` (0.49)
- **충격·이벤트:** `peak_candidates_gt10` (0.58), `peak_candidates_gt5` (0.27)
- **주파수 분포:** `z_energy_fraction` (0.43), `spectral_spread_hz` (0.40), `wavelet_fraction_a4_0_3p125` (0.12)
- **회전·자세:** `gyro_rms_rad_s` (0.25), `pitch_p95_p5_deg` (0.08)

## 회차마다 방향이 바뀌는 특징 (재질 특징으로 쓰기 어려움)

`x_energy_fraction`, `wavelet_fraction_d4_3p125_6p25`, `y_energy_fraction`, `power_3_8hz_mps4`, `x_skewness`, `wavelet_fraction_d1_25_50`, `peak_candidates_gt20`, `peak_rate_gt20_per_s`, `fraction_8_15hz`, `z_skewness`, `gyro_z_rms_rad_s`, `dominant_hz`, `vertical_dominant_hz`, `pitch_median_deg`, `fraction_25_40hz`, `roll_median_deg`, `median_frequency_hz`, `vertical_centroid_hz`, `wavelet_fraction_d3_6p25_12p5`, `centroid_hz`, `gravity_tilt_from_sensor_z_deg`

## 속도에 크게 따라가는 특징

- 없음

frames 이동 창의 GPS 속도는 중앙값 1.13m/s, 사분위 1.06~1.19m/s로 범위가 좁다. 상관이 작다고 속도 영향이 없다는 뜻이 아니라, 이 데이터로는 판단할 수 없다는 뜻이다. imu 실험에는 GPS가 없어 붉은 노면과 블록 구간의 속도가 같았는지도 확인할 수 없다.

## 중복 묶음

- `fraction_25_40hz` ≈ `median_frequency_hz`, `vertical_centroid_hz`, `centroid_hz`
- `jerk_rms_mps3` ≈ `horizontal_rms_mps2`, `rms_mps2`, `acc_norm_std_mps2`, `z_rms_mps2`, `x_rms_mps2`, `vertical_rms_mps2`, `fourth_moment_root_mps2`, `unweighted_dose_proxy_mps175`, `acc_norm_p95_p5_mps2`, `p95_mps2`, `psd_total_mps4`, `vertical_p95_abs_mps2`, `y_rms_mps2`
- `peak_candidates_gt10` ≈ `peak_rate_gt10_per_s`
- `peak_candidates_gt20` ≈ `peak_rate_gt20_per_s`
- `peak_candidates_gt5` ≈ `peak_rate_gt5_per_s`
- `roll_median_deg` ≈ `gravity_tilt_from_sensor_z_deg`
- `spectral_flatness_0_1` ≈ `spectral_entropy_0_1`
- `z_energy_fraction` ≈ `vertical_energy_share`
- `z_excess_kurtosis` ≈ `vertical_excess_kurtosis`

## 주의

- `raw_acc_norm_median_mps2`는 중력을 포함한 가속도 크기의 중앙값이다. 진동이 클수록 9.8에서 올라가므로 진동 크기 특징과 같은 정보다. 센서 보정 상태에 따라 기준값이 달라지므로 단독 특징보다는 진동 크기의 다른 표현으로 본다
- 2초 창은 같은 구간(약 20초)에서 연속으로 나와 서로 독립이 아니다. 회차가 3개뿐이라 일반화는 안 된다
- imu 실험의 붉은 노면·블록은 한 경로의 특정 구간이다. 다른 장소의 같은 재질에서 같은 값이 나오는지는 아직 모른다
- 진동 크기는 속도, 하중, 바퀴 공기압, 장착 상태에 따라 달라진다. 실험 조건을 바꾸면 다시 확인해야 한다
- 주파수 특징은 센서 대역폭(42Hz)과 샘플링(100Hz) 안에서만 의미가 있다
