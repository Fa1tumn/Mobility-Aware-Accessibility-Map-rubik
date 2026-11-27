"""노면 재질 판별에 쓸 만한 IMU 특징 후보를 고른다. extract_features.py를 먼저 돌린다.

기준 데이터는 imu 실험(현장에서 red_*·block_*를 표시한 주행)의 2초 창이다. 재질 이름이 붙은 유일한
주행 기록이라 이것으로 '붉은 노면과 블록을 가르는가'를 본다. 나머지 기록은 보조로 쓴다.

    imu 2초 창      회차(1·2·3)마다 붉은 노면 vs 블록의 Cliff's delta. 세 회차 모두 같은 방향인지
    회차 안정성      같은 노면의 회차 간 중앙값 차이가 노면 간 차이보다 작은지
    frames 2초 창   GPS 속도와의 상관. 속도에 크게 따라가는 특징은 속도 보정이 필요
    coast 시험      red_worst vs red_good, 블록 계열 vs red 계열 (측정 오류 시험 제외, 시험 수가 적어 참고만)
    중복            imu 2초 창에서 순위가 높은 특징과 |Spearman| 0.9 이상이면 중복으로 표시

결과: feature_candidates.csv(전체 표), feature_candidates.md(정리), plots/candidates_top.png
판별력이 확인됐다는 뜻은 아니다. 2초 창들은 같은 구간에서 나와 서로 독립이 아니므로 p값은 참고만 한다.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
NOT_FEATURES = {'run', 'window', 'window_start_s', 'window_end_s', 'samples', 'ble_max_gap_s', 'order', 'mark_s', 'end_s',
                'feature_start_s', 'feature_end_s', 'trial', 'release_s', 'stop_s', 'gps_speed_fixes', 'latitude', 'longitude',
                'accuracy_m', 'gps_time_offset_s', 'frame_t_s', 'material_confidence', 'road_sharpness', 'label_changed',
                'gps_speed_mps', 'feature_duration_s', 'coast_duration_s', 'inverse_coast_time_per_s'}
REDUNDANT_RHO = .9
SPEED_RHO = .5

GROUPS = [  # (이름, 이름에 들어가는 문자열) 앞에서부터 먼저 맞는 묶음
    ('충격·이벤트', ('peak_candidates', 'peak_rate')),
    ('공간 주파수', ('cycles_per_m',)),
    ('회전·자세', ('gyro', 'pitch_', 'roll_', 'gravity_tilt')),
    ('분포 모양', ('kurtosis', 'skew', 'crest', 'autocorr')),
    ('주파수 분포', ('_hz', 'fraction', 'wavelet', 'entropy', 'flatness', 'coherence', 'energy_share')),
    ('크기', ('rms', 'p95', 'peak', 'ptp', 'fourth', 'dose', 'jerk', 'power', 'psd_total', 'std', 'median')),
]


def group_of(name):
    for label, keys in GROUPS:
        if any(k in name for k in keys):
            return label
    return '기타'


def cliffs_delta(x, y):
    """P(x>y) - P(x<y). x가 블록, y가 붉은 노면이면 양수 = 블록 쪽이 크다."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return None
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def numeric_features(df):
    cols = [c for c in df.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in cols if df[c].notna().sum() >= 10 and df[c].nunique() > 2]


def main():
    imu = pd.read_csv(OUT / 'imu_windows_2s_features.csv')
    frames = pd.read_csv(OUT / 'frames_2s_features.csv')
    coast = pd.read_csv(OUT / 'coast_trials_features.csv')
    coast = coast[~coast.excluded_measurement_error.astype(bool)]
    imu = imu[imu.surface.isin(['red', 'block'])]
    flagged = imu.ble_max_gap_s > .1
    imu_clean = imu[~flagged]
    runs = sorted(imu_clean.run.unique())

    rows = []
    for f in numeric_features(imu_clean):
        r = {'feature': f, 'group': group_of(f)}
        deltas = []
        for run in runs:
            d = imu_clean[imu_clean.run == run]
            delta = cliffs_delta(d[d.surface == 'block'][f], d[d.surface == 'red'][f])
            r[f'delta_run{run}'] = delta
            r[f'red_median_run{run}'] = d[d.surface == 'red'][f].median()
            r[f'block_median_run{run}'] = d[d.surface == 'block'][f].median()
            if delta is not None:
                deltas.append(delta)
        red, block = imu_clean[imu_clean.surface == 'red'][f].dropna(), imu_clean[imu_clean.surface == 'block'][f].dropna()
        r['delta_all'] = cliffs_delta(block, red)
        r['mannwhitney_p'] = float(stats.mannwhitneyu(block, red).pvalue) if len(red) > 2 and len(block) > 2 else None
        r['red_median'], r['block_median'] = red.median(), block.median()
        r['runs_same_sign'] = int(len(deltas) == len(runs) and (all(x > 0 for x in deltas) or all(x < 0 for x in deltas)))
        r['min_abs_delta'] = min(abs(x) for x in deltas) if deltas else None
        # 같은 노면의 회차 간 흔들림 대비 노면 간 차이. 1보다 크면 회차가 바뀌어도 노면 차이가 남는다.
        gap = abs(r['block_median'] - r['red_median'])
        spread = np.mean([np.ptp([r[f'{s}_median_run{run}'] for run in runs]) for s in ('red', 'block')])
        r['surface_gap_over_run_spread'] = float(gap / spread) if spread > 0 else None
        if f in frames:
            m = frames[(frames.motion_status == 'gps_moving') & frames[f].notna()]
            if len(m) > 30:
                r['speed_spearman_frames'] = float(stats.spearmanr(m[f], m.gps_speed_mps).statistic)
        if f in coast and coast[f].notna().sum() > 6:
            red_like = coast.surface_id.str.startswith('red')
            r['coast_delta_block_vs_red'] = cliffs_delta(coast[~red_like][f], coast[red_like][f])
            r['coast_delta_worst_vs_good'] = cliffs_delta(coast[coast.surface_id == 'red_worst'][f],
                                                          coast[coast.surface_id == 'red_good'][f])
        rows.append(r)
    table = pd.DataFrame(rows)
    table['score'] = table.min_abs_delta.fillna(0) * table.runs_same_sign
    table = table.sort_values(['score', 'delta_all'], key=lambda s: s.abs() if s.name == 'delta_all' else s,
                              ascending=False).reset_index(drop=True)

    # 순위가 높은 특징부터 남기고, 그와 거의 같은 정보를 담은 특징은 중복으로 표시
    rho = imu_clean[table.feature].rank().corr()
    kept, redundant = [], {}
    for f in table.feature:
        twin = next((k for k in kept if abs(rho.loc[f, k]) >= REDUNDANT_RHO), None)
        if twin is None:
            kept.append(f)
        else:
            redundant[f] = twin
    table['redundant_with'] = table.feature.map(redundant)
    table['speed_sensitive'] = table.speed_spearman_frames.abs() >= SPEED_RHO
    table.to_csv(OUT / 'feature_candidates.csv', index=False, encoding='utf-8-sig')

    plot_top(imu_clean, table[table.redundant_with.isna() & (table.score > 0)].head(8))
    write_markdown(table, imu, flagged, frames, coast, runs)
    print(table[['feature', 'group', 'score', 'delta_all', 'runs_same_sign', 'redundant_with']].head(25).to_string(index=False))


def plot_top(imu, top):
    if top.empty:
        return
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), constrained_layout=True)
    for ax, (_, r) in zip(axs.ravel(), top.iterrows()):
        f = r.feature
        for i, run in enumerate(sorted(imu.run.unique())):
            for j, (s, color) in enumerate([('red', '#d6604d'), ('block', '#4393c3')]):
                v = imu[(imu.run == run) & (imu.surface == s)][f].dropna()
                x = np.full(len(v), i * 3 + j) + np.random.default_rng(0).uniform(-.25, .25, len(v))
                ax.scatter(x, v, s=10, color=color, alpha=.7)
                ax.plot([i * 3 + j - .35, i * 3 + j + .35], [v.median()] * 2, color='black', lw=2)
        ax.set_xticks([0.5, 3.5, 6.5], [f'run {r}' for r in sorted(imu.run.unique())])
        ax.set_title(f"{f}\nmin |delta| {r.min_abs_delta:.2f}", fontsize=9)
        ax.grid(axis='y', alpha=.2)
    for ax in axs.ravel()[len(top):]:
        ax.axis('off')
    fig.suptitle('imu experiment 2 s windows: red surface (red) vs block (blue), black = median')
    fig.savefig(OUT / 'plots' / 'candidates_top.png', dpi=140)
    plt.close(fig)


def fmt(v, d=2):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def write_markdown(table, imu, flagged, frames, coast, runs):
    good = table[(table.score > 0) & table.redundant_with.isna()]
    L = ['# 노면 특징 후보 정리', '',
         '## 데이터 범위', '',
         f'- 기준: imu 실험 회차 {", ".join(map(str, runs))}의 2초 창 {len(imu)}개 (붉은 노면 {int((imu.surface == "red").sum())},'
         f' 블록 {int((imu.surface == "block").sum())}). 블루투스 간격 0.1초 초과 {int(flagged.sum())}개는 뺌',
         '- 구간: marks.csv의 red_1→block_1→red_2→block_2 사이, 표시 앞뒤 1초 제외. 회차 1은 route_end 표시가 없어 마지막 구간을 기록 끝까지 씀',
         f'- coast: 측정 오류 시험(block 4번째, block4 3·4번째)을 뺀 {len(coast)}회. 시험 수가 적어 방향만 참고',
         f'- frames: GPS로 이동 중인 2초 창 {int((frames.motion_status == "gps_moving").sum())}개를 속도 상관 확인에만 씀. '
         '폴더 이름과 모델 예측은 재질 정답이 아니어서 판별력 평가에는 쓰지 않음',
         '', '## 읽는 법', '',
         "- delta: Cliff's delta, 블록 창이 붉은 노면 창보다 클 확률 - 작을 확률. ±1에 가까울수록 잘 갈림, 0이면 못 가름",
         '- 점수: 세 회차 모두 같은 방향일 때 회차별 |delta| 중 가장 작은 값. 회차가 바뀌어도 유지되는 차이만 남긴다',
         '- 노면 차이/회차 흔들림: 1보다 크면 같은 노면의 회차 간 차이보다 노면 간 차이가 크다',
         f'- 속도 상관: frames 이동 창에서 GPS 속도와의 Spearman. |값| {SPEED_RHO} 이상이면 속도 보정 필요',
         f'- 중복: 위 순위의 특징과 |Spearman| {REDUNDANT_RHO} 이상. 같은 정보라 하나만 쓰면 된다',
         '', '## 후보 (중복 제외, 점수 순)', '',
         '| 특징 | 묶음 | 점수 | delta 전체 | 회차별 delta | 붉은/블록 중앙값 | 노면 차이/회차 흔들림 | 속도 상관 | coast 블록-red | coast worst-good |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in good.head(20).iterrows():
        per = ' / '.join(fmt(r[f'delta_run{x}']) for x in runs)
        L.append(f"| `{r.feature}` | {r.group} | {fmt(r.score)} | {fmt(r.delta_all)} | {per} | {fmt(r.red_median, 3)} / {fmt(r.block_median, 3)} |"
                 f" {fmt(r.surface_gap_over_run_spread, 1)} | {fmt(r.get('speed_spearman_frames'))}{' ⚠' if r.speed_sensitive else ''} |"
                 f" {fmt(r.get('coast_delta_block_vs_red'))} | {fmt(r.get('coast_delta_worst_vs_good'))} |")
    L += ['', '## 묶음별 대표', '']
    for g, sub in good.groupby('group', sort=False):
        top = sub.head(3)
        L.append(f"- **{g}:** " + ', '.join(f"`{f}` ({fmt(s)})" for f, s in zip(top.feature, top.score)))
    weak = table[(table.runs_same_sign == 0)].feature.tolist()
    L += ['', '## 회차마다 방향이 바뀌는 특징 (재질 특징으로 쓰기 어려움)', '', ', '.join(f'`{f}`' for f in weak[:40]) or '-']
    speed = table[table.speed_sensitive].sort_values('speed_spearman_frames', key=abs, ascending=False)
    moving = frames[frames.motion_status == 'gps_moving'].gps_speed_mps
    q1, q2, q3 = moving.quantile([.25, .5, .75])
    L += ['', '## 속도에 크게 따라가는 특징', '',
          ', '.join(f"`{f}` ({fmt(v)})" for f, v in zip(speed.feature, speed.speed_spearman_frames)) or '- 없음',
          '', f'frames 이동 창의 GPS 속도는 중앙값 {q2:.2f}m/s, 사분위 {q1:.2f}~{q3:.2f}m/s로 범위가 좁다. '
          '상관이 작다고 속도 영향이 없다는 뜻이 아니라, 이 데이터로는 판단할 수 없다는 뜻이다. '
          'imu 실험에는 GPS가 없어 붉은 노면과 블록 구간의 속도가 같았는지도 확인할 수 없다.',
          '', '## 중복 묶음', '']
    for keep, sub in table[table.redundant_with.notna()].groupby('redundant_with'):
        L.append(f"- `{keep}` ≈ " + ', '.join(f'`{f}`' for f in sub.feature))
    L += ['', '## 주의', '',
          '- `raw_acc_norm_median_mps2`는 중력을 포함한 가속도 크기의 중앙값이다. 진동이 클수록 9.8에서 올라가므로 진동 크기 특징과 같은 정보다. '
          '센서 보정 상태에 따라 기준값이 달라지므로 단독 특징보다는 진동 크기의 다른 표현으로 본다',
          '- 2초 창은 같은 구간(약 20초)에서 연속으로 나와 서로 독립이 아니다. 회차가 3개뿐이라 일반화는 안 된다',
          '- imu 실험의 붉은 노면·블록은 한 경로의 특정 구간이다. 다른 장소의 같은 재질에서 같은 값이 나오는지는 아직 모른다',
          '- 진동 크기는 속도, 하중, 바퀴 공기압, 장착 상태에 따라 달라진다. 실험 조건을 바꾸면 다시 확인해야 한다',
          '- 주파수 특징은 센서 대역폭(42Hz)과 샘플링(100Hz) 안에서만 의미가 있다']
    (OUT / 'feature_candidates.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
