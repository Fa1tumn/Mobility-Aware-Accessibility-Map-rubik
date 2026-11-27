"""coast(타력 주행) 8종 노면의 굴림·진동 특성을 정리한다. extract_features.py를 먼저 돌린다.

대상: coast/block, block1, block2, block3, block4, block5, red_good, red_worst
제외: 데이터 안내서의 측정 오류 주행. 폴더명과 trial로 고른다
      block 4번째, block4 3번째, block4 4번째 (block4는 회차가 둘이라 두 회차의 3번째를 모두 뺀다)

시험마다 쓰는 값 (coast_trials_features.csv)
    주행 시간        coast_s = stop_s - release_s. 같은 속도로 밀었다면 길수록 잘 구른다
    감속도(기존)     정지 후 2초(stop+0.5~2.5초) 전방 가속도 평균 - 주행 중 평균. 수집 프로그램의 mean_decel_m_s2와 같다
    감속도(보정)     위 계산 전에 x축에 섞인 중력 성분(-g·sin(pitch))을 뺀 값
    초기 속도 추정   보정 감속도 × 주행 시간. 같은 속도로 밀었다면 시험끼리 비슷해야 한다(감속도 검증용)
    진동            주행 중(앞뒤 0.25초 제외) 수직 진동 RMS, 첨도, 큰 충격 비율, 주 진동수, 대역 비율

노면별 대표값은 중앙값과 사분위다. 임시 비용(낮을수록 좋음)은 두 가지를 나눠 둔다.
    떨림 비용   수직 RMS / 8종 중앙값
    굴림 비용   (1 / 주행 시간) / 8종 중앙값. 같은 초기 속도라면 평균 감속도에 비례한다
    합계        0.5 × 떨림 + 0.5 × 굴림. 무게는 정하지 않은 값이다

material 번호: 노면 8종을 1~8로 부른다(팀 지정 순서).
    block=1, block1=2, red_worst=3, block2=4, block3=5, block4=6, red_good=7, block5=8
    material_code = '<번호>_<합계 순위>'. 예: red_worst는 material 3, 합계 순위 8이면 3_8
    표와 파일은 material 번호 순으로 정렬한다.

결과: coast_profiles.csv, coast_profiles.json, coast_profiles.md, coast_trials_by_material.csv, plots/coast_profiles.png
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
MATERIAL_ID = {'block': 1, 'block1': 2, 'red_worst': 3, 'block2': 4, 'block3': 5, 'block4': 6, 'red_good': 7, 'block5': 8}
SURFACES = sorted(MATERIAL_ID, key=MATERIAL_ID.get)
WEIGHTS = {'vibration': .5, 'rolling': .5}
BANDS = ['1_3', '3_8', '8_15', '15_25', '25_40']


def q(s, p):
    s = s.dropna()
    return float(s.quantile(p)) if len(s) else None


def load():
    c = pd.read_csv(OUT / 'coast_trials_features.csv')
    c = c[c.surface_id.isin(SURFACES)]
    excluded = c[c.excluded_measurement_error.astype(bool)]
    return c[~c.excluded_measurement_error.astype(bool)].copy(), excluded


def profile(name, g):
    frac = {b: q(g[f'fraction_{b}hz'], .5) for b in BANDS}
    k = q(g.z_excess_kurtosis, .5)
    flags = g.screen_flags.fillna('')
    return {
        'surface': name, 'trials': len(g), 'sessions': g.session.nunique(),
        'coast_s_median': q(g.coast_duration_s, .5), 'coast_s_q25': q(g.coast_duration_s, .25),
        'coast_s_q75': q(g.coast_duration_s, .75), 'coast_s_min': float(g.coast_duration_s.min()),
        'coast_s_max': float(g.coast_duration_s.max()),
        'inverse_coast_per_s_median': q(1 / g.coast_duration_s, .5),
        'decel_existing_median_mps2': q(g.ax_decel_proxy_mps2, .5),
        'decel_existing_nonpositive': int((g.ax_decel_proxy_mps2 <= 0).sum()),
        'decel_corrected_median_mps2': q(g.ax_decel_pitch_corrected_mps2, .5),
        'decel_corrected_q25_mps2': q(g.ax_decel_pitch_corrected_mps2, .25),
        'decel_corrected_q75_mps2': q(g.ax_decel_pitch_corrected_mps2, .75),
        'decel_corrected_nonpositive': int((g.ax_decel_pitch_corrected_mps2 <= 0).sum()),
        'initial_speed_estimate_median_mps': q(g.initial_speed_estimate_mps, .5),
        'pitch_change_median_deg': q(g.pitch_coast_minus_rest_deg, .5),
        'vertical_rms_median_mps2': q(g.vertical_rms_mps2, .5), 'vertical_rms_q25_mps2': q(g.vertical_rms_mps2, .25),
        'vertical_rms_q75_mps2': q(g.vertical_rms_mps2, .75), 'rms_median_mps2': q(g.rms_mps2, .5),
        'z_kurtosis_median': k,
        'shock_rate_gt10_per_s_median': q(g.peak_rate_gt10_per_s, .5),
        'dominant_hz_median': q(g.dominant_hz, .5), 'dominant_hz_q25': q(g.dominant_hz, .25),
        'dominant_hz_q75': q(g.dominant_hz, .75), 'centroid_hz_median': q(g.centroid_hz, .5),
        'main_band_hz': max(frac, key=lambda b: frac[b] or 0).replace('_', '-'),
        **{f'fraction_{b}hz_median': v for b, v in frac.items()},
        'rms_last_first_ratio_median': q(g.rms_last_first_ratio, .5),
        'trials_turning_flag': int(flags.str.contains('turning').sum()),
        'trials_rest_motion_flag': int(flags.str.contains('rest_motion').sum()),
        'trials_ble_gap_flag': int(flags.str.contains('ble_gap').sum()),
    }


def main():
    c, excluded = load()
    prof = pd.DataFrame([profile(s, c[c.surface_id == s]) for s in SURFACES if (c.surface_id == s).any()])
    vib_ref = prof.vertical_rms_median_mps2.median()
    roll_ref = prof.inverse_coast_per_s_median.median()
    prof['vibration_cost'] = prof.vertical_rms_median_mps2 / vib_ref
    prof['rolling_cost'] = prof.inverse_coast_per_s_median / roll_ref
    prof['total_cost'] = WEIGHTS['vibration'] * prof.vibration_cost + WEIGHTS['rolling'] * prof.rolling_cost
    for col in ('vibration_cost', 'rolling_cost', 'total_cost'):
        prof[col.replace('_cost', '_rank')] = prof[col].rank(method='min').astype(int)
    prof.insert(0, 'material_id', prof.surface.map(MATERIAL_ID))
    prof.insert(1, 'material_code', prof.material_id.astype(str) + '_' + prof.total_rank.astype(str))
    prof = prof.sort_values('material_id').reset_index(drop=True)
    # 주행별 표에도 같은 번호를 붙여 material 순으로 정렬해 둔다
    code = prof.set_index('surface').material_code
    trials = c.assign(material_id=c.surface_id.map(MATERIAL_ID), material_code=c.surface_id.map(code))
    front = ['material_id', 'material_code', 'surface_id', 'session', 'trial']
    trials = trials[front + [x for x in trials.columns if x not in front]].sort_values(['material_id', 'session', 'trial'])
    trials.to_csv(OUT / 'coast_trials_by_material.csv', index=False, encoding='utf-8-sig')

    # 감속도 검증: 같은 속도로 밀었다면 감속도는 1/주행 시간에 비례해야 한다
    check = {}
    for col in ('ax_decel_proxy_mps2', 'ax_decel_pitch_corrected_mps2'):
        ok = c.dropna(subset=[col])
        v0 = ok[col] * ok.coast_duration_s
        check[col] = {'trials': len(ok), 'nonpositive': int((ok[col] <= 0).sum()),
                      'spearman_with_inverse_coast': float(stats.spearmanr(ok[col], 1 / ok.coast_duration_s).statistic),
                      'initial_speed_median_mps': float(v0.median()),
                      'initial_speed_q25_mps': float(v0.quantile(.25)), 'initial_speed_q75_mps': float(v0.quantile(.75))}
    meta = {'surfaces': SURFACES, 'material_id': MATERIAL_ID,
            'material_code_rule': '<material 번호>_<합계 순위>. 합계 = 0.5 × 떨림 비용 + 0.5 × 굴림 비용, 낮을수록 1위',
            'weights': WEIGHTS, 'vibration_reference_mps2': float(vib_ref),
            'rolling_reference_per_s': float(roll_ref), 'trials_used': len(c),
            'excluded': [{'session': r.session, 'trial': int(r.trial), 'coast_s': float(r.coast_duration_s)}
                         for r in excluded.itertuples()],
            'decel_check': check}
    prof.to_csv(OUT / 'coast_profiles.csv', index=False, encoding='utf-8-sig')
    (OUT / 'coast_profiles.json').write_text(json.dumps({'meta': meta, 'profiles': json.loads(prof.to_json(orient='records'))},
                                                        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    plot(c, prof)
    write_md(prof, meta, c)
    print(prof[['material_id', 'material_code', 'surface', 'trials', 'coast_s_median', 'decel_existing_median_mps2', 'decel_corrected_median_mps2',
                'initial_speed_estimate_median_mps', 'vertical_rms_median_mps2', 'z_kurtosis_median', 'dominant_hz_median',
                'vibration_rank', 'rolling_rank', 'total_rank']].round(2).to_string(index=False))
    print(json.dumps(check, indent=1))


def plot(c, prof):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {s: ('#d6604d' if s.startswith('red') else '#4393c3') for s in SURFACES}
    for s in prof.surface:
        g = c[c.surface_id == s]
        axs[0].scatter(g.coast_duration_s, g.vertical_rms_mps2, color=colors[s], s=22, alpha=.7)
        p = prof[prof.surface == s].iloc[0]
        axs[0].scatter(p.coast_s_median, p.vertical_rms_median_mps2, color=colors[s], s=90, edgecolor='black')
        axs[0].annotate(f'{p.material_code} {s}', (p.coast_s_median, p.vertical_rms_median_mps2), xytext=(5, 5), textcoords='offset points', fontsize=9)
    axs[0].set(xlabel='Coast time (s): longer = rolls more easily', ylabel='Vertical vibration RMS (m/s²)',
               title='Rolling vs vibration (small = trial, large = median)')
    for i, s in enumerate(prof.surface):
        g = c[c.surface_id == s]
        for j, (col, mark) in enumerate([('ax_decel_proxy_mps2', 'o'), ('ax_decel_pitch_corrected_mps2', 's')]):
            axs[1].scatter(np.full(len(g), i + (j - .5) * .3), g[col], marker=mark, s=18,
                           color=['#999', '#333'][j], label=['existing', 'pitch corrected'][j] if i == 0 else None)
    axs[1].axhline(0, color='black', lw=.6)
    axs[1].set_xticks(range(len(prof)), [f'{m} {s}' for m, s in zip(prof.material_code, prof.surface)], rotation=40, ha='right')
    axs[1].set(ylabel='Deceleration (m/s²)', title='Deceleration per trial: existing vs pitch corrected')
    axs[1].legend(fontsize=8)
    for s in prof.surface:
        g = c[c.surface_id == s]
        axs[2].scatter(g.coast_duration_s, g.initial_speed_estimate_mps, color=colors[s], s=20)
    axs[2].axhline(0, color='black', lw=.6)
    axs[2].set(xlabel='Coast time (s)', ylabel='Corrected decel × coast time (m/s)',
               title='Initial speed estimate (equal push = similar)')
    fig.savefig(OUT / 'plots' / 'coast_profiles.png', dpi=140)
    plt.close(fig)


def f(v, d=2):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def write_md(prof, meta, c):
    L = ['# coast 8종 노면 굴림·진동 특성', '',
         f"- 대상: {', '.join(meta['surfaces'])}",
         f"- 사용한 주행: {meta['trials_used']}회. 측정 오류로 뺀 주행: " +
         ', '.join(f"{e['session'].split('/')[1]} {e['session'].split('/')[2]} {e['trial']}번째({e['coast_s']:.2f}초)" for e in meta['excluded']),
         '- 대표값은 중앙값. 비용은 8종 중앙값을 1로 둔 상대값이고 낮을수록 좋다',
         '- material 번호: ' + ', '.join(f'{s}={i}' for s, i in meta['material_id'].items()) +
         '. material_code = <번호>_<합계 순위> (예: red_worst는 3_8)', '',
         '## material별 요약 (material 번호 순)', '',
         '| material_code | 노면 | 주행 | 주행 시간 s (사분위) | 감속도 기존 | 감속도 보정 | 초기 속도 추정 m/s | 수직 진동 RMS | 첨도 | 주 진동수 Hz (사분위) | 떨림 순위 | 굴림 순위 | 합계 순위 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in prof.itertuples():
        L.append(f"| {r.material_code} | {r.surface} | {r.trials} | {f(r.coast_s_median)} ({f(r.coast_s_q25)}–{f(r.coast_s_q75)}) |"
                 f" {f(r.decel_existing_median_mps2, 3)} ({r.decel_existing_nonpositive} 음수) |"
                 f" {f(r.decel_corrected_median_mps2, 3)} ({r.decel_corrected_nonpositive} 음수) | {f(r.initial_speed_estimate_median_mps)} |"
                 f" {f(r.vertical_rms_median_mps2)} | {f(r.z_kurtosis_median, 1)} | {f(r.dominant_hz_median, 1)} ({f(r.dominant_hz_q25, 1)}–{f(r.dominant_hz_q75, 1)}) |"
                 f" {r.vibration_rank} | {r.rolling_rank} | {r.total_rank} |")
    e, k = meta['decel_check']['ax_decel_proxy_mps2'], meta['decel_check']['ax_decel_pitch_corrected_mps2']
    L += ['', '## 감속도 검증', '',
          '같은 속도로 밀었다면 평균 감속도 × 주행 시간(= 손 뗄 때 속도)이 시험끼리 비슷해야 하고, 감속도는 1/주행 시간과 같이 움직여야 한다.', '',
          '| 계산 | 0 이하 | 1/주행 시간과 Spearman | 초기 속도 추정 중앙값 (사분위) m/s |', '|---|---|---|---|',
          f"| 기존 (수집 프로그램과 같음) | {e['nonpositive']}/{e['trials']} | {f(e['spearman_with_inverse_coast'])} |"
          f" {f(e['initial_speed_median_mps'])} ({f(e['initial_speed_q25_mps'])}–{f(e['initial_speed_q75_mps'])}) |",
          f"| pitch 중력 보정 | {k['nonpositive']}/{k['trials']} | {f(k['spearman_with_inverse_coast'])} |"
          f" {f(k['initial_speed_median_mps'])} ({f(k['initial_speed_q25_mps'])}–{f(k['initial_speed_q75_mps'])}) |",
          '', '보정하면 음수가 줄고 초기 속도 추정이 걷는 속도 근처로 모이지만, 1/주행 시간과의 상관은 여전히 거의 없다. '
          '가속도계로 구한 감속도는 아직 노면 비교에 쓸 만큼 안정적이지 않다. 노면 비교에는 주행 시간을 쓰고, 감속도는 참고로 둔다.',
          '초기 속도 추정이 block·block1·red_worst는 0.6~0.8m/s, 나머지는 1.4~3.0m/s로 회차에 따라 갈린다. 세 노면은 주행 중 pitch가 정지 때보다 '
          '작아지고(음수) 나머지는 커진다. 회차마다 센서 장착 기울기가 달랐을 가능성이 있어, 장착 0점을 회차마다 잡거나 쿼터니언(imu_registers.csv 0x51)으로 '
          '중력을 빼는 방법을 다음에 확인한다. 2~3m/s는 휠체어를 미는 속도로는 빠르므로 보정 감속도도 아직 크게 나온다.',
          '', '## 주의', '',
          '- 첨도는 8종 모두 2 이상이다. 붉은 노면(2.7~2.9)이 블록(5.9~25)보다 낮아 상대적으로 덜 충격형이다. '
          '4초 안팎의 짧은 주행이라 frames·imu 실험의 2초 창 기준(1 미만 = 연속 떨림)을 그대로 쓰지 않았다',
          '- 초기 속도와 하중은 같게 했다고 기록됐지만 측정하지 않았다. 주행 시간 비교는 이 조건에 기댄다',
          '- 주행 중 속도가 손 뗄 때 속도에서 0까지 줄어들므로 주 진동수는 여러 속도가 섞인 값이다',
          '- red_good, red_worst는 사람이 붙인 이름이다. 순위는 이 기록의 값으로만 매겼다',
          f"- 회전 표시(각속도 z RMS 0.1rad/s 초과)가 있는 주행 {int(prof.trials_turning_flag.sum())}회, "
          f"정지 구간 움직임 표시 {int(prof.trials_rest_motion_flag.sum())}회는 행을 남겨 두었다",
          '- 떨림·굴림 비용의 무게(0.5, 0.5)는 정하지 않은 값이다']
    (OUT / 'coast_profiles.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
