"""라벨 붙은 블록 길마다 IMU 진동 특성을 정리하고, 편한 길 안내에 쓸 임시 비용을 붙인다.

extract_features.py가 만든 frames_2s_features.csv를 읽는다. 라벨은 수집할 때 붙인 frames 폴더 이름
(block1, red_block2, ramp_block3 ...)이다. 카메라 모델 예측은 쓰지 않는다.

    쓰는 창      GPS로 이동 중(0.3m/s 이상)이고 블루투스 끊김(0.1초 초과)이 없는 2초 창
    대표값       중앙값과 사분위(25~75%). 튀는 창에 덜 흔들린다
    떨림         수직 진동 RMS (중력 방향 성분, 장착 방향과 무관)
    덜컹거림     10m당 큰 충격 수 = 2초 창의 10m/s² 초과 피크 수 / 이동 거리 × 10
    진동 모양    첨도 < 1이면 '연속 떨림', 2 이상이면 '충격형', 사이는 '혼합'
    진동수       주 진동수 사분위(Hz)와 1m당 반복 수. 다른 속도의 예상 대역은 '1m당 반복 수 × 속도'로
                 계산하지만, 측정 속도가 1.07~1.18m/s로 좁아 검증하지 않은 가정이다

임시 편안함 비용 (창마다, 낮을수록 편함)
    cost = 0.5 × (수직 RMS / 전체 중앙값) + 0.5 × ((10m당 충격 수 + 1) / (전체 중앙값 + 1))
    1이면 전체 창의 보통 수준. 두 항의 무게(0.5, 0.5)는 정하지 않은 값이라 COMFORT_WEIGHTS로 바꾼다.
    사람이 느낀 불편과 맞는지는 확인하지 않았다.

결과: block_profiles.csv, block_profiles.json, block_comfort.geojson, block_comfort_map.html
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
MIN_SPEED = .3
COMFORT_WEIGHTS = {'vibration': .5, 'shock': .5}
COMFORT_LEVELS = [(.85, '편함'), (1.15, '보통'), (np.inf, '불편')]   # 라벨 중앙값 비용 기준
SPEEDS_FOR_BAND = (.8, 1.1, 1.4)
FEW_WINDOWS = 30
BANDS = ['1_3', '3_8', '8_15', '15_25', '25_40']


def kind_of(label):
    if label.startswith('red_block'):
        return '붉은 블록'
    if label.startswith('ramp_block'):
        return '경사로 블록'
    return '블록'


def shape_of(kurtosis):
    if not np.isfinite(kurtosis):
        return '-'
    return '연속 떨림' if kurtosis < 1 else ('충격형' if kurtosis >= 2 else '혼합')


def level_of(cost):
    return next(name for limit, name in COMFORT_LEVELS if cost < limit)


def q(series, p):
    v = series.dropna()
    return float(v.quantile(p)) if len(v) else None


def windows():
    w = pd.read_csv(OUT / 'frames_2s_features.csv')
    w = w[(w.gps_speed_mps >= MIN_SPEED) & (~w.ble_gap_flag.astype(bool))].copy()
    w['distance_m'] = w.gps_speed_mps * (w.window_end_s - w.window_start_s)
    w['shocks_per_10m'] = w.peak_candidates_gt10 / w.distance_m * 10
    vib_ref, shock_ref = w.vertical_rms_mps2.median(), w.shocks_per_10m.median()
    w['comfort_cost'] = (COMFORT_WEIGHTS['vibration'] * w.vertical_rms_mps2 / vib_ref
                         + COMFORT_WEIGHTS['shock'] * (w.shocks_per_10m + 1) / (shock_ref + 1))
    return w, {'vertical_rms_median_mps2': float(vib_ref), 'shocks_per_10m_median': float(shock_ref)}


def profile(label, g):
    k = float(g.z_excess_kurtosis.median())
    frac = {b: float(g[f'fraction_{b}hz'].median()) for b in BANDS}
    main_band = max(frac, key=frac.get)
    cpm_lo, cpm_mid, cpm_hi = q(g.dominant_cycles_per_m, .25), q(g.dominant_cycles_per_m, .5), q(g.dominant_cycles_per_m, .75)
    cost = float(g.comfort_cost.median())
    r = {'label': label, 'kind': kind_of(label), 'session': g.session.iloc[0], 'windows': len(g),
         'distance_m': round(float(g.distance_m.sum()), 1), 'few_windows': len(g) < FEW_WINDOWS,
         'speed_median_mps': q(g.gps_speed_mps, .5),
         'vertical_rms_median_mps2': q(g.vertical_rms_mps2, .5),
         'vertical_rms_q25_mps2': q(g.vertical_rms_mps2, .25), 'vertical_rms_q75_mps2': q(g.vertical_rms_mps2, .75),
         'rms_median_mps2': q(g.rms_mps2, .5),
         'shocks_per_10m_median': q(g.shocks_per_10m, .5), 'shocks_per_10m_q75': q(g.shocks_per_10m, .75),
         'z_kurtosis_median': k, 'vibration_shape': shape_of(k),
         'power_25_40hz_median_mps4': q(g.power_25_40hz_mps4, .5),
         'dominant_hz_q25': q(g.dominant_hz, .25), 'dominant_hz_median': q(g.dominant_hz, .5),
         'dominant_hz_q75': q(g.dominant_hz, .75), 'centroid_hz_median': q(g.centroid_hz, .5),
         'main_band_hz': main_band.replace('_', '-'), 'main_band_fraction': frac[main_band],
         **{f'fraction_{b}hz_median': v for b, v in frac.items()},
         'cycles_per_m_q25': cpm_lo, 'cycles_per_m_median': cpm_mid, 'cycles_per_m_q75': cpm_hi,
         'comfort_cost_median': cost, 'comfort_cost_q75': q(g.comfort_cost, .75), 'comfort_level': level_of(cost),
         'lat_min': float(g.latitude.min()), 'lat_max': float(g.latitude.max()),
         'lon_min': float(g.longitude.min()), 'lon_max': float(g.longitude.max())}
    for v in SPEEDS_FOR_BAND:   # 가정: 진동수는 속도에 비례
        r[f'expected_band_hz_at_{v}mps'] = None if cpm_lo is None else f'{cpm_lo * v:.0f}-{cpm_hi * v:.0f}'
    r['guidance'] = guidance(r)
    return r


def guidance(r):
    """안내 문구 초안. 거리는 실제 경로에서 간선 길이로 바꾼다."""
    shake = {'편함': '떨림이 적은', '보통': '보통 수준의', '불편': '떨림이 큰'}[r['comfort_level']]
    shape = {'연속 떨림': '계속 잘게 떨리는', '충격형': '이음매 충격이 있는', '혼합': '떨림과 충격이 섞인', '-': ''}[r['vibration_shape']]
    note = ' (기록이 짧아 참고용)' if r['few_windows'] else ''
    return f"{shake} {r['kind']} 길, {shape} 노면{note}"


def main():
    w, ref = windows()
    rows = [profile(label, g) for label, g in w.groupby('surface_id')]
    prof = pd.DataFrame(rows).sort_values('comfort_cost_median').reset_index(drop=True)
    prof.insert(1, 'comfort_rank', range(1, len(prof) + 1))
    prof.to_csv(OUT / 'block_profiles.csv', index=False, encoding='utf-8-sig')
    meta = {'reference': ref, 'comfort_weights': COMFORT_WEIGHTS, 'comfort_levels': [[l if np.isfinite(l) else None, n] for l, n in COMFORT_LEVELS],
            'min_speed_mps': MIN_SPEED, 'windows_used': len(w), 'few_windows_below': FEW_WINDOWS,
            'note': '임시 비용. 무게와 등급 기준은 정하지 않은 값이며 사람이 느낀 불편과 비교하지 않았다. '
                    '진동수 예상 대역은 진동수가 속도에 비례한다는 검증하지 않은 가정에 따른다.'}
    (OUT / 'block_profiles.json').write_text(json.dumps({'meta': meta, 'profiles': json.loads(prof.to_json(orient='records'))},
                                                        ensure_ascii=False, indent=2), encoding='utf-8')
    by_label = prof.set_index('label')
    features = []
    for _, r in w.dropna(subset=['latitude', 'longitude']).iterrows():
        p = by_label.loc[r.surface_id]
        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [r.longitude, r.latitude]},
                         'properties': {'label': r.surface_id, 'kind': p.kind, 'window_start_s': r.window_start_s,
                                        'utc': r.utc, 'speed_mps': round(r.gps_speed_mps, 2),
                                        'vertical_rms_mps2': round(r.vertical_rms_mps2, 3),
                                        'shocks_per_10m': round(r.shocks_per_10m, 2),
                                        'z_kurtosis': round(r.z_excess_kurtosis, 2), 'dominant_hz': r.dominant_hz,
                                        'comfort_cost': round(r.comfort_cost, 3),
                                        'label_comfort_level': p.comfort_level, 'label_guidance': p.guidance}})
    geo = {'type': 'FeatureCollection', 'properties': meta, 'features': features}
    (OUT / 'block_comfort.geojson').write_text(json.dumps(geo, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    write_map(geo, prof)
    show = prof[['comfort_rank', 'label', 'windows', 'vertical_rms_median_mps2', 'shocks_per_10m_median', 'vibration_shape',
                 'dominant_hz_q25', 'dominant_hz_q75', 'main_band_hz', 'comfort_cost_median', 'comfort_level']]
    print(show.round(2).to_string(index=False))


def write_map(geo, prof):
    table = prof[['comfort_rank', 'label', 'kind', 'windows', 'distance_m', 'vertical_rms_median_mps2', 'shocks_per_10m_median',
                  'vibration_shape', 'dominant_hz_q25', 'dominant_hz_q75', 'main_band_hz', 'comfort_cost_median',
                  'comfort_level', 'guidance']]
    rows = ''.join(
        '<tr>' + ''.join(f'<td>{v:.2f}</td>' if isinstance(v, float) else f'<td>{v}</td>' for v in r) + '</tr>'
        for r in table.itertuples(index=False))
    head = ''.join(f'<th>{h}</th>' for h in ['순위', '라벨', '종류', '창', '거리 m', '수직 RMS', '충격/10m', '진동 모양',
                                             '주Hz 25%', '주Hz 75%', '주 대역 Hz', '비용', '등급', '안내 문구'])
    data = json.dumps(geo, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>블록 길 진동과 편안함</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{font-family:-apple-system,sans-serif;margin:0;color:#1d3249}}#map{{height:62vh}}main{{padding:12px 18px}}
table{{border-collapse:collapse;font-size:12px}}td,th{{padding:5px 7px;border-bottom:1px solid #ddd}}th{{background:#edf3f9}}
.legend{{background:#fff;padding:6px 8px;font-size:12px;line-height:1.5}}.sw{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:4px}}</style>
<div id="map"></div><main><h2>라벨별 블록 길 특성 (편한 순)</h2>
<p>점 하나가 이동 중 2초 창입니다. 색은 임시 편안함 비용(1 = 전체 보통)입니다. 무게와 등급 기준은 정하지 않은 값이고, 사람이 느낀 불편과 비교하지 않았습니다.</p>
<div style="overflow:auto"><table><tr>{head}</tr>{rows}</table></div></main>
<script id="data" type="application/json">{data}</script>
<script>
const geo = JSON.parse(document.getElementById('data').textContent);
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 20, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap'}}).addTo(map);
const color = c => c < 0.85 ? '#1a9850' : c < 1.0 ? '#91cf60' : c < 1.15 ? '#fee08b' : c < 1.4 ? '#fc8d59' : '#d73027';
const layer = L.geoJSON(geo, {{pointToLayer: (f, ll) => L.circleMarker(ll, {{radius: 4, weight: 0, fillOpacity: .8, fillColor: color(f.properties.comfort_cost)}}),
  onEachFeature: (f, l) => {{ const p = f.properties; const box = document.createElement('div');
    box.innerText = `${{p.label}} (${{p.kind}})\\n${{p.label_guidance}}\\n비용 ${{p.comfort_cost}} · 수직 RMS ${{p.vertical_rms_mps2}} m/s² · 충격 ${{p.shocks_per_10m}}/10m\\n첨도 ${{p.z_kurtosis}} · 주 진동수 ${{p.dominant_hz}} Hz · 속도 ${{p.speed_mps}} m/s`;
    l.bindPopup(box); }} }}).addTo(map);
map.fitBounds(layer.getBounds());
const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{ const d = L.DomUtil.create('div', 'legend');
  d.innerHTML = [['< 0.85 편함', 0.8], ['0.85–1.0', 0.9], ['1.0–1.15', 1.1], ['1.15–1.4 불편', 1.3], ['≥ 1.4', 1.5]]
    .map(([t, c]) => `<span class="sw" style="background:${{color(c)}}"></span>${{t}}`).join('<br>'); return d; }};
legend.addTo(map);
</script></html>"""
    (OUT / 'block_comfort_map.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    main()
