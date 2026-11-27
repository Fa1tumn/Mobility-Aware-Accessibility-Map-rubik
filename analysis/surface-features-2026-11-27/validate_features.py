"""Numerical and provenance checks without rerunning the extraction pipeline."""
from pathlib import Path
import ast, hashlib, json, struct, math
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
tree=ast.parse((P/'extract_features.py').read_text())
selected=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) or
          (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['FS','BANDS','SOS'] for t in n.targets))]
ns={};exec(compile(ast.Module(body=selected,type_ignores=[]),'feature_definitions','exec'),ns)
t=np.arange(2000)/100
x=np.column_stack([np.sin(2*np.pi*10*t),np.zeros(len(t)),np.full(len(t),9.80665)])
y=ns['signal'].sosfiltfilt(ns['SOS'],x,axis=0)[500:700]
f=ns['feat'](y,np.zeros_like(y),x[500:700])
assert abs(f['rms_mps2']-1/np.sqrt(2))<.01
assert f['dominant_hz']==10
assert abs(sum(v for k,v in f.items() if k.startswith('fraction_'))-1)<1e-12
assert abs(sum(v for k,v in f.items() if k.startswith('wavelet_fraction_'))-1)<1e-12
assert abs(ns['mean_interval'](t,t,1.234,3.456)-(1.234+3.456)/2)<1e-12
manifest=json.loads((P/'source_manifest.json').read_text())
assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest)
checked=0
for r in manifest:
    if Path(r['path']).name!='imu.csv':continue
    d=pd.read_csv(r['path'])
    if len(d)==0:continue
    # Deterministic across-session samples to verify acquisition units, including signed values.
    sample=d.iloc[np.linspace(0,len(d)-1,min(101,len(d))).astype(int)]
    v=np.array([struct.unpack('<9h',bytes.fromhex(s)[2:20]) for s in sample.raw_hex])
    assert np.allclose(sample[['ax','ay','az']],v[:,:3]*16*9.80665/32768)
    assert np.allclose(sample[['gx','gy','gz']],v[:,3:6]*math.radians(2000)/32768)
    checked+=len(sample)
w=pd.read_csv(P/'frames_2s_features.csv');c=pd.read_csv(P/'coast_trials_features.csv')
cols=[col for col in w if col.startswith('fraction_')]
assert np.allclose(w[cols].sum(axis=1),1)
wave=[col for col in w if col.startswith('wavelet_fraction_')]
assert np.allclose(w[wave].sum(axis=1),1)
assert not w.duplicated(['session','window_start_s']).any()
assert not c.duplicated(['session','trial']).any()
assert c.existing_decel_difference.abs().max()<1e-10
assert w.gps_time_offset_s.max()<=1.5
pred=w[w.frame_t_s.notna()]
assert ((pred.frame_t_s>=pred.window_start_s)&(pred.frame_t_s<pred.window_end_s)).all()
assert c.friction_force_N.isna().all() and c.rolling_resistance_coefficient.isna().all()
geo=json.loads((P/'observations_features.geojson').read_text())
assert len(geo['features'])==len(w.dropna(subset=['latitude','longitude']))
assert len(set(r['properties']['feature_id'] for r in geo['features']))==len(geo['features'])
report={'source_files_unchanged':len(manifest),'raw_packet_samples_checked':checked,
        'synthetic_10hz_peak':f['dominant_hz'],'synthetic_rms':f['rms_mps2'],
        'base_feature_count':len(f),'coast_trials':len(c),'frame_windows':len(w),
        'geojson_points':len(geo['features']),'band_and_wavelet_fractions_sum_to_one':True,
        'acquisition_decel_reproduced':True,'no_invented_force_or_friction':True}
(P/'validation.json').write_text(json.dumps(report,indent=2))
pd.DataFrame({'feature':list(f)}).to_csv(P/'base_feature_list.csv',index=False)
print(json.dumps(report,indent=2))
