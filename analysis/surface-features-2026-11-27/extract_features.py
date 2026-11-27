"""Reproducible, offline exploratory surface features; sources remain immutable."""
from pathlib import Path
from datetime import datetime, timedelta
import json, hashlib, html, os
import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

# 압축을 푼 recordings 폴더. 다른 위치면 SURFACE_RECORDINGS 환경변수로 바꾼다.
ROOT=Path(os.environ.get('SURFACE_RECORDINGS','/Users/jeonhyeonji/Downloads/recordings'))
OUT=Path(__file__).resolve().parent
(OUT/'plots').mkdir(exist_ok=True)
FS=100.
BANDS=[(1,3),(3,8),(8,15),(15,25),(25,40)]
SOS=signal.butter(4,[1,40],btype='bandpass',fs=FS,output='sos')
MANIFEST=[]; QUALITY=[]; TRIALS=[]; WINDOWS=[]; SPECTRA=[]; PEAKS=[]; MATCHES=[]; PHOTOS=[]
SEGMENTS=[]; SEG_WINDOWS=[]; BLE={}
# 데이터 안내서의 측정 오류 구간: (coast 폴더명, trial). 안내서대로 폴더명+trial로 고른다.
# block4는 회차가 두 개라 trial 3은 두 회차 모두 해당한다. 한 회차만이면 세 번째 칸에 회차 이름을 넣는다.
EXCLUDED_COAST={('block',4,None),('block4',3,None),('block4',4,None)}
BLE_GAP_FLAG_S=.1        # 블루투스 알림 간격이 이보다 길면 데이터가 끊긴 구간으로 표시
IMU_SEGMENT_TRIM_S=1.    # imu 실험: 표시 직후·다음 표시 직전 1초는 노면 경계로 보고 뺀다
SKIP_FOLDERS={'test'}
STOP_MARKS=('route_end',)

def track(p):
    MANIFEST.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

def read(p):
    track(p)
    return pd.read_csv(p)

def psd(a):
    f,p=signal.welch(a,fs=FS,window='hann',nperseg=min(200,len(a)),noverlap=min(200,len(a))//2,axis=0)
    return f,p.sum(axis=1)

def feat(a,g,raw):
    if len(a)<100:return {}
    norm=np.linalg.norm(a,axis=1); dur=len(a)/FS
    rms=float(np.sqrt(np.mean(norm**2)))
    f,p=psd(a);mask=(f>=1)&(f<=40); ff=f[mask]; pp=p[mask]; total=pp.sum();prob=pp/total if total>0 else np.zeros(len(pp))
    centroid=float((ff*prob).sum()); nz=prob>0
    r={'rms_mps2':rms,'peak_mps2':float(norm.max()),'p95_mps2':float(np.percentile(norm,95)),
       'crest_factor':float(norm.max()/rms) if rms else None,
       'fourth_moment_root_mps2':float(np.mean(norm**4)**.25),
       'unweighted_dose_proxy_mps175':float((np.sum(norm**4)/FS)**.25),
       'jerk_rms_mps3':float(np.sqrt(np.mean(np.sum(np.gradient(a,1/FS,axis=0)**2,axis=1)))),
       'gyro_rms_rad_s':float(np.sqrt(np.mean(np.sum(g*g,axis=1)))),
       'gyro_z_rms_rad_s':float(np.sqrt(np.mean(g[:,2]**2))),
       'dominant_hz':float(ff[np.argmax(pp)]),'centroid_hz':centroid,
       'spectral_spread_hz':float(np.sqrt(((ff-centroid)**2*prob).sum())),
       'spectral_entropy_0_1':float(-(prob[nz]*np.log(prob[nz])).sum()/np.log(len(prob))),
       'spectral_flatness_0_1':float(np.exp(np.mean(np.log(pp+1e-20)))/(np.mean(pp)+1e-20)),
       'median_frequency_hz':float(ff[min(np.searchsorted(np.cumsum(prob),.5),len(ff)-1)]),
       'rolloff95_hz':float(ff[min(np.searchsorted(np.cumsum(prob),.95),len(ff)-1)]),
       'psd_total_mps4':float(total*(f[1]-f[0])),
       'raw_acc_norm_median_mps2':float(np.median(np.linalg.norm(raw,axis=1))),
       'acc_norm_std_mps2':float(np.std(np.linalg.norm(raw,axis=1))),
       'acc_norm_p95_p5_mps2':float(np.ptp(np.percentile(np.linalg.norm(raw,axis=1),[5,95])))}
    for j,axis in enumerate('xyz'):
        x=a[:,j]; ar=float(np.sqrt(np.mean(x*x)))
        r.update({f'{axis}_rms_mps2':ar,f'{axis}_ptp_mps2':float(np.ptp(x)),
                  f'{axis}_skewness':float(stats.skew(x)),f'{axis}_excess_kurtosis':float(stats.kurtosis(x)),
                  f'{axis}_energy_fraction':ar*ar/(rms*rms) if rms else None})
    # 장착 방향이 확인되지 않았으므로 중력 방향(원시 가속도 평균)으로 수직·수평 성분을 나눈다.
    # 센서 축(x,y,z)에 의존하지 않는 특징이다. 기울어진 경사로에서는 중력 방향과 노면 법선이 다를 수 있다.
    gvec=raw.mean(axis=0);gn=np.linalg.norm(gvec)
    if gn>0:
        u=gvec/gn;v=a@u;h=a-np.outer(v,u);vr=float(np.sqrt(np.mean(v*v)));hr=float(np.sqrt(np.mean(np.sum(h*h,axis=1))))
        fv,pv=signal.welch(v,fs=FS,window='hann',nperseg=min(200,len(v)),noverlap=min(200,len(v))//2)
        mv=(fv>=1)&(fv<=40);pvv=pv[mv];tv=pvv.sum()
        r.update({'vertical_rms_mps2':vr,'horizontal_rms_mps2':hr,'vertical_energy_share':vr*vr/(rms*rms) if rms else None,
                  'vertical_dominant_hz':float(fv[mv][np.argmax(pvv)]),'vertical_centroid_hz':float((fv[mv]*pvv).sum()/tv) if tv else None,
                  'vertical_p95_abs_mps2':float(np.percentile(abs(v),95)),'vertical_excess_kurtosis':float(stats.kurtosis(v)),
                  'gravity_tilt_from_sensor_z_deg':float(np.degrees(np.arccos(np.clip(abs(u[2]),0,1))))})
    for lo,hi in BANDS:
        m=(f>=lo)&((f<hi) if hi<40 else (f<=hi))
        r[f'power_{lo}_{hi}hz_mps4']=float(p[m].sum()*(f[1]-f[0]))
        r[f'fraction_{lo}_{hi}hz']=float(p[m].sum()/total) if total else None
    # Autocorrelation is a periodicity descriptor, not a known wheel-joint spacing.
    ac=sum(signal.correlate(a[:,j]-a[:,j].mean(),a[:,j]-a[:,j].mean(),mode='full',method='fft')[len(a)-1:] for j in range(3))
    ac=ac/ac[0] if ac[0]>0 else ac
    hi=min(100,len(ac)-1);k=5+int(np.argmax(ac[5:hi+1]))
    r['autocorr_peak_lag_s']=k/FS;r['autocorr_peak_0_05_1s']=float(ac[k])
    # At least four seconds and multiple Welch averages, to avoid unity from a single segment.
    r['z_gyro_y_coherence_mean']=None
    if len(a)>=400 and np.std(g[:,1])>1e-8:
        cf,cp=signal.coherence(a[:,2],g[:,1],fs=FS,nperseg=100,noverlap=50)
        r['z_gyro_y_coherence_mean']=float(np.mean(cp[(cf>=1)&(cf<=40)]))
    # db4 four-level DWT, approximate bands; pad behavior is documented.
    for band in ['a4_0_3p125','d4_3p125_6p25','d3_6p25_12p5','d2_12p5_25','d1_25_50']:
        r[f'wavelet_fraction_{band}']=None
    if pywt.dwt_max_level(len(a),pywt.Wavelet('db4').dec_len)>=4:
        coeff=pywt.wavedec(a,'db4',mode='periodization',level=4,axis=0)
        energy=np.array([np.sum(c*c) for c in coeff]);energy=energy/energy.sum() if energy.sum() else energy
        for band,e in zip(['a4_0_3p125','d4_3p125_6p25','d3_6p25_12p5','d2_12p5_25','d1_25_50'],energy):r[f'wavelet_fraction_{band}']=float(e)
    return r

def mean_interval(t,x,lo,hi):
    if hi<=lo or lo<t[0] or hi>t[-1]:return None
    ii=(t>lo)&(t<hi);tt=np.r_[lo,t[ii],hi];xx=np.interp(tt,t,x)
    return float(np.trapezoid(xx,tt)/(hi-lo))

def load_session(folder,group):
    mp=folder/'session.json';track(mp);meta=json.loads(mp.read_text())
    d=read(folder/'imu.csv');ble=read(folder/'imu_ble.csv')
    if (folder/'imu_registers.csv').exists():track(folder/'imu_registers.csv')
    name=folder.parent.name;sid=f'{group}/{name}/{folder.name}'
    q={'session':sid,'group':group,'surface_id':name,'samples':len(d),'status':meta['status'],'timing_resets':meta.get('imu_timing_resets'),
       'discarded_ble_bytes':int(ble.discarded_bytes.sum()),'packet_count_matches':int(ble.motion_packets.sum())==len(d)}
    # imu.csv의 t_s는 수신기가 설정 간격(0.01초)으로 붙인 값이라 끊김이 보이지 않는다. 실제 끊김은 알림 수신 시각으로 본다.
    tb=ble.t_s.to_numpy();BLE[sid]=tb
    if len(tb)>1:q.update(ble_max_gap_s=float(np.diff(tb).max()),ble_gaps_gt100ms=int((np.diff(tb)>BLE_GAP_FLAG_S).sum()))
    if len(d)<200:
        q['usable']=False;QUALITY.append(q);return None
    t=d.t_s.to_numpy();dt=np.diff(t);a=d[['ax','ay','az']].to_numpy();g=d[['gx','gy','gz']].to_numpy()
    q.update(duration_s=float(t[-1]-t[0]),host_span_hz=float((len(d)-1)/(t[-1]-t[0])),max_dt_ms=float(dt.max()*1000),
             gaps_gt30ms=int((dt>.03).sum()),saturated_samples=int((abs(a)>=.99*16*9.80665).any(axis=1).sum()))
    q['usable']=bool(np.isfinite(a).all() and np.isfinite(g).all() and (dt>0).all() and meta.get('imu_timing_resets',0)==0 and dt.max()<.1)
    QUALITY.append(q)
    if not q['usable']:return None
    filtered=signal.sosfiltfilt(SOS,a,axis=0);norm=np.linalg.norm(filtered,axis=1)
    peaks={thr:signal.find_peaks(norm,height=thr,prominence=thr/2,distance=25)[0] for thr in [5,10,20]}
    return meta,d,t,a,g,filtered,peaks,sid

def ble_gap(sid,lo,hi):
    """[lo,hi) 안에서 가장 긴 블루투스 알림 간격(초). 구간 경계까지의 간격도 포함한다."""
    tb=BLE.get(sid)
    if tb is None or len(tb)==0:return None
    inside=tb[(tb>=lo)&(tb<hi)]
    return float(np.diff(np.r_[lo,inside,hi]).max())

def excluded_trial(name,session,trial):
    return any(n==name and t==trial and (s is None or s==session) for n,t,s in EXCLUDED_COAST)

def gps_speed(gps,lo,hi,max_accuracy=10.):
    """구간 안 GPS 점들의 이동 거리/시간(m/s). 정확도 10m 이하 점이 2개 이상, 1.5초 이상 걸쳐야 계산한다."""
    g=gps[(gps.t_s>=lo-.5)&(gps.t_s<=hi+.5)&(gps.accuracy_m<=max_accuracy)]
    if len(g)<2 or g.t_s.iloc[-1]-g.t_s.iloc[0]<1.5:return None,len(g)
    la=np.radians(g.latitude.to_numpy());lo_=np.radians(g.longitude.to_numpy())
    d=2*6371000*np.arcsin(np.sqrt(np.sin(np.diff(la)/2)**2+np.cos(la[:-1])*np.cos(la[1:])*np.sin(np.diff(lo_)/2)**2))
    return float(d.sum()/(g.t_s.iloc[-1]-g.t_s.iloc[0])),len(g)

def stamp(meta,t):return (datetime.fromisoformat(meta['start_utc'])+timedelta(seconds=float(t))).isoformat()

def extras(r,idx,peaks,t,lo,hi,d):
    for thr,ii in peaks.items():
        n=int(((t[ii]>=lo)&(t[ii]<hi)).sum());r[f'peak_candidates_gt{thr}']=n;r[f'peak_rate_gt{thr}_per_s']=n/(hi-lo)
    r['pitch_median_deg']=float(d.pitch_deg.iloc[idx].median())
    r['pitch_p95_p5_deg']=float(np.ptp(np.percentile(d.pitch_deg.iloc[idx],[5,95])))
    r['roll_median_deg']=float(d.roll_deg.iloc[idx].median())

for folder in sorted((ROOT/'coast').glob('*/*')):
    if not (folder/'session.json').exists() or folder.parent.name in SKIP_FOLDERS:continue
    trials=read(folder/'trials.csv');track(folder/'marks.csv')
    data=load_session(folder,'coast')
    if data is None:continue
    meta,d,t,a,g,bp,peaks,sid=data;name=folder.parent.name
    original=read(folder/'coast_summary.csv') if (folder/'coast_summary.csv').exists() else pd.DataFrame()
    for j,row in trials.iterrows():
        release=float(row.release_s);stop=float(row.stop_s);dur=stop-release
        lo=release+.25;hi=stop-.25;idx=np.flatnonzero((t>=lo)&(t<hi))
        r={'session':sid,'surface_id':name,'trial':int(row.trial),'release_s':release,'stop_s':stop,'release_utc':stamp(meta,release),
           'coast_duration_s':dur,'inverse_coast_time_per_s':1/dur if dur>0 else None,'feature_duration_s':max(0,hi-lo),'matched_image_location':False,
           'initial_speed_control':'user_reports_equal_not_instrumented','load_control':'user_reports_equal_numeric_mass_missing',
           'friction_force_N':None,'rolling_resistance_coefficient':None,'release_speed_mps':None,'rolling_distance_m':None,
           'excluded_measurement_error':excluded_trial(name,folder.name,int(row.trial)),'ble_max_gap_s':ble_gap(sid,release,stop)}
        flags=[]
        if r['excluded_measurement_error']:flags.append('excluded_measurement_error')
        if r['ble_max_gap_s'] is not None and r['ble_max_gap_s']>BLE_GAP_FLAG_S:flags.append('ble_gap_gt100ms')
        if dur<2:flags.append('short_trial_lt2s')
        restlo=stop+.5;resthi=stop+2.5;ri=np.flatnonzero((t>=restlo)&(t<=resthi))
        nextrelease=float(trials.iloc[j+1].release_s) if j+1<len(trials) else np.inf
        if resthi>=nextrelease:flags.append('rest_overlaps_next_release')
        moving=mean_interval(t,a[:,0],release,stop);rest=mean_interval(t,a[:,0],restlo,resthi)
        decel=rest-moving if moving is not None and rest is not None else None
        r['ax_decel_proxy_mps2']=decel
        # 자세(pitch)가 바뀌면 x축에 중력이 섞인다(1°당 약 0.17m/s²). x축 중력 성분 -g·sin(pitch)를 빼서 다시 계산한다.
        # 부호는 데이터로 골랐다: 이 보정에서 음수 감속이 7→3회로 줄고, 감속×시간(초기 속도 추정)이 걷는 속도(약 1.1m/s)에 모인다.
        lin=a[:,0]+9.80665*np.sin(np.deg2rad(d.pitch_deg.to_numpy()))
        mv2=mean_interval(t,lin,release,stop);rs2=mean_interval(t,lin,restlo,resthi)
        dc=rs2-mv2 if mv2 is not None and rs2 is not None else None
        r['ax_decel_pitch_corrected_mps2']=dc
        r['initial_speed_estimate_mps']=dc*dur if dc is not None else None   # 평균 감속 × 주행 시간 = 손 뗄 때 속도
        alt=[]
        for tag,startoff,endoff in [('early',.25,1.25),('late',1.5,3.5)]:
            baseline=mean_interval(t,a[:,0],stop+startoff,stop+endoff) if stop+endoff<nextrelease else None
            value=baseline-moving if baseline is not None and moving is not None else None
            r[f'ax_decel_{tag}_rest_proxy_mps2']=value
            if value is not None:alt.append(value)
        if decel is not None:alt.append(decel)
        r['rest_window_decel_range_mps2']=float(np.ptp(alt)) if len(alt)>1 else None
        if decel is None:flags.append('insufficient_rest_coverage')
        elif decel<=0:flags.append('nonpositive_ax_decel')
        if len(ri)>0:
            rv=float(np.sqrt(np.mean(np.sum((a[ri]-a[ri].mean(axis=0))**2,axis=1))))
            rg=float(np.sqrt(np.mean(np.sum(g[ri]**2,axis=1))))
            r['rest_acc_vector_std_mps2']=rv;r['rest_gyro_rms_rad_s']=rg
            if rv>.15 or rg>.05:flags.append('rest_motion_screen_failed')
        if len(idx)>=100:
            r.update(feat(bp[idx],g[idx],a[idx]));extras(r,idx,peaks,t,lo,hi,d)
            if len(ri):
                delta=float(d.pitch_deg.iloc[idx].median()-d.pitch_deg.iloc[ri].median())
                r['pitch_coast_minus_rest_deg']=delta
                r['gravity_x_sensitivity_mps2']=float(9.80665*abs(np.sin(np.deg2rad(d.pitch_deg.iloc[idx].median()))-np.sin(np.deg2rad(d.pitch_deg.iloc[ri].median()))))
                if abs(delta)>1:flags.append('pitch_change_gt1deg')
            if r['gyro_z_rms_rad_s']>.1:flags.append('turning_screen_gt0_1rad_s')
            thirds=np.array_split(idx,3)
            rr=[float(np.sqrt(np.mean(np.sum(bp[ii]**2,axis=1)))) for ii in thirds]
            r['rms_first_third_mps2'],r['rms_middle_third_mps2'],r['rms_last_third_mps2']=rr
            r['rms_last_first_ratio']=rr[-1]/rr[0] if rr[0] else None
            f,p=psd(bp[idx]);SPECTRA.extend({'session':sid,'surface_id':name,'trial':int(row.trial),'feature_duration_s':hi-lo,'hz':float(ff),'psd':float(pp)} for ff,pp in zip(f,p))
        else:flags.append('insufficient_interior_samples')
        if not original.empty:
            old=original[original.trial==row.trial]
            if len(old) and decel is not None:r['existing_decel_difference']=decel-float(old.iloc[0].mean_decel_m_s2)
        r['screen_flags']=';'.join(flags);r['decel_screen_pass']=not flags
        r['physical_resistance_status']='not_identifiable_without_verified_slope_speed_axis_mass'
        TRIALS.append(r)
        for thr,ii in peaks.items():
            for k in ii[(t[ii]>=release)&(t[ii]<stop)]:
                PEAKS.append({'session':sid,'surface_id':name,'trial':int(row.trial),'t_s':float(t[k]),'utc':stamp(meta,t[k]),'threshold_mps2':thr,'peak_mps2':float(np.linalg.norm(bp[k])),'boundary_within_025s':bool(t[k]<lo or t[k]>=hi),
                              'excluded_measurement_error':r['excluded_measurement_error']})
    # Show the original signal plus frequency evolution and trial markers.
    fig,axs=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    axs[0].plot(t,np.linalg.norm(bp,axis=1),lw=.5)
    for _,row in trials.iterrows():
        axs[0].axvspan(row.release_s,row.stop_s,color='#67a9cf',alpha=.2)
        axs[0].text(row.release_s,axs[0].get_ylim()[1]*.9,str(int(row.trial)),fontsize=8)
    axs[0].set(title=f'{name} — {folder.name}: shaded = marked coast',xlabel='Session time (s)',ylabel='Band-limited acceleration (m/s²)')
    total=None
    for k in range(3):
        f,tt,p=signal.spectrogram(bp[:,k],fs=FS,window='hann',nperseg=200,noverlap=150)
        total=p if total is None else total+p
    hosttime=np.interp(tt,np.arange(len(t))/FS,t)
    plot=axs[1].pcolormesh(hosttime,f,10*np.log10(total+1e-12),shading='auto',vmin=-35,vmax=10,cmap='magma')
    axs[1].set(ylim=(1,40),xlabel='Session time (s)',ylabel='Hz',title='STFT power: common color scale across recordings')
    fig.colorbar(plot,ax=axs[1],label='dB relative to 1 (m/s²)²/Hz')
    fig.savefig(OUT/'plots'/f'coast_{name}_{folder.name}.png',dpi=130);plt.close(fig)

# ---- imu 실험: 현장에서 표시한 붉은 노면(red_*)·블록(block_*) 구간. 재질 이름이 있는 유일한 주행 기록이다.
for folder in sorted((ROOT/'imu').glob('*/*')):
    if not (folder/'session.json').exists() or folder.parent.name in SKIP_FOLDERS:continue
    marks=read(folder/'marks.csv').sort_values('t_s').reset_index(drop=True)
    data=load_session(folder,'imu')
    if data is None:continue
    meta,d,t,a,g,bp,peaks,sid=data;run=folder.parent.name
    for j,m in marks.iterrows():
        label=str(m.label)
        if label.startswith(STOP_MARKS) or '_' not in label or label.startswith('mark'):continue
        surface,order=label.rsplit('_',1)
        start=float(m.t_s);end=float(marks.t_s.iloc[j+1]) if j+1<len(marks) else float(t[-1])
        end_source='next_mark' if j+1<len(marks) else 'recording_end'
        lo=start+IMU_SEGMENT_TRIM_S;hi=end-IMU_SEGMENT_TRIM_S;idx=np.flatnonzero((t>=lo)&(t<hi))
        base={'session':sid,'run':run,'mark':label,'surface':surface,'order':int(order) if order.isdigit() else order,
              'mark_s':start,'end_s':end,'end_source':end_source,'feature_start_s':lo,'feature_end_s':hi,'samples':len(idx),
              'ble_max_gap_s':ble_gap(sid,lo,hi),'utc':stamp(meta,lo)}
        if len(idx)<200:SEGMENTS.append({**base,'status':'too_short'});continue
        r={**base,'status':'ok',**feat(bp[idx],g[idx],a[idx])};extras(r,idx,peaks,t,lo,hi,d);SEGMENTS.append(r)
        # 같은 구간을 2초 창으로 나눠 분포를 본다(창 사이 비교·통계용).
        for k,w0 in enumerate(np.arange(lo,hi-2+1e-9,2.)):
            wi=np.flatnonzero((t>=w0)&(t<w0+2))
            if len(wi)<180:continue
            w={'session':sid,'run':run,'mark':label,'surface':surface,'window':k,'window_start_s':float(w0),'window_end_s':float(w0+2),
               'samples':len(wi),'ble_max_gap_s':ble_gap(sid,w0,w0+2),**feat(bp[wi],g[wi],a[wi])}
            extras(w,wi,peaks,t,w0,w0+2,d);SEG_WINDOWS.append(w)
    fig,ax=plt.subplots(figsize=(11,3.2),constrained_layout=True)
    ax.plot(t,np.linalg.norm(bp,axis=1),lw=.5,color='#333')
    colors={'red':'#d6604d','block':'#4393c3'}
    for r in [x for x in SEGMENTS if x['session']==sid and x['status']=='ok']:
        ax.axvspan(r['feature_start_s'],r['feature_end_s'],color=colors.get(r['surface'],'#999'),alpha=.25)
        ax.text(r['feature_start_s'],ax.get_ylim()[1]*.9,r['mark'],fontsize=8)
    ax.set(title=f'imu run {run}: red = red surface, blue = block (1 s trimmed at each mark)',xlabel='Session time (s)',ylabel='Band-limited acc (m/s²)')
    fig.savefig(OUT/'plots'/f'imu_run_{run}_{folder.name}.png',dpi=130);plt.close(fig)

for predfile in sorted((ROOT/'frames_v3_2s_wheel_ahead').glob('*/*/predictions.csv')):
    name=predfile.parent.parent.name;folder=ROOT/'frames'/name/predfile.parent.name
    if name in SKIP_FOLDERS:continue
    if not folder.exists():MATCHES.append({'prediction':str(predfile),'raw_found':False});continue
    pred=read(predfile);data=load_session(folder,'frames')
    if data is None:continue
    meta,d,t,a,g,bp,peaks,sid=data;gps=read(folder/'gps.csv');track(folder/'marks.csv')
    MATCHES.append({'prediction':str(predfile),'raw_found':True,'raw_session':str(folder),'join':'identical_folder_and_session_id'})
    for _,row in pred.iterrows():
        lo=float(row.window_start_s);hi=float(row.window_end_s);idx=np.flatnonzero((t>=lo)&(t<hi))
        if len(idx)<180 or lo<t[0]+1 or hi>t[-1]-1:continue
        r={'session':sid,'surface_id':name,'window_start_s':lo,'window_end_s':hi,'utc':stamp(meta,lo),'samples':len(idx),
           'predicted_material':row.get('surface_label'),'material_confidence':row.get('surface_confidence'),
           'camera_status':row.get('status'),'model_version':row.get('model_version'),'policy_version':row.get('policy_version'),'frame_t_s':row.get('t_s'),'road_sharpness':row.get('road_sharpness'),
           'selected_frame':str(predfile.parent/str(row.get('selected_frame'))),'ground_truth_material':None,
           'wheel_contact_alignment':'same_time_window_not_spatially_lag_corrected',**feat(bp[idx],g[idx],a[idx])}
        extras(r,idx,peaks,t,lo,hi,d)
        r['ble_max_gap_s']=ble_gap(sid,lo,hi);r['ble_gap_flag']=bool(r['ble_max_gap_s'] is not None and r['ble_max_gap_s']>BLE_GAP_FLAG_S)
        r['label_changed']=row.get('label_changed');r['previous_surface_label']=row.get('previous_surface_label')
        # 같은 노면이라도 빨리 가면 진동 주파수가 올라간다. 속도로 나누면 1m당 반복 수(블록 이음매 간격 후보)가 된다.
        speed,nfix=gps_speed(gps,lo,hi) if len(gps) else (None,0)
        r['gps_speed_mps']=speed;r['gps_speed_fixes']=nfix
        if speed is not None and speed>=.3:
            r['dominant_cycles_per_m']=r['dominant_hz']/speed;r['centroid_cycles_per_m']=r['centroid_hz']/speed
            r['rms_per_speed_mps2_per_mps']=r['rms_mps2']/speed
        if len(gps):
            k=int(np.argmin(abs(gps.t_s.to_numpy()-(lo+hi)/2)));gp=gps.iloc[k];age=abs(float(gp.t_s)-(lo+hi)/2)
            r['gps_time_offset_s']=age
            if age<=1.5:
                for c in ['latitude','longitude','accuracy_m']:r[c]=gp[c]
        # 진동이 작다고 정지로 보지 않는다. GPS 속도가 있을 때만 이동·정지를 붙이고, 없으면 미확인.
        r['motion_status']='not_verified' if speed is None else ('gps_moving' if speed>=.3 else 'gps_stationary_or_slow')
        WINDOWS.append(r)
    # Sample 3 frames as visual evidence, not manual material ground truth.
    good=pred[pred.selected_frame.notna()]
    photos=[]
    if len(good):
        for k in sorted(set([len(good)//4,len(good)//2,3*len(good)//4])):
            pp=predfile.parent/str(good.iloc[k].selected_frame)
            if pp.exists():photos.append(pp)
    if photos:
        canvas=Image.new('RGB',(900,205),'white');draw=ImageDraw.Draw(canvas)
        for i,pp in enumerate(photos):
            im=ImageOps.contain(Image.open(pp).convert('RGB'),(294,170));canvas.paste(im,(i*300,25));draw.text((i*300+4,5),f'{name}: {pp.stem}',fill='black')
        canvas.save(OUT/'plots'/f'frames_{name}.jpg');PHOTOS.append(name)

tr=pd.DataFrame(TRIALS);ws=pd.DataFrame(WINDOWS);sp=pd.DataFrame(SPECTRA)
sg=pd.DataFrame(SEGMENTS);sw=pd.DataFrame(SEG_WINDOWS)
sg.to_csv(OUT/'imu_segments_features.csv',index=False,encoding='utf-8-sig')
sw.to_csv(OUT/'imu_windows_2s_features.csv',index=False,encoding='utf-8-sig')
if len(sw):
    keep=['rms_mps2','vertical_rms_mps2','horizontal_rms_mps2','p95_mps2','crest_factor','centroid_hz','vertical_centroid_hz','dominant_hz',
          'fraction_3_8hz','fraction_8_15hz','fraction_15_25hz','jerk_rms_mps3','gyro_rms_rad_s','spectral_entropy_0_1']
    isum=sw.groupby(['run','surface'])[keep].median();isum['windows']=sw.groupby(['run','surface']).size()
    isum.to_csv(OUT/'imu_surface_summary.csv',encoding='utf-8-sig')
# 측정 오류 시험은 행은 남기고 요약·그림에서 뺀다.
tr_all=tr;tr=tr[~tr.excluded_measurement_error].copy()
sp=sp.merge(tr_all[['session','trial','excluded_measurement_error']],on=['session','trial'],how='left')
sp=sp[~sp.excluded_measurement_error.fillna(False).astype(bool)].drop(columns='excluded_measurement_error')
tr_all.to_csv(OUT/'coast_trials_features.csv',index=False,encoding='utf-8-sig')
ws.to_csv(OUT/'frames_2s_features.csv',index=False,encoding='utf-8-sig')
geo=[]
for row in json.loads(ws.to_json(orient='records')):
    if row.get('latitude') is None or row.get('longitude') is None:continue
    properties={k:row.get(k) for k in ['session','surface_id','window_start_s','utc','accuracy_m','gps_time_offset_s','rms_mps2','peak_mps2','centroid_hz','dominant_hz','fraction_1_3hz','fraction_3_8hz','fraction_8_15hz','fraction_15_25hz','fraction_25_40hz','predicted_material','material_confidence','model_version','policy_version','motion_status','wheel_contact_alignment',
                                  # 편한 길 안내에 쓰는 값: 속도, 끊김, 장착 방향과 무관한 수직 진동, 충격형 정도, 고주파, 큰 충격 수
                                  'gps_speed_mps','ble_max_gap_s','ble_gap_flag','vertical_rms_mps2','z_excess_kurtosis','power_25_40hz_mps4',
                                  'peak_candidates_gt10','dominant_cycles_per_m']}
    properties['feature_id']=hashlib.sha256(f"{row['session']}:{row['window_start_s']}".encode()).hexdigest()[:20]
    properties['source']='field_imu_derived_offline_v1';properties['road_cost']=None
    geo.append({'type':'Feature','geometry':{'type':'Point','coordinates':[row['longitude'],row['latitude']]},'properties':properties})
(OUT/'observations_features.geojson').write_text(json.dumps({'type':'FeatureCollection','features':geo},ensure_ascii=False))

sp.to_csv(OUT/'coast_psd.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(PEAKS).to_csv(OUT/'coast_peak_candidates.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(QUALITY).to_csv(OUT/'recording_quality.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(MATCHES).to_csv(OUT/'source_matches.csv',index=False,encoding='utf-8-sig')
(OUT/'source_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2))
summary=tr.groupby('surface_id').agg(trials=('trial','size'),duration_median_s=('coast_duration_s','median'),duration_min_s=('coast_duration_s','min'),duration_max_s=('coast_duration_s','max'),
     rms_median_mps2=('rms_mps2','median'),centroid_median_hz=('centroid_hz','median'),decel_proxy_median_mps2=('ax_decel_proxy_mps2','median'),screen_pass_trials=('decel_screen_pass','sum')).reset_index()
eligible=tr[tr.coast_duration_s>=2]
summary['trials_ge2s']=summary.surface_id.map(eligible.groupby('surface_id').size())
summary['duration_median_ge2s']=summary.surface_id.map(eligible.groupby('surface_id').coast_duration_s.median())
summary['rms_median_ge2s']=summary.surface_id.map(eligible.groupby('surface_id').rms_mps2.median())
summary.to_csv(OUT/'coast_surface_summary.csv',index=False,encoding='utf-8-sig')
metrics=['rms_mps2','centroid_hz','jerk_rms_mps3','crest_factor','fourth_moment_root_mps2','spectral_entropy_0_1','gyro_rms_rad_s']
fsum=ws.groupby('surface_id')[metrics].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
fsum.columns=[f'{a}_{ {"<lambda_0>":"q25","<lambda_1>":"q75"}.get(b,b)}' for a,b in fsum.columns]
fsum['windows']=ws.groupby('surface_id').size();fsum.to_csv(OUT/'frames_surface_summary.csv',encoding='utf-8-sig')
names=summary.surface_id.tolist();fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for ax,col,title in zip(axs,['coast_duration_s','rms_mps2','ax_decel_proxy_mps2'],['Coast time (s): same initial speed reported','Vibration RMS (m/s²): trial interior','Axial deceleration proxy (m/s²): NOT friction']):
    for i,name in enumerate(names):
        ss=tr[tr.surface_id==name];vals=ss[col].to_numpy();xx=np.linspace(i-.16,i+.16,len(vals))
        ax.scatter(xx,vals,c=['#d55e00' if x else '#0072b2' for x in ss.decel_screen_pass],s=23)
        ax.plot([i-.25,i+.25],[np.nanmedian(vals)]*2,color='black',lw=2)
    ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'plots'/'coast_comparison.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
for ax,name in zip(axs.ravel(),names):
    ss=sp[(sp.surface_id==name)&(sp.feature_duration_s>=2)]
    for _,gg in ss.groupby(['session','trial']):ax.semilogy(gg.hz,gg.psd,color='#8aabc2',alpha=.5,lw=.7)
    med=ss.groupby('hz').psd.median();ax.semilogy(med.index,med.values,color='#143f62',lw=2)
    ax.set(title=name,xlim=(1,40),ylim=(1e-7,10),xlabel='Hz',ylabel='PSD ((m/s²)²/Hz)')
fig.savefig(OUT/'plots'/'coast_spectra.png',dpi=150);plt.close(fig)

definitions={
 'wavelet_fraction_*':'db4 DWT, four levels, periodization, summed three-axis coefficient energy fractions. Approximate bands A4:0–3.125, D4:3.125–6.25, D3:6.25–12.5, D2:12.5–25, D1:25–50Hz. Input already 1–40Hz filtered; extension affects boundaries. Not exact FFT bands.',
 'inverse_coast_time_per_s':'1/coast duration = average deceleration per unit initial speed if true release and full stop are marked. Not absolute deceleration, force or friction. Initial speed equality was reported by user.',
 'rest_window_decel_range_mps2':'Range of axial proxies across rest windows +0.25..1.25,+0.5..2.5,+1.5..3.5s; omit alternative windows crossing next release. Sensitivity diagnostic, not noise confidence interval.',
 'acc_norm_std_mps2':'Standard deviation of raw acceleration magnitude; orientation-invariant magnitude but still contains gravity/dynamic coupling; not linear acceleration.',
 'acc_norm_p95_p5_mps2':'95th–5th percentile span of raw acceleration magnitude; robust unfiltered variability.',
 'rms_mps2':'sqrt(mean(sum of squared 1–40Hz filtered acceleration axes)); body-frame resultant, not ISO weighted comfort',
 'ax_decel_proxy_mps2':'mean raw ax in stop+0.5..2.5s minus mean raw ax during release..stop, time weighted; axis and pitch sensitive',
 'ax_decel_pitch_corrected_mps2':'ax에서 x축 중력 성분(-g·sin(pitch))을 빼고 같은 방식(정지 평균 - 주행 평균)으로 구한 감속도. 장착 0점 보정은 없음',
 'initial_speed_estimate_mps':'보정 감속도 × 주행 시간. 같은 속도로 밀었다면 시험끼리 비슷해야 하는 검증용 값',
 'decel_screen_pass':'Exploratory quality screen only: duration>=2s, sufficient nonoverlapping rest, positive proxy, rest acceleration std<=0.15m/s² and gyro RMS<=0.05rad/s, pitch change<=1deg, gyro-z RMS<=0.1rad/s. Passing does not validate friction.',
 'dominant_hz':'Maximum of summed axis Welch PSD in 1–40Hz; resolution 0.5Hz for >=2s samples',
 'centroid_hz':'PSD weighted average frequency in 1–40Hz',
 'spectral_entropy_0_1':'Normalized Shannon entropy of 1–40Hz PSD bins',
 'spectral_flatness_0_1':'geometric mean PSD / arithmetic mean PSD',
 'fraction_*hz':'Band power / total 1–40Hz power; nonoverlapping bins',
 'jerk_rms_mps3':'RMS vector time derivative after bandpass filtering; filter-dependent',
 'unweighted_dose_proxy_mps175':'(integral resultant filtered acceleration^4 dt)^0.25; NOT ISO VDV; depends on duration',
 'fourth_moment_root_mps2':'Fourth moment root normalized by duration; emphasizes peaks',
 'autocorr_peak_0_05_1s':'Maximum normalized summed-axis autocorrelation in 0.05–1s lag; not wheel spacing',
 'z_gyro_y_coherence_mean':'Mean Welch magnitude-squared coherence 1–40Hz, >=4s only, 1s Hann windows; sensor axes not verified as vertical/pitch',
 'pitch_*':'Raw sensor attitude, no mounting zero calibration; not road grade',
 'peak_candidates_gt*':'Peaks of resultant filtered acceleration, threshold 5/10/20m/s², prominence threshold/2, min separation0.25s; not validated defects',
 'friction_force_N':'Unavailable: numeric mass, verified translational deceleration, grade and wheel losses absent',
 'rolling_resistance_coefficient':'Unavailable: not equivalent to sliding friction; slope/steering/bearing effects unseparated',
 'frames_summary':'All eligible 2s windows; includes unknown stationary/moving states. Folder names and model predictions are not material ground truth',
 'vertical_*':'필터한 가속도를 구간 평균 중력 방향에 투영한 성분. 센서 장착 축과 무관. 경사로에서는 노면 법선과 다를 수 있음',
 'horizontal_rms_mps2':'중력 방향에 수직인 성분의 RMS(전후·좌우 합)',
 'vertical_energy_share':'수직 성분 에너지 / 전체 에너지',
 'gravity_tilt_from_sensor_z_deg':'중력 방향과 센서 z축 사이 각도. 장착 기울기 진단용',
 'ble_max_gap_s':'구간 안 가장 긴 블루투스 알림 간격(초). imu.csv t_s는 수신기가 붙인 값이라 끊김은 이것으로 확인. 0.1초 초과면 표시',
 'gps_speed_mps':'창 ±0.5초 안 GPS 점(정확도 10m 이하) 이동 거리/시간. 휴대폰 GPS 오차가 커서 저속에서 불안정',
 '*_cycles_per_m':'주파수(Hz)/GPS 속도(m/s) = 1m당 반복 수. 속도 차이를 지운 공간 주파수 후보(블록 이음매 간격 등). 속도 0.3m/s 이상만',
 'rms_per_speed_mps2_per_mps':'진동 RMS / GPS 속도. 빠를수록 커지는 효과를 단순 보정한 탐색용 값',
 'imu_segments':'imu 실험의 red_*/block_* 표시 사이 구간. 앞뒤 1초를 잘라 경계 효과를 뺌. 마지막 구간은 route_end 또는 기록 끝까지',
 'excluded_measurement_error':'데이터 안내서의 측정 오류 시험(coast block 4번째, block4 3·4번째). 행은 남기고 요약·그림에서 제외',
}
(OUT/'feature_dictionary.json').write_text(json.dumps(definitions,ensure_ascii=False,indent=2))
import importlib.metadata as _md
config={'sample_rate_config_hz':FS,'sensor_bandwidth_hz':42,'bandpass_hz':[1,40],'filter':'Butterworth order4 sosfiltfilt offline',
        'welch_seconds':2,'welch_overlap':.5,'stft_hop_seconds':.5,'wavelet':'db4 level4 periodization','shock_thresholds_mps2':[5,10,20],
        'coast_edge_trim_seconds':.25,'coast_rest_window_s':[.5,2.5],'coast_excluded_trials':sorted([list(x[:2]) for x in EXCLUDED_COAST]),
        'coast_excluded_rule':'폴더명+trial (회차 구분 없음)','imu_segment_trim_seconds':IMU_SEGMENT_TRIM_S,'ble_gap_flag_seconds':BLE_GAP_FLAG_S,
        'skipped_folders':sorted(SKIP_FOLDERS),'vertical_axis':'구간 평균 중력 방향','gps_speed':'창±0.5초, 정확도 10m 이하 점 2개 이상, 1.5초 이상',
        'gps_window_join_seconds':1.5,'pitch_gravity_correction':'ax + g·sin(pitch)',
        'user_confirmed_equal_initial_speed':True,'initial_speed_instrumented':False,'user_confirmed_equal_load':True,'mass_kg':None,
        'coast_to_frames_location_mapping_verified':False,
        'libraries':{n:_md.version(n) for n in ['numpy','scipy','pandas','matplotlib','PyWavelets','pillow']}}
(OUT/'analysis_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
method='''<h2>해석과 사용 방법</h2><ul>
<li>사용자 확인: 관성 시험은 동일한 초기 속도·하중으로 밀고 손을 놓음. 실제 속도 수치·질량·노면 경사·타이어 조건은 파일에 없어 기록된 통제 조건으로만 취급.</li>
<li>관성 8종은 coast/block~block5, red_good, red_worst. 사진 폴더 block~block7과 장소 대응이 확인되지 않아 합치지 않음. 이름 good/worst는 제공된 식별자이며 분석 판정이 아님.</li>
<li>지속 시간은 동일 출발 속도·경사·방향·타이어·무제동 조건에서 비교 가능. 오래 굴렀다고 마찰력이 작다고 확정할 수 없음. 노면 진동과 굴림 저항은 별도 속성.</li>
<li>원시 가속도 m/s², 각속도 rad/s. 설정 100Hz 샘플 순서로 DSP, 호스트 시각으로 프레임·GPS·표시 연결. 기기 하드웨어 타임스탬프는 없으며 42Hz는 설정값. 앨리어싱 부재를 보증하지 않음.</li>
<li>각 축 4차 Butterworth 1–40Hz, 오프라인 전후방 필터. 관성 시험은 시작/끝 0.25초 제외. 중력 DC/느린 성분을 억제한 진동이지, 완전한 자세 보정 선형 가속도가 아님. 감속 계산에는 대역통과 신호를 쓰지 않음.</li>
<li>FFT 기반 Welch PSD: 2초 Hann, 50% 중첩. STFT: 2초 창,0.5초 간격. 대역 1–3/3–8/8–15/15–25/25–40Hz, 분포/자기상관/축간 coherence 추출. 주파수는 속도와 장착·바퀴 공진의 영향을 받음. db4 4단계 웨이블릿 계수 에너지 비율도 추가(periodization, 경계 영향 있음).</li>
<li>정지 기준 검사: stop+0.5~2.5초, 다음 release와 겹침 검사, 가속도 벡터 표준편차0.15m/s²·각속도 RMS0.05rad/s 초과를 탐색용으로 표시. 다음 밀기 시작은 release보다 빠를 수 있어 통과해도 정지를 보증하지 않음.</li>
<li>기존 감속도는 raw X축 정지 평균−주행 평균. 음수·자세 변화·회전·정지 불안정은 플래그로 보존. 임의로 절댓값을 취하거나 음수를 삭제해 좋은 노면 순위를 만들지 않음.</li>
<li>질량·검증된 감속·경사가 없어 마찰력(N), 굴림 저항계수, 초기속도·굴림거리는 빈 값. IMU 적분으로 이를 확정하지 않음. ISO 승차감/VDV 또는 안전 인증 수치가 아님.</li>
<li>카메라는 동일 세션·2초 창으로 조인하고 GPS는 창 중심 ±1.5초 최근값만 사용. 바퀴 접촉 시차는 미보정. 정지/이동 표시가 없으므로 폴더 전체 RMS 평균을 노면 순위로 사용하지 않음.</li>
<li>현재 우선 사용: RMS·피크·주파수 대역 비율을 관측 속성으로 저장. 시험별 반복 분산을 보고, 경사·위치 대응 확인 후 길별 비용 정책을 별도 검증. 특징 수가 많다는 것이 재질 판별력이 검증됐다는 뜻은 아님.</li></ul>
<p>참조: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7302523/">표준 속도 구간·양방향 coast-down 연구</a>, <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html">Welch PSD</a>, <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html">Coherence</a></p>'''
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>8종 노면 관성·진동 특징</title><style>body{font-family:-apple-system,sans-serif;max-width:1200px;margin:28px auto;padding:0 18px;line-height:1.65;color:#1d3249}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f9}img{max-width:100%;height:auto}.scroll{overflow:auto}.note{background:#fff1d5;padding:16px}li{margin:9px 0}a{color:#1761a8}</style><h1>8종 노면 관성·진동 특징 분석</h1>'''
page+=f'<p>관성 시험 {len(tr_all)}회(측정 오류 {int(tr_all.excluded_measurement_error.sum())}회 제외 후 {len(tr)}회) · imu 구간 {len(sg)}개 · 시각 기록 {len(MATCHES)}개 세션 · 2초 특징 {len(ws):,}개. 원본 유지.</p><p class="note">진동 특징 추출 완료. 마찰력과 굴림 저항계수는 현재 자료만으로 확정할 수 없습니다. 사진 기록과 관성 기록의 장소 대응은 미확인입니다.</p>'
show=summary[['surface_id','trials','duration_median_s','duration_min_s','duration_max_s','rms_median_mps2','centroid_median_hz','decel_proxy_median_mps2','screen_pass_trials']].rename(columns={'surface_id':'관성 장소 ID','trials':'시험 수','duration_median_s':'관성 시간 중앙값(초)','duration_min_s':'최소(초)','duration_max_s':'최대(초)','rms_median_mps2':'진동 RMS 중앙값(m/s²)','centroid_median_hz':'중심 주파수(Hz)','decel_proxy_median_mps2':'축 감속 대리지표(m/s²)','screen_pass_trials':'탐색 검사 통과 수'})
page+='<div class="scroll">'+show.to_html(index=False,float_format=lambda x:f'{x:.3f}')+'</div><p>표는 장소별 시험 중앙값입니다. 품질 플래그가 있는 시험도 보존하며, 통과 수는 탐색 검사이지 물리적 타당성 인증이 아닙니다.</p>'
med=tr.groupby('surface_id')[['rms_mps2','coast_duration_s']].median()
obs=''
if {'red_worst','red_good'}<=set(med.index):
    obs=(f"red_worst는 red_good보다 진동 RMS 중앙값이 약 {med.rms_mps2['red_worst']/med.rms_mps2['red_good']:.1f}배이지만 관성 시간 중앙값은 "
         f"{med.coast_duration_s['red_worst']:.2f}초와 {med.coast_duration_s['red_good']:.2f}초입니다. 진동과 굴림 지속성은 별도 지표입니다. ")
neg=int((tr.ax_decel_proxy_mps2<=0).sum());rest=int(tr.screen_flags.fillna('').str.contains('rest_motion_screen_failed').sum())
page+=(f'<p><b>핵심 관찰:</b> {obs}측정 오류로 안내된 {int(tr_all.excluded_measurement_error.sum())}회는 요약과 그림에서 뺐습니다. '
       f'남은 {len(tr)}회 중 감속 대리값이 0 이하인 시험은 {neg}회, 정지 구간 움직임 기준을 넘은 시험은 {rest}회입니다. '
       '기준을 넘었다고 바로 무효는 아니며 정지 구간 재확인이 필요하다는 뜻입니다. 2초 미만 시험을 뺀 결과는 CSV의 ge2s 열에 있습니다.</p>')
if len(sw):
    page+='<h2>imu 실험: 붉은 노면과 블록의 진동 (2초 창 중앙값)</h2><div class="scroll">'+isum.reset_index().to_html(index=False,float_format=lambda x:f'{x:.3f}')+'</div>'
    page+='<p>현장 표시(marks.csv) 사이 구간에서 앞뒤 1초를 뺐습니다. 재질 이름이 표시된 유일한 주행 기록이라 특징 비교의 기준으로 씁니다.</p>'
    for path in sorted((OUT/'plots').glob('imu_run_*.png')):page+=f'<img loading="lazy" src="plots/{path.name}">'
page+='<img src="plots/coast_comparison.png"><p>점은 개별 시험, 검은 선은 중앙값. 주황: 탐색 검사 통과, 파랑: 확인 필요.</p><img src="plots/coast_spectra.png"><p>주파수 비교 그림은 내부 분석 길이 2초 이상인 시험만 포함하며 모든 선은 0.5Hz 간격입니다. 짧은 시험의 원시 스펙트럼은 CSV에 별도 보존했습니다.</p>'+method
page+='<h2>분석 파일</h2><ul>'
for file,label in [('imu_segments_features.csv','imu 실험 노면 구간별 특징'),('imu_windows_2s_features.csv','imu 실험 2초 창 특징'),('imu_surface_summary.csv','imu 실험 회차·노면별 중앙값'),('coast_trials_features.csv','관성 시험별 전체 특징·감속 검사(제외 표시 포함)'),('coast_surface_summary.csv','8종 관성 요약'),('frames_2s_features.csv','시각·GPS·IMU 결합 2초 특징'),('observations_features.geojson','지도에 올릴 관측 점과 진동 특징(길 연결·비용 미확정)'),('frames_surface_summary.csv','시각 기록 폴더별 특징 분포'),('coast_psd.csv','관성 시험별 주파수 스펙트럼'),('coast_peak_candidates.csv','충격 후보 시각(기준별 중복)'),('recording_quality.csv','수집 품질'),('feature_dictionary.json','특징 정의')]:page+=f'<li><a href="{file}">{label}</a></li>'
page+='</ul><h2>관성 시험의 시간·주파수 변화</h2>'
for path in sorted((OUT/'plots').glob('coast_*Z.png')):page+=f'<details><summary>{html.escape(path.stem)}</summary><img loading="lazy" src="plots/{path.name}"></details>'
page+='<h2>사진 기록 식별용 표본</h2><p>동일 이름의 coast 자료와 같은 장소라는 뜻은 아닙니다.</p>'
for name in PHOTOS:page+=f'<details><summary>{name}</summary><img loading="lazy" src="plots/frames_{name}.jpg"></details>'
(OUT/'index.html').write_text(page+'</html>')
print(summary.to_string(index=False));print('Trials:',len(tr_all),'excluded:',int(tr_all.excluded_measurement_error.sum()),'windows:',len(ws),
      'imu segments:',len(sg),'imu 2s windows:',len(sw),'columns:',len(ws.columns),'sources:',len(MANIFEST))
print('Screen flags',tr_all.screen_flags.value_counts().to_string())
