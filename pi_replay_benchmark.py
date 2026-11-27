"""녹화한 사진을 Pi에서 실시간 속도로 다시 흘려 보내 처리 간격별 비용을 잰다.

현장 녹화(record_replay.py)의 사진을 촬영 때와 같은 시각 간격으로 가짜 카메라에 넣고,
실제 수집 코드와 같은 방식으로 분석시킨다. 같은 사진을 쓰므로 PC replay(compare_replay.py)의
정확도 결과와 구간 단위로 맞대어 볼 수 있다.

    python pi_replay_benchmark.py recordings/campus_<시각> --intervals 1 2 4 8

재는 것 (처리 간격마다)
    사진 한 장의 단계별 시간(ms)   전처리, Area, Material, 후처리, 합계. 품질 검사(밝기·선명도) 시간도 따로
    처리하지 못한 사진 수          카메라가 내보낸 사진 - 분석한 사진. 읽지도 못하고 버려진 사진은 따로
    CPU·메모리·온도               1초마다 이 프로세스 CPU, 시스템 CPU, RSS, 최고 온도, 최저 CPU 클럭
    결과 지연                     구간이 끝난 뒤 결과가 나오기까지 걸린 시간

분석 방식 (--policy)
    live     main.py와 같은 camera_inference.iter_windows. 3장마다 가장 선명한 사진을 분석
    timed    구간마다 한 장, 구간의 마지막 사진 (compare_replay 방법 A/B)
    quality  구간마다 한 장, 밝기 검사 + 선명도·자이로 순위로 고른 사진 (compare_replay 방법 C/D)

한계: 실제 수집에서 함께 도는 카메라 인코딩, BLE(IMU), GPS 수신, 업로드 부하는 빠진다.
짧은 실제 수집(main.py)의 timing_ms와 비교해 차이를 확인한다.
"""
import argparse
import csv
import json
import math
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import camera_inference
import inference

POLICIES = ("live", "timed", "quality")
STAGES = ("preprocess", "area", "material", "postprocess", "total")
GYRO_WINDOW_S = 0.2   # compare_replay와 같은 자이로 RMS 구간
# compare_replay.py의 방법 이름. 모델 폴더 이름으로 PC 결과의 어느 방법과 비교할지 정한다.
PC_METHOD = {("area-surface-v3", "timed"): "A", ("finetuned-v2-onnx", "timed"): "B",
             ("area-surface-v3", "quality"): "C", ("finetuned-v2-onnx", "quality"): "D"}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_rows(path):
    with path.open(newline="") as source:
        return sorted(csv.DictReader(source), key=lambda row: float(row["t_s"]))


def percentile(values, q):
    """선형 보간 백분위수. 값이 없으면 None."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "p50": round(percentile(values, 50), 1),
            "p95": round(percentile(values, 95), 1), "max": round(max(values), 1)}


# ---------------------------------------------------------------- 가짜 카메라

class ReplayCamera:
    """녹화 사진을 원래 시각 간격대로 내보낸다. vision.capture_frame이 그대로 읽을 수 있는 형태다.

    실제 카메라처럼 별도 스레드에서 JPEG를 디코딩하고, 큐에는 가장 최근 사진 한 장만 둔다.
    분석하느라 읽지 못한 사이에 새 사진이 오면 이전 사진은 버려진다(dropped).
    source_fps를 주면 그 속도로 내보내되 각 시점의 가장 최근 녹화 사진을 쓴다(라이브 카메라 30fps 부하 흉내).
    """

    def __init__(self, recording, rows, start_s, duration_s, source_fps=None, tail_s=1.0):
        import cv2
        self.cv2 = cv2
        self.recording = recording
        # 마지막 구간이 사진을 기다리며 멈추지 않도록 측정 구간 뒤 tail_s초까지 사진을 더 내보낸다.
        # 구간 경계 밖 사진은 어느 구간에도 세지 않는다.
        duration_s += tail_s
        self.rows = [r for r in rows if start_s <= float(r["t_s"]) < start_s + duration_s]
        if not self.rows:
            raise ValueError(f"{start_s}~{start_s + duration_s}초 구간에 녹화 사진이 없습니다.")
        self.times = [float(r["t_s"]) for r in self.rows]
        self.start_s, self.duration_s, self.source_fps = start_s, duration_s, source_fps
        self.camera = {"frames": queue.Queue(maxsize=1), "stop_event": threading.Event(),
                       "error": None, "thread": None}
        self.emitted = []        # (frame_monotonic, 녹화 t_s, 파일)
        self.dropped = []        # 큐에서 밀려나 아무도 읽지 않은 사진의 frame_monotonic
        self.decode_ms = []
        self.late_ms = []        # 예정 시각보다 늦게 내보낸 정도(CPU가 바빠 리더 스레드가 밀린 경우)
        self.origin = None

    def schedule(self):
        """(예정 시각 오프셋, 녹화 행) 목록."""
        if not self.source_fps:
            return [(t - self.start_s, row) for t, row in zip(self.times, self.rows)]
        out = []
        for k in range(int(self.duration_s * self.source_fps)):
            offset = k / self.source_fps
            i = bisect_right(self.times, self.start_s + offset) - 1
            if i >= 0:
                out.append((offset, self.rows[i]))
        return out

    def start(self, origin):
        self.origin = origin
        self.camera["thread"] = threading.Thread(target=self._run, daemon=True)
        self.camera["thread"].start()

    def _run(self):
        frames = self.camera["frames"]
        for offset, row in self.schedule():
            if self.camera["stop_event"].is_set():
                return
            due = self.origin + offset
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                self.late_ms.append(-wait * 1000)
            t0 = time.perf_counter()
            jpeg = (self.recording / row["file"]).read_bytes()
            image = self.cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), self.cv2.IMREAD_COLOR)
            self.decode_ms.append((time.perf_counter() - t0) * 1000)
            if image is None:
                print(f"사진을 디코딩하지 못했습니다: {row['file']}", file=sys.stderr)
                continue
            stamp = time.monotonic()
            frame = {"image": image, "jpeg_bytes": jpeg, "frame_time": utc_now(), "frame_monotonic": stamp,
                     "recording_t_s": float(row["t_s"]), "file": row["file"]}
            self.emitted.append((stamp, float(row["t_s"]), row["file"]))
            try:
                frames.put_nowait(frame)
            except queue.Full:
                try:
                    old = frames.get_nowait()
                    self.dropped.append(old["frame_monotonic"])
                except queue.Empty:
                    pass
                frames.put_nowait(frame)

    def stop(self):
        self.camera["stop_event"].set()
        if self.camera["thread"] is not None:
            self.camera["thread"].join(timeout=5)


# ---------------------------------------------------------------- 시스템 사용량

def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _system_cpu_times():
    line = _read("/proc/stat")
    if not line:
        return None
    fields = [int(x) for x in line.splitlines()[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


def _rss_mb():
    status = _read("/proc/self/status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        pass
    try:   # macOS 등 /proc가 없는 곳(개발 확인용)
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _max_temp_c():
    temps = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        value = _read(zone)
        if value and value.lstrip("-").isdigit():
            temps.append(int(value) / 1000)
    return max(temps) if temps else None


def _min_freq_mhz():
    freqs = []
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        value = _read(f)
        if value and value.isdigit():
            freqs.append(int(value) / 1000)
    return min(freqs) if freqs else None


class SystemSampler:
    """period_s마다 CPU·메모리·온도를 기록하는 스레드. Linux가 아니면 없는 값은 비운다."""

    def __init__(self, period_s=1.0):
        self.period_s = period_s
        self.rows = []
        self.tag = {}
        self._stop = threading.Event()
        self._thread = None
        self.cpu_count = os.cpu_count() or 1

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        last_wall, last_proc, last_sys = time.monotonic(), sum(os.times()[:2]), _system_cpu_times()
        while not self._stop.wait(self.period_s):
            wall, proc, sys_times = time.monotonic(), sum(os.times()[:2]), _system_cpu_times()
            dt = wall - last_wall
            system_pct = None
            if sys_times and last_sys and sys_times[0] > last_sys[0]:
                busy = (sys_times[0] - last_sys[0]) - (sys_times[1] - last_sys[1])
                system_pct = 100 * busy / (sys_times[0] - last_sys[0])
            self.rows.append({**self.tag, "monotonic": round(wall, 3),
                              "process_cpu_pct": round(100 * (proc - last_proc) / dt, 1),   # 코어 하나 = 100
                              "system_cpu_pct": None if system_pct is None else round(system_pct, 1),  # 전체 코어 = 100
                              "rss_mb": None if (r := _rss_mb()) is None else round(r, 1),
                              "temp_c": _max_temp_c(), "min_cpu_freq_mhz": _min_freq_mhz()})
            last_wall, last_proc, last_sys = wall, proc, sys_times

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------- 시간 재기

class Meter:
    """inference.analyze_image와 품질 검사 함수를 감싸 호출마다 시간을 남긴다."""

    def __init__(self):
        self.inferences = []    # {"monotonic", "wall_ms", stage...}
        self.quality_ms = []    # 사진 한 장의 밝기 통계 + 선명도 계산 시간
        self._patched = []

    def patch(self):
        original_analyze = inference.analyze_image
        original_stats = camera_inference.frame_stats
        original_sharpness = camera_inference.road_sharpness

        def analyze(image, models, timing=False):
            t0 = time.perf_counter()
            result = original_analyze(image, models, True)
            wall = (time.perf_counter() - t0) * 1000
            self.inferences.append({"monotonic": time.monotonic(), "wall_ms": round(wall, 1),
                                    **(result.get("timing_ms") or {}),
                                    "surface_label": result.get("surface_label")})
            if not timing:
                result.pop("timing_ms", None)
            return result

        def timed(func):
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    self.quality_ms.append((time.perf_counter() - t0) * 1000)
            return wrapper

        for module, name, replacement in ((inference, "analyze_image", analyze),
                                          (camera_inference, "frame_stats", timed(original_stats)),
                                          (camera_inference, "road_sharpness", timed(original_sharpness))):
            self._patched.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)

    def unpatch(self):
        for module, name, original in reversed(self._patched):
            setattr(module, name, original)
        self._patched.clear()


# ---------------------------------------------------------------- 분석 방식

def gyro_lookup(recording):
    imu = read_rows(recording / "imu.csv")
    times = np.array([float(r["t_s"]) for r in imu])
    speed_sq = np.array([sum(float(r[k]) ** 2 for k in ("gx", "gy", "gz")) for r in imu])

    def rms(t):
        lo, hi = times.searchsorted([t - GYRO_WINDOW_S, t], side="right")
        return float(np.sqrt(speed_sq[lo:hi].mean())) if hi > lo else None
    return rms


def run_live(camera, models, interval_s, n_windows):
    """main.py와 같은 iter_windows를 그대로 쓴다."""
    windows = camera_inference.iter_windows(camera.camera, models, interval_s, True, sampler=None,
                                            save_sharp_frames=False, save_rejected_frames=False,
                                            camera_config={"save_sharp_frames": False, "save_rejected_frames": False,
                                                           "sharp_frame_dir": "sharp_frames"})
    out = []
    try:
        for window in windows:
            done = time.monotonic()
            result = window["result"] or {}
            out.append({"start_monotonic": window["start_monotonic"], "end_monotonic": window["end_monotonic"],
                        "done_monotonic": done, "read": window["frame_count"],
                        "inferred": window["inference_count"], "rejected": window["rejected_count"],
                        "surface_label": result.get("surface_label"), "file": None})
            if len(out) >= n_windows:
                break
    finally:
        windows.close()
    return out


def run_single(camera, models, interval_s, n_windows, policy, origin, gyro_rms):
    """구간마다 한 장만 분석한다(compare_replay 방법 A~D와 같은 선택 규칙)."""
    if policy == "quality":
        from compare_replay import quality_pick
    out = []
    pending = None
    for index in range(n_windows):
        start, end = origin + index * interval_s, origin + (index + 1) * interval_s
        group, read = [], 0
        while True:
            if pending is not None:
                frame, pending = pending, None
            else:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    frame = camera.camera["frames"].get(timeout=remaining)
                except queue.Empty:
                    break
            stamp = frame["frame_monotonic"]
            if stamp >= end:
                pending = frame            # 다음 구간 사진
                break
            if stamp < start:
                continue                   # 이전 구간 사진을 늦게 읽음. 이 구간 후보가 아님
            read += 1
            meta = {"file": frame["file"], "t_s": frame["recording_t_s"], "image": frame["image"]}
            if policy == "quality":
                image = frame["image"]
                stats = camera_inference.frame_stats(image)
                meta.update(quality=camera_inference.frame_problem(image, stats) or "ok",
                            road_sharpness=camera_inference.road_sharpness(image),
                            gyro_rms_rad_s=gyro_rms(frame["recording_t_s"]))
            group.append(meta)
        picked = (group[-1] if group else None) if policy == "timed" else quality_pick(group)
        result = None
        if picked is not None:
            try:
                result = inference.analyze_image(picked["image"], models, True)
            except Exception as error:
                print(f"분석 실패: {error}", file=sys.stderr)
        out.append({"start_monotonic": start, "end_monotonic": end, "done_monotonic": time.monotonic(),
                    "read": read, "inferred": int(result is not None),
                    "rejected": sum(1 for g in group if g.get("quality", "ok") != "ok"),
                    "surface_label": (result or {}).get("surface_label"),
                    "file": picked["file"] if picked else None})
    return out


# ---------------------------------------------------------------- 한 번의 측정

def measure(recording, rows, models, meter, sampler, *, policy, interval_s, start_s, duration_s,
            source_fps, warmup, gyro_rms):
    n_windows = int(duration_s // interval_s)
    if n_windows < 1:
        raise ValueError(f"측정 시간 {duration_s}초가 간격 {interval_s}초보다 짧습니다.")
    camera = ReplayCamera(recording, rows, start_s, n_windows * interval_s, source_fps, tail_s=interval_s)
    first = camera.cv2.imread(str(recording / camera.rows[0]["file"]))
    for _ in range(warmup):                       # 첫 호출의 초기화 비용은 빼고 잰다
        inference.analyze_image(first, models, True)
    meter.inferences.clear()
    meter.quality_ms.clear()
    sampler.tag = {"policy": policy, "interval_s": interval_s}
    first_sample = len(sampler.rows)
    origin = time.monotonic()
    camera.start(origin)
    try:
        if policy == "live":
            windows = run_live(camera, models, interval_s, n_windows)
        else:
            windows = run_single(camera, models, interval_s, n_windows, policy, origin, gyro_rms)
    finally:
        camera.stop()
    sampler.tag = {}
    system = sampler.rows[first_sample:]

    # 사진·분석 기록을 구간에 나눠 담는다(구간 경계는 각 방식이 실제로 쓴 값).
    emitted = [e[0] for e in camera.emitted]
    dropped = sorted(camera.dropped)
    calls = [c["monotonic"] for c in meter.inferences]
    for i, w in enumerate(windows):
        lo, hi = w["start_monotonic"], w["end_monotonic"]
        w["index"] = i
        w["delivered"] = bisect_left(emitted, hi) - bisect_left(emitted, lo)
        w["dropped"] = bisect_left(dropped, hi) - bisect_left(dropped, lo)
        w["unprocessed"] = w["delivered"] - w["inferred"]
        w["lag_ms"] = round((w["done_monotonic"] - hi) * 1000, 1)
        prev_done = windows[i - 1]["done_monotonic"] if i else lo
        in_window = [c for c, t in zip(meter.inferences, calls) if prev_done < t <= w["done_monotonic"]]
        w["busy_ms"] = round(sum(c["wall_ms"] for c in in_window), 1)
    inferences = [{"policy": policy, "interval_s": interval_s, **c} for c in meter.inferences]
    order = ("policy", "interval_s", "index", "delivered", "read", "inferred", "rejected", "unprocessed", "dropped",
             "busy_ms", "lag_ms", "surface_label", "file", "start_monotonic", "end_monotonic", "done_monotonic")
    windows = [{"policy": policy, "interval_s": interval_s, **w} for w in windows]
    windows = [{k: w[k] for k in order} for w in windows]
    summary = summarize(policy, interval_s, windows, inferences, meter.quality_ms, camera, system)
    return windows, inferences, system, summary


def summarize(policy, interval_s, windows, inferences, quality_ms, camera, system):
    total = lambda key: sum(w[key] for w in windows)
    lags = [w["lag_ms"] for w in windows]
    busy = [w["busy_ms"] for w in windows]
    delivered = total("delivered")
    summary = {
        "policy": policy, "interval_s": interval_s, "windows": len(windows),
        "duration_s": round(len(windows) * interval_s, 1),
        "frames": {"delivered": delivered, "read": total("read"), "inferred": total("inferred"),
                   "rejected_quality": total("rejected"), "unprocessed": total("unprocessed"),
                   "dropped_unread": total("dropped"),
                   "unprocessed_pct": round(100 * total("unprocessed") / delivered, 1) if delivered else None,
                   "per_window": {k: round(total(k) / len(windows), 2)
                                  for k in ("delivered", "read", "inferred", "unprocessed", "dropped")}},
        "stage_ms": {s: describe([c.get(s) for c in inferences]) for s in STAGES},
        "inference_wall_ms": describe([c["wall_ms"] for c in inferences]),
        "quality_check_ms_per_call": describe(quality_ms),
        "decode_ms": describe(camera.decode_ms),
        "camera_late_ms": describe(camera.late_ms),
        "window_busy_ms": describe(busy),
        "result_lag_ms": describe(lags),
        "windows_lag_over_interval": sum(1 for x in lags if x > interval_s * 1000),
        "windows_without_result": sum(1 for w in windows if w["surface_label"] is None),
        "system": {"process_cpu_pct": describe([r["process_cpu_pct"] for r in system]),
                   "system_cpu_pct": describe([r["system_cpu_pct"] for r in system]),
                   "rss_mb": describe([r["rss_mb"] for r in system]),
                   "temp_c": describe([r["temp_c"] for r in system]),
                   "temp_c_first": system[0]["temp_c"] if system else None,
                   "temp_c_last": system[-1]["temp_c"] if system else None,
                   "min_cpu_freq_mhz": describe([r["min_cpu_freq_mhz"] for r in system])},
    }
    p95_busy, p95_lag = summary["window_busy_ms"]["p95"], summary["result_lag_ms"]["p95"]
    summary["fits_interval"] = bool(p95_busy is not None and p95_busy < interval_s * 1000
                                    and p95_lag is not None and p95_lag < interval_s * 1000)
    return summary


# ---------------------------------------------------------------- PC 결과와 비교

def compare_with_pc(recording, policy, interval_s, start_s, windows, model_dir):
    """PC replay(four_method_replay/results.json)와 같은 구간의 판정을 비교한다. 비교할 수 없으면 None."""
    method = PC_METHOD.get((Path(model_dir).name, policy))
    path = recording / "four_method_replay" / "results.json"
    if method is None or not path.exists() or start_s % interval_s:
        return None
    run = next((r for r in json.loads(path.read_text())["runs"]
                if r["method"] == method and r["interval_s"] == interval_s), None)
    if run is None:
        return None
    offset = int(start_s // interval_s)
    pc = {w["index"]: w for w in run["windows"]}
    both = same_frame = same_label = same_label_same_frame = 0
    for w in windows:
        ref = pc.get(w["index"] + offset)
        if ref is None or ref["prediction"] is None or w["surface_label"] is None:
            continue
        both += 1
        frame_match = ref["frame"]["file"] == w["file"]
        label_match = ref["prediction"]["surface_label"] == w["surface_label"]
        same_frame += frame_match
        same_label += label_match
        same_label_same_frame += frame_match and label_match
    return {"pc_method": method, "compared_windows": both, "same_frame": same_frame, "same_label": same_label,
            "same_label_when_same_frame": same_label_same_frame}


# ---------------------------------------------------------------- 출력

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=0):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(report):
    env = report["environment"]
    lines = [f"# Pi replay 측정: {report['recording']}", "",
             f"- 측정 시각: {report['started_utc']}, 기기: {env['hostname']} ({env['machine']}, CPU {env['cpu_count']}개)",
             f"- 실행: {env['backend']} (torch 스레드 {env['torch_threads']}), 모델 `{env['model_dir']}`,"
             f" 영역 {env['surface_region'] or '기본'}",
             f"- 녹화 {report['start_s']}초부터 간격마다 {report['duration_s']}초, 카메라 속도"
             f" {report['source_fps'] or '녹화 그대로'}, 간격 사이 쉬는 시간 {report['cooldown_s']}초", ""]
    lines += ["## 처리 간격별 요약", "",
              "| 방식 | 간격(s) | 구간 | 분석/구간 | 받은 사진 | 처리 못 함 (%) | 읽지 못함 | 분석 p95(ms) | 구간 처리 p95(ms) |"
              " 결과 지연 p95(ms) | 프로세스 CPU 평균/p95(%) | 시스템 CPU 평균(%) | RSS 최대(MB) | 온도 시작→최고(°C) | 감당 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in report["runs"]:
        f, sysu = s["frames"], s["system"]
        lines.append(
            f"| {s['policy']} | {s['interval_s']:g} | {s['windows']} | {f['per_window']['inferred']:.2f} | {f['delivered']} |"
            f" {f['unprocessed']} ({fmt(f['unprocessed_pct'], 1)}) | {f['dropped_unread']} |"
            f" {fmt(s['inference_wall_ms']['p95'])} | {fmt(s['window_busy_ms']['p95'])} | {fmt(s['result_lag_ms']['p95'])} |"
            f" {fmt(sysu['process_cpu_pct']['mean'])}/{fmt(sysu['process_cpu_pct']['p95'])} |"
            f" {fmt(sysu['system_cpu_pct']['mean'])} | {fmt(sysu['rss_mb']['max'])} |"
            f" {fmt(sysu['temp_c_first'], 1)}→{fmt(sysu['temp_c']['max'], 1)} | {'예' if s['fits_interval'] else '아니오'} |")
    lines += ["", "## 단계별 시간 (ms, 평균 / p95)", "",
              "| 방식 | 간격(s) | 전처리 | Area | Material | 후처리 | 합계 | 품질 검사(호출당) | JPEG 디코딩 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in report["runs"]:
        st = s["stage_ms"]
        cell = lambda d: f"{fmt(d['mean'], 1)} / {fmt(d['p95'], 1)}"
        lines.append(f"| {s['policy']} | {s['interval_s']:g} | " + " | ".join(cell(st[k]) for k in STAGES) +
                     f" | {cell(s['quality_check_ms_per_call'])} | {cell(s['decode_ms'])} |")
    pcs = [(s, s.get("pc_comparison")) for s in report["runs"] if s.get("pc_comparison")]
    if pcs:
        lines += ["", "## PC replay와 비교", "",
                  "| 방식 | 간격(s) | PC 방법 | 비교 구간 | 같은 사진 선택 | 같은 판정 | 같은 사진일 때 같은 판정 |",
                  "|---|---|---|---|---|---|---|"]
        for s, c in pcs:
            lines.append(f"| {s['policy']} | {s['interval_s']:g} | {c['pc_method']} | {c['compared_windows']} |"
                         f" {c['same_frame']} | {c['same_label']} | {c['same_label_when_same_frame']} |")
    lines += ["", "## 용어", "",
              "- 받은 사진: 가짜 카메라가 구간 안에 내보낸 사진 수. 녹화 속도(약 5장/초) 기준",
              "- 처리 못 함: 받은 사진 - 분석한 사진. 품질 검사로 거른 사진과 선택되지 않은 사진, 읽지 못한 사진을 모두 포함",
              "- 읽지 못함: 분석하는 동안 새 사진에 밀려 버려진 사진",
              "- 구간 처리: 한 구간에서 분석에 쓴 시간의 합. 간격보다 작아야 밀리지 않는다",
              "- 결과 지연: 구간이 끝난 뒤 그 구간 결과가 나오기까지 걸린 시간",
              "- 프로세스 CPU: 코어 하나를 100%로 본 값. 시스템 CPU: 전체 코어를 100%로 본 값",
              "- 감당: 구간 처리 p95와 결과 지연 p95가 모두 간격보다 작으면 예",
              "- 실제 수집에서 함께 도는 카메라 인코딩, IMU(BLE), GPS, 업로드 부하는 포함하지 않았다"]
    return "\n".join(lines) + "\n"


def environment(args):
    def version(module):
        try:
            return __import__(module).__version__
        except Exception:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    return {"hostname": platform.node(), "machine": platform.machine(), "system": platform.platform(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(), "numpy": version("numpy"),
            "torch": version("torch") if args.backend == "torch" else None,
            "onnxruntime": version("onnxruntime") if args.backend == "onnxruntime" else None,
            "opencv": version("cv2"), "backend": args.backend,
            "torch_threads": args.torch_threads if args.backend == "torch" else None,
            "model_dir": inference.CONFIG["model_dir"], "surface_region": inference.CONFIG["surface_region"],
            "git_commit": commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="record_replay.py 녹화 폴더")
    parser.add_argument("--intervals", type=float, nargs="+", default=[1, 2, 4, 8], help="처리 간격(초)")
    parser.add_argument("--policy", choices=POLICIES, nargs="+", default=["live"])
    parser.add_argument("--start-s", type=float, default=0.0, help="녹화에서 시작할 위치(초)")
    parser.add_argument("--duration-s", type=float, default=300.0, help="간격마다 측정할 시간(초). 0이면 녹화 끝까지")
    parser.add_argument("--source-fps", type=float, default=None,
                        help="카메라 속도. 비우면 녹화 간격 그대로(약 5장/초), 30이면 라이브 카메라 부하 흉내")
    parser.add_argument("--cooldown-s", type=float, default=60.0, help="간격 사이에 식히는 시간(초)")
    parser.add_argument("--warmup", type=int, default=3, help="측정 전에 버리는 분석 횟수")
    parser.add_argument("--sample-s", type=float, default=1.0, help="CPU·메모리 기록 주기(초)")
    parser.add_argument("--backend", choices=("torch", "onnxruntime"), default=inference.CONFIG["backend"])
    parser.add_argument("--torch-threads", type=int, default=inference.CONFIG["torch_threads"])
    parser.add_argument("--model-dir", default=inference.CONFIG["model_dir"])
    parser.add_argument("--surface-region", choices=("wheel_ahead",), default=inference.CONFIG["surface_region"])
    parser.add_argument("--output", type=Path, help="결과 폴더 (기본: 녹화/pi_benchmark/<UTC 시각>)")
    args = parser.parse_args()

    recording = args.recording.resolve()
    session = json.loads((recording / "session.json").read_text())
    rows = read_rows(recording / "frames.csv")
    available = float(rows[-1]["t_s"]) - args.start_s
    duration_s = available if args.duration_s <= 0 else min(args.duration_s, available)
    if any(i <= 0 for i in args.intervals) or duration_s < max(args.intervals):
        parser.error(f"간격은 양수이고 측정 시간({duration_s:.0f}초)보다 짧아야 합니다.")
    started = utc_now()
    output = (args.output or recording / "pi_benchmark" / started.replace(":", "-")).resolve()
    if output.exists():
        parser.error(f"결과 폴더가 이미 있습니다: {output}")

    inference.CONFIG.update(backend=args.backend, torch_threads=args.torch_threads,
                            model_dir=args.model_dir, surface_region=args.surface_region)
    import cv2
    cv2.setNumThreads(1)          # compare_replay와 같게
    models = inference.load_models()
    gyro_rms = gyro_lookup(recording) if "quality" in args.policy else None
    meter, sampler = Meter(), SystemSampler(args.sample_s)
    meter.patch()
    sampler.start()
    all_windows, all_inferences, runs = [], [], []
    plan = [(p, i) for p in args.policy for i in args.intervals]
    try:
        for n, (policy, interval_s) in enumerate(plan):
            if n and args.cooldown_s > 0:
                print(f"식히는 중 {args.cooldown_s:.0f}초 (온도 {fmt(_max_temp_c(), 1)}°C)", flush=True)
                time.sleep(args.cooldown_s)
            print(f"[{n + 1}/{len(plan)}] {policy}, 간격 {interval_s:g}초, {duration_s:.0f}초 측정", flush=True)
            windows, inferences, _, summary = measure(
                recording, rows, models, meter, sampler, policy=policy, interval_s=interval_s,
                start_s=args.start_s, duration_s=duration_s, source_fps=args.source_fps,
                warmup=args.warmup, gyro_rms=gyro_rms)
            summary["pc_comparison"] = compare_with_pc(recording, policy, interval_s, args.start_s, windows,
                                                       args.model_dir)
            f = summary["frames"]
            print(f"  분석 p95 {fmt(summary['inference_wall_ms']['p95'])}ms, 처리 못 함 {f['unprocessed']}/{f['delivered']},"
                  f" 결과 지연 p95 {fmt(summary['result_lag_ms']['p95'])}ms,"
                  f" CPU {fmt(summary['system']['process_cpu_pct']['mean'])}%,"
                  f" RSS 최대 {fmt(summary['system']['rss_mb']['max'])}MB, 감당 {'예' if summary['fits_interval'] else '아니오'}",
                  flush=True)
            all_windows += windows
            all_inferences += inferences
            runs.append(summary)
    except KeyboardInterrupt:
        print("중단했습니다. 끝난 측정까지 저장합니다.", file=sys.stderr)
    finally:
        sampler.stop()
        meter.unpatch()
        inference.close_models(models)

    report = {"recording": recording.name, "recording_duration_s": session.get("duration_s"),
              "recording_target_fps": session.get("target_fps"), "started_utc": started, "finished_utc": utc_now(),
              "start_s": args.start_s, "duration_s": round(duration_s, 1), "source_fps": args.source_fps,
              "cooldown_s": args.cooldown_s, "warmup": args.warmup, "environment": environment(args), "runs": runs}
    output.mkdir(parents=True)
    for w in all_windows:
        for key in ("start_monotonic", "end_monotonic", "done_monotonic"):
            w[key] = round(w[key], 3)
    write_csv(output / "windows.csv", all_windows)
    write_csv(output / "inferences.csv", all_inferences)
    write_csv(output / "system.csv", sampler.rows)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
