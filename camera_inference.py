"""Camera -> both surface models -> fixed-duration observation windows.

Shared by main.py and the standalone JSONL CLI. Window membership uses frame
monotonic decode time, never inference completion time. Only the latest camera
frame is processed, so a slow CPU drops frames instead of queuing old images.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import inference
import vision

# 추론하면 안 되는 프레임 기준. 2026-11-20 한낮 주행에서 하얗게 날아간 프레임(전체 255)이
# 창의 42%에 들어갔고, 모델은 이를 soil_stone(0.86)으로 분류했다.
OVEREXPOSED_MEAN = 240      # 평균 밝기(0~255)가 이보다 크면 과다 노출
OVEREXPOSED_SHARE = 0.6     # 밝기 250 이상 픽셀이 이 비율을 넘어도 과다 노출
UNDEREXPOSED_MEAN = 15      # 평균 밝기가 이보다 작으면 노출 부족
FLAT_STD = 6                # 밝기 표준편차가 이보다 작으면 거의 단색(가림, 초점 실패 등)


def utc_string(value):
    return value.isoformat().replace("+00:00", "Z")


def aggregate_results(results):
    """Equal votes per successful frame (including unknown); average ratios.

    Ties use summed confidence then label order. Confidence is the mean of the
    frames voting for the chosen class, not a probability calibrated over time.
    Failed inference contributes no vote; an entirely failed window returns None.
    """
    if not results:
        return None
    counts = Counter(r["surface_label"] for r in results)
    scores = {label: sum(r.get("surface_confidence") or 0.0 for r in results
                         if r["surface_label"] == label) for label in counts}
    winner = min(counts, key=lambda label: (-counts[label], -scores[label], label))
    combined = dict(results[-1])
    combined["surface_label"] = winner
    confidences = [r["surface_confidence"] for r in results
                   if r["surface_label"] == winner and r.get("surface_confidence") is not None]
    combined["surface_confidence"] = sum(confidences) / len(confidences) if confidences else None
    for key in ("area_accept_ratio", "material_accept_ratio"):
        values = [r[key] for r in results if r.get(key) is not None]
        combined[key] = sum(values) / len(values) if values else None
    timings = [r["timing_ms"] for r in results if r.get("timing_ms")]
    if timings:
        combined["timing_ms"] = {
            key: round(sum(t[key] for t in timings) / len(timings), 1)
            for key in timings[0]
        }
    return combined


def frame_stats(image):
    """밝기 평균·표준편차·포화(>=250) 비율과 표본 해시. 배열이 아니면 None.

    8픽셀 간격 표본(1280x720이면 160x90)만 본다. 해시는 같은 화면이 반복되는지 찾는 용도다.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return None
    small = np.ascontiguousarray(image[::8, ::8])
    gray = small.mean(axis=2)
    return {"mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2),
            "saturated": round(float((gray >= 250).mean()), 4),
            "hash": hashlib.sha1(small.tobytes()).hexdigest()[:16]}


def frame_problem(image, stats=None):
    """추론에 쓰면 안 되는 프레임이면 이유를, 괜찮으면 None을 돌려준다.

    기준값은 초기 실험값이다. 밝은 시멘트처럼 정상인 바닥이 걸리는지 표본 저장으로 확인해 조정한다.
    """
    stats = stats if stats is not None else frame_stats(image)
    if stats is None:
        return None
    if stats["mean"] > OVEREXPOSED_MEAN or stats["saturated"] > OVEREXPOSED_SHARE:
        return "overexposed"
    if stats["mean"] < UNDEREXPOSED_MEAN:
        return "underexposed"
    if stats["std"] < FLAT_STD:
        return "flat"
    return None


class FrameSampler:
    """원인 확인용 프레임 표본 저장. 환경변수 FRAME_SAMPLE_DIR가 있을 때만 켜진다.

    - 품질 검사에 걸린 프레임: 창마다 첫 장
    - 정상 프레임: every_s초마다 한 장(모델 결과와 함께)
    JPEG와 같은 이름의 줄을 samples.jsonl에 남긴다(시각, 창 번호, 밝기 수치, 해시, 결과).
    """

    def __init__(self, directory, every_s=30.0, max_files=2000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every_s = every_s
        self.max_files = max_files
        self.saved = 0
        self.last_ok = float("-inf")
        self.rejected_windows = set()

    @classmethod
    def from_env(cls):
        directory = os.environ.get("FRAME_SAMPLE_DIR")
        if not directory:
            return None
        return cls(directory, float(os.environ.get("FRAME_SAMPLE_EVERY_S", "30")))

    def maybe_save(self, frame, window_index, stats, problem, result=None):
        if self.saved >= self.max_files:
            return None
        stamp = frame["frame_monotonic"]
        if problem is not None:
            if window_index in self.rejected_windows:
                return None
            self.rejected_windows.add(window_index)
        elif stamp - self.last_ok < self.every_s:
            return None
        else:
            self.last_ok = stamp
        import cv2
        name = f"{frame.get('frame_time', stamp)}_{problem or 'ok'}".replace(":", "-")
        cv2.imwrite(str(self.directory / f"{name}.jpg"), frame["image"])
        keep = ("surface_label", "surface_confidence", "area_accept_ratio", "material_accept_ratio")
        line = {"file": f"{name}.jpg", "frame_time": frame.get("frame_time"), "window": window_index,
                "problem": problem, **(stats or {}),
                "result": {k: result.get(k) for k in keep} if result else None}
        with open(self.directory / "samples.jsonl", "a", encoding="utf-8") as out:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.saved += 1
        return name


def iter_windows(camera, models, window_s=2.0, timing=False, sampler=None):
    """Yield [start, end) windows, including windows with no successful images.

    An inference already in progress finishes in its original capture window.
    Therefore delivery can lag a boundary by one inference (or consumer/upload
    time), while membership never shifts. Ctrl+C flushes completed results in
    the partial window; a second Ctrl+C can still abort cleanup.
    """
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    if sampler is None:
        sampler = FrameSampler.from_env()
    origin = time.monotonic()
    utc_origin = datetime.now(timezone.utc)
    index = 0
    results = []
    frame_count = 0
    rejected = Counter()          # 추론하지 않고 버린 프레임 수(이유별)
    last_frame = float("-inf")

    def pack(end=None):
        start = origin + index * window_s
        end = origin + (index + 1) * window_s if end is None else end
        return {
            "window_start_time": utc_string(utc_origin + timedelta(seconds=start - origin)),
            "window_end_time": utc_string(utc_origin + timedelta(seconds=end - origin)),
            "start_monotonic": start,
            "end_monotonic": end,
            "frame_count": frame_count,
            "inference_count": len(results),
            "rejected_frames": dict(rejected),
            # 로컬 SQLite에만 저장(업로드하지 않음). 이미지 품질 문제와 모델의 unknown을 구분한다.
            "rejected_count": sum(rejected.values()),
            "rejected_reasons": ",".join(f"{k}:{v}" for k, v in sorted(rejected.items())) or None,
            "result": aggregate_results(results),
        }

    try:
        while True:
            end = origin + (index + 1) * window_s
            if time.monotonic() >= end:
                yield pack()
                index += 1
                results = []
                frame_count = 0
                rejected = Counter()
                continue
            if camera is None or models is None:
                time.sleep(min(0.1, max(0, end - time.monotonic())))
                continue
            read_start = time.monotonic()
            try:
                frame = vision.capture_frame(camera)
            except Exception as error:
                print(f"Camera unavailable: {error}", file=sys.stderr)
                frame = None
            if frame is None:
                # Some failed camera implementations return immediately.
                if time.monotonic() - read_start < 0.05:
                    time.sleep(0.05)
                continue
            stamp = frame["frame_monotonic"]
            if stamp < origin + index * window_s or stamp <= last_frame:
                continue
            # A blocking read can cross a boundary. Flush preceding windows
            # before analyzing this frame, preserving its original timestamp.
            while stamp >= origin + (index + 1) * window_s:
                yield pack()
                index += 1
                results = []
                frame_count = 0
                rejected = Counter()
            last_frame = stamp
            frame_count += 1
            stats = frame_stats(frame["image"])
            problem = frame_problem(frame["image"], stats)
            if problem is not None:
                rejected[problem] += 1
                if sampler is not None:
                    sampler.maybe_save(frame, index, stats, problem)
                continue
            try:
                result = inference.analyze_image(frame["image"], models, timing)
                if result is not None:
                    results.append(result)
                    if sampler is not None:
                        sampler.maybe_save(frame, index, stats, None, result)
            except Exception as error:
                print(f"Image inference unavailable: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        if frame_count:
            yield pack(end=min(time.monotonic(), origin + (index + 1) * window_s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "onnxruntime"), default=inference.CONFIG["backend"])
    parser.add_argument("--camera-id", type=int, default=vision.CONFIG["camera_id"])
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--windows", type=int, default=0, help="Stop after N windows; 0 runs until Ctrl+C")
    parser.add_argument("--output", help="Write JSONL to this file instead of stdout")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.windows < 0 or not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("windows must be nonnegative and window-seconds finite and positive")
    inference.CONFIG["backend"] = args.backend
    vision.CONFIG["camera_id"] = args.camera_id
    models = camera = output = windows = None
    try:
        models = inference.load_models()
        camera = vision.open_camera()
        output = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        windows = iter_windows(camera, models, args.window_seconds, args.timing)
        for count, window in enumerate(windows, 1):
            output.write(json.dumps(window, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if args.windows and count >= args.windows:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if windows is not None:
            windows.close()
        try:
            if camera is not None:
                vision.close_camera(camera)
        finally:
            inference.close_models(models)
            if output is not None and output is not sys.stdout:
                output.close()


if __name__ == "__main__":
    main()
