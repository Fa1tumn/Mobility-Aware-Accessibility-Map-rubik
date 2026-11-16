"""Camera -> both surface models -> fixed-duration observation windows.

Shared by main.py and the standalone JSONL CLI. Window membership uses frame
monotonic decode time, never inference completion time. Only the latest camera
frame is processed, so a slow CPU drops frames instead of queuing old images.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import sys
import time

import inference
import vision


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


def iter_windows(camera, models, window_s=2.0, timing=False):
    """Yield [start, end) windows, including windows with no successful images.

    An inference already in progress finishes in its original capture window.
    Therefore delivery can lag a boundary by one inference (or consumer/upload
    time), while membership never shifts. Ctrl+C flushes completed results in
    the partial window; a second Ctrl+C can still abort cleanup.
    """
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    origin = time.monotonic()
    utc_origin = datetime.now(timezone.utc)
    index = 0
    results = []
    frame_count = 0
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
            last_frame = stamp
            frame_count += 1
            try:
                result = inference.analyze_image(frame["image"], models, timing)
                if result is not None:
                    results.append(result)
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
