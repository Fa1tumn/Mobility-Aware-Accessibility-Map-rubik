"""Compare timed and quality-selected frames in 2-second windows with both campus ONNX models.

Usage: python compare_replay.py recordings/campus_<timestamp> [--output DIR]
Requires the adapter dependencies and Matplotlib. Output paths in results.json
are relative to its directory; source frame paths are relative to the recording.
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import camera_inference as camera
import surface_model as surface


MODELS = (surface.MODEL_DIR, surface.MODEL_DIR.parent / "finetuned-v2-onnx")
METHODS = {
    "A": ("Baseline", "Timed"),
    "B": ("Fine-tuned v2", "Timed"),
    "C": ("Baseline", "Quality"),
    "D": ("Fine-tuned v2", "Quality"),
}
INTERVALS = (2,)


def read_rows(recording, name):
    with (recording / name).open(newline="") as source:
        return sorted(csv.DictReader(source), key=lambda row: float(row["t_s"]))


def match_gps(t, fixes, times):
    i = bisect_left(times, t)
    nearby = fixes[max(i - 1, 0):i + 1]
    if not nearby:
        return None
    fix = min(nearby, key=lambda row: abs(float(row["t_s"]) - t))
    if abs(float(fix["t_s"]) - t) > 1 or float(fix["accuracy_m"]) > 10:
        return None
    return {key: float(fix[key]) for key in ("t_s", "latitude", "longitude", "accuracy_m")}


def frame_metadata(recording):
    imu = read_rows(recording, "imu.csv")
    imu_times = np.array([float(r["t_s"]) for r in imu])
    angular_speed_sq = np.array([sum(float(r[k]) ** 2 for k in ("gx", "gy", "gz")) for r in imu])
    fixes = read_rows(recording, "gps.csv")
    fix_times = [float(r["t_s"]) for r in fixes]
    frames = []
    for row in read_rows(recording, "frames.csv"):
        t = float(row["t_s"])
        image = cv2.imread(str(recording / row["file"]))
        if image is None:
            raise ValueError(f"Unreadable recording frame: {row['file']}")
        lo, hi = imu_times.searchsorted([t - 0.2, t], side="right")
        frames.append(dict(file=row["file"], t_s=t, frame_time=row["frame_time"],
                           quality=camera.frame_problem(image) or "ok",
                           road_sharpness=camera.road_sharpness(image),
                           gyro_rms_rad_s=float(np.sqrt(angular_speed_sq[lo:hi].mean())) if hi > lo else None,
                           gps=match_gps(t, fixes, fix_times)))
    return frames


def ranks(values):
    """Midranks in increasing order; all candidates share the same denominator."""
    values = np.array(values)
    return (values[:, None] > values).sum(axis=1) + 0.5 * (values[:, None] == values).sum(axis=1)


def quality_pick(group):
    valid = [f for f in group if f["quality"] == "ok" and f["gyro_rms_rad_s"] is not None]
    if not valid:
        return None
    scores = ranks([f["road_sharpness"] for f in valid]) + ranks([-f["gyro_rms_rad_s"] for f in valid])
    best = max(range(len(valid)), key=lambda i: (scores[i], valid[i]["road_sharpness"], valid[i]["t_s"]))
    return valid[best]


def build_predictor(recording):
    specs = [surface.load_spec(d / "model-spec.json") for d in MODELS]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    def model_path(i, head):
        return MODELS[i] / specs[i]["models"][head]["path"]

    def load(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    # Both bundles ship the same area weights. Verify this before sharing logits.
    hashes = [hashlib.sha256(model_path(i, "area").read_bytes()).digest() for i in (0, 1)]
    if hashes[0] != hashes[1] or any(specs[0][k] != specs[1][k] for k in ("normalization", "input_shape", "color_order")):
        raise ValueError("Expected the campus models to share the same preprocessing and area weights.")
    area = load(model_path(0, "area"))
    materials = [load(model_path(i, "material")) for i in (0, 1)]

    @lru_cache(maxsize=None)
    def predict(file):
        tensor = surface.preprocess(cv2.imread(str(recording / file)), specs[0])
        area_logits = area.run(None, {area.get_inputs()[0].name: tensor})[0]
        results = []
        for material, spec in zip(materials, specs):
            logits = material.run(None, {material.get_inputs()[0].name: tensor})[0]
            result = surface.apply_policy(area_logits, logits, spec)
            results.append({k: result[k] for k in ("surface_label", "surface_confidence", "area_accept_ratio",
                                                   "material_accept_ratio", "model_version", "policy_version")})
        return results

    return predict


def replay(recording, dest, session):
    frames = frame_metadata(recording)
    times = [f["t_s"] for f in frames]
    predict = build_predictor(recording)
    runs = []
    for seconds in INTERVALS:
        count = int(session["duration_s"] // seconds)
        runs_by_method = {m: dict(interval_s=seconds, method=m, model=model, selection=selection, windows=[])
                          for m, (model, selection) in METHODS.items()}
        previous = dict.fromkeys(METHODS)
        for index in range(count):
            start = index * seconds
            end = start + seconds
            group = frames[bisect_left(times, start):bisect_left(times, end)]
            picked = quality_pick(group)
            for method, run in runs_by_method.items():
                window = dict(index=index, start_s=start, end_s=end, candidates=len(group),
                              frame=None, prediction=None, change=None, skip_reason=None)
                run["windows"].append(window)
                if picked is None:
                    window["skip_reason"] = "no_eligible_frame" if group else "empty_window"
                    continue
                frame = group[-1] if run["selection"] == "Timed" else picked
                result = predict(frame["file"])[int(method in ("B", "D"))]
                label = result["surface_label"]
                window.update(frame=frame, prediction=result)
                if previous[method] is not None and label != previous[method]:
                    relative = Path("changes") / f"{seconds}s" / method / (
                        f"w{index:04d}_{frame['t_s']:.3f}s_{previous[method]}_to_{label}.jpg")
                    (dest / relative.parent).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(recording / frame["file"], dest / relative)
                    window["change"] = dict(previous_label=previous[method], new_label=label, image=relative.as_posix())
                previous[method] = label
            if (index + 1) % 100 == 0:
                print(f"{seconds}s: {index + 1}/{count} windows", flush=True)
        for run in runs_by_method.values():
            labels = Counter(w["prediction"]["surface_label"] for w in run["windows"] if w["prediction"])
            run["summary"] = dict(windows=count, predictions=sum(labels.values()), skipped=count-sum(labels.values()),
                                  label_counts=dict(labels), changes=sum(w["change"] is not None for w in run["windows"]),
                                  gps_matched=sum(bool(w["frame"] and w["frame"]["gps"]) for w in run["windows"]))
        runs.extend(runs_by_method.values())
        print(f"{seconds}s: {count} windows complete", flush=True)
    return dict(recording=recording.name, duration_s=session["duration_s"], start_utc=session["start_utc"],
                candidate_frames=len(frames), unique_frames_inferred_per_model=predict.cache_info().currsize,
                models=[d.name for d in MODELS], window_policy="Complete half-open windows; skip all methods if no quality candidate.",
                selection_policy="Timed: latest in window. Quality: pass exposure/contrast checks and have gyro data; "
                                 "equal midranks for higher road sharpness and lower preceding 0.2 s gyro RMS; "
                                 "break ties by sharpness, then latest frame.",
                transition_policy="Compare successive inferred labels within each method and interval. "
                                  "Include unknown. Skip missing windows. First prediction establishes the baseline.",
                runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, help="New output directory (default: RECORDING/four_method_replay)")
    args = parser.parse_args()
    recording = args.recording.resolve()
    output = (args.output or recording / "four_method_replay").resolve()
    if output.exists():
        parser.error(f"Output already exists. Use --output with a new directory: {output}")
    session = json.loads((recording / "session.json").read_text())
    if session["mode"] != "route" or session["status"] not in ("finished", "operator_stop"):
        parser.error("A completed route recording is required.")
    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".replay-") as working:
        dest = Path(working) / "results"
        dest.mkdir()
        report = replay(recording, dest, session)
        (dest / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        from plot_replay import render
        render(report, recording, dest)
        dest.rename(output)
    print(output)


if __name__ == "__main__":
    main()
