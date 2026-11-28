"""Run the fine-tuned v3 ONNX model on the sharpest normally exposed frame per 2 s."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import onnxruntime as ort

import camera_inference as camera
from inference import LABELS as MATERIAL_LABELS
import surface_model as surface


MODEL_DIR = surface.MODEL_DIR.parent / "finetuned-v3-onnx"
WINDOW_S = 2
LABELS = (*MATERIAL_LABELS, "unknown")
PREDICTION_FIELDS = (
    "surface_label", "surface_confidence", "area_accept_ratio", "material_accept_ratio",
    "model_version", "policy_version",
)
ROW_FIELDS = (
    "window_start_s", "window_end_s", "candidates", "normal_exposure", "unreadable",
    "overexposed", "underexposed", "status", "source_frame", "selected_frame", "t_s",
    "frame_time", "road_sharpness", "brightness_mean", "brightness_std", "saturated_fraction",
    *PREDICTION_FIELDS,
)
SUMMARY_FIELDS = (
    "name", "session", "frames", "windows", "predicted", "skipped", "unreadable",
    "overexposed", "underexposed", "model_version", "policy_version",
    *LABELS, "result_csv",
)
_model = None


def initialize_worker(region):
    global _model
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    _model = surface.SurfaceModel.from_onnxruntime(MODEL_DIR, session_options=options)
    if region:
        _model.spec["policy"]["region"] = region


def read_windows(recording):
    windows = defaultdict(list)
    with (recording / "frames.csv").open(newline="") as handle:
        for frame in csv.DictReader(handle):
            frame["t_s"] = float(frame["t_s"])
            if frame["t_s"] < 0:
                raise ValueError(f"Negative frame time in {recording}: {frame['file']}")
            windows[int(frame["t_s"] // WINDOW_S)].append(frame)
    return windows


def analyze_window(recording, target, index, frames):
    summary = dict(window_start_s=index * WINDOW_S, window_end_s=(index + 1) * WINDOW_S,
                   candidates=len(frames), normal_exposure=0, unreadable=0,
                   overexposed=0, underexposed=0)
    winner = None
    for frame in frames:
        image = cv2.imread(str(recording / frame["file"]))
        if image is None:
            summary["unreadable"] += 1
            continue
        stats = camera.frame_stats(image)
        exposure = camera.frame_problem(image, stats)
        if exposure in ("overexposed", "underexposed"):
            summary[exposure] += 1
            continue
        summary["normal_exposure"] += 1
        score = (camera.road_sharpness(image), frame["t_s"], frame["file"])
        if winner is None or score > winner[0]:
            winner = (score, image, frame, stats)
    if winner is None:
        summary["status"] = "no_normal_exposure" if frames else "empty_window"
        if frames and summary["unreadable"] == len(frames):
            summary["status"] = "unreadable_window"
        return summary

    score, image, frame, stats = winner
    source = Path(frame["file"])
    selected = Path("selected_frames") / source.name
    result = _model.analyze(image)
    shutil.copyfile(recording / source, target / selected)
    summary.update(status="predicted", source_frame=source.as_posix(),
                   selected_frame=selected.as_posix(), t_s=frame["t_s"], frame_time=frame["frame_time"],
                   road_sharpness=score[0], brightness_mean=stats["mean"], brightness_std=stats["std"],
                   saturated_fraction=stats["saturated"])
    summary.update({field: result[field] for field in PREDICTION_FIELDS})
    return summary


def reanalyze_selection(previous, target):
    with previous.open(newline="") as handle:
        for prior in csv.DictReader(handle):
            row = {field: prior[field] for field in ROW_FIELDS if field not in PREDICTION_FIELDS}
            if row["status"] == "predicted":
                selected = Path(row["selected_frame"])
                source = previous.parent / selected
                image = cv2.imread(str(source))
                if image is None:
                    raise ValueError(f"Could not read previously selected frame: {source}")
                result = _model.analyze(image)
                shutil.copyfile(source, target / selected)
                row.update({field: result[field] for field in PREDICTION_FIELDS})
                row["previous_surface_label"] = prior["surface_label"]
                row["label_changed"] = int(row["surface_label"] != prior["surface_label"])
            yield row


def replay_session(job):
    name, session, recording, output, previous = job
    target = output / name / session
    target.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    labels = Counter()
    with tempfile.TemporaryDirectory(prefix=".v3-", dir=target.parent) as scratch:
        staging = Path(scratch) / "results"
        (staging / "selected_frames").mkdir(parents=True)
        if previous is None:
            windows = read_windows(recording)
            total = max(windows, default=-1) + 1
            rows = (analyze_window(recording, staging, index, windows.get(index, ()))
                    for index in range(total))
            fields = ROW_FIELDS
        else:
            rows = reanalyze_selection(previous, staging)
            fields = (*ROW_FIELDS, "previous_surface_label", "label_changed")
        count = 0
        with (staging / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for count, row in enumerate(rows, 1):
                writer.writerow(row)
                stats["frames"] += int(row["candidates"])
                stats.update({k: int(row[k]) for k in ("unreadable", "overexposed", "underexposed")})
                if row["status"] == "predicted":
                    labels[row["surface_label"]] += 1
                    if previous is not None:
                        stats["changed"] += row["label_changed"]
                        stats["previous_unknown"] += int(row["previous_surface_label"] == "unknown")
                if count % 100 == 0:
                    print(f"{name}/{session}: {count} windows", flush=True)
        staging.rename(target)

    predicted = sum(labels.values())
    print(f"{name}/{session}: {predicted}/{count} predicted", flush=True)
    return dict(name=name, session=session,
                windows=count, predicted=predicted, skipped=count - predicted, **stats,
                model_version=surface.model_version(_model.spec),
                policy_version=surface.policy_version(_model.spec),
                **{label: labels[label] for label in LABELS},
                result_csv=(target / "predictions.csv").relative_to(output).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, help="directory with <name>/<session>/frames.csv")
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--region", choices=surface.REGIONS, help="override the model's voting region")
    parser.add_argument("--reuse-selected", action="store_true",
                        help="use a previous result directory as input and preserve its frame selections")
    args = parser.parse_args()
    jobs = []
    fields = SUMMARY_FIELDS
    if args.reuse_selected:
        with (args.recordings / "summary.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                previous = args.recordings / row["result_csv"]
                jobs.append((row["name"], row["session"], None, args.output, previous))
        fields = (*SUMMARY_FIELDS, "changed", "previous_unknown")
        jobs.sort(key=lambda job: job[-1].stat().st_size, reverse=True)
    else:
        for manifest in sorted(args.recordings.glob("*/*/session.json")):
            meta = json.loads(manifest.read_text())
            if meta.get("status") != "finished":
                print(f"Skipping {manifest.parent}: {meta.get('status')}", flush=True)
                continue
            recording = manifest.parent
            jobs.append((recording.parent.name, recording.name, recording, args.output, None))
        jobs.sort(key=lambda job: (job[2] / "frames.csv").stat().st_size, reverse=True)
    if not jobs:
        parser.error("No finished frame sessions found.")
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(args.region,)) as pool:
        summaries = list(pool.map(replay_session, jobs))
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: (row["name"], row["session"])))


if __name__ == "__main__":
    main()
