"""Measure the on-device CPU cost of the Area+Material pipeline.

Runs the ONNX Runtime CPU backend repeatedly on one or more images and reports
the preprocess / Area / Material / postprocess split plus end-to-end latency,
so you can judge whether the 2 s collection target is met on the Rubik Pi and
whether sustained running stays stable (a proxy for thermal/throughput drift).

    python benchmark_cpu.py --image photo.jpg --iters 60
    python benchmark_cpu.py --dir /path/to/photos --iters 120

This measures compute only (no camera/GPS/IMU). Run it on the Pi for the real
numbers; the figures on a laptop only indicate relative cost.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

import surface_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_images(args):
    paths = []
    if args.dir:
        paths.extend(sorted(p for p in Path(args.dir).iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES))
    paths.extend(Path(p) for p in args.image)
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        images.append((path.name, image))
    return images


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the Area+Material CPU pipeline.")
    parser.add_argument("--image", action="append", default=[], help="Image file (repeatable).")
    parser.add_argument("--dir", help="Directory of images to cycle through.")
    parser.add_argument("--iters", type=int, default=60, help="Number of inferences.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup inferences, excluded from stats.")
    parser.add_argument("--threads", type=int, default=None, help="ORT intra-op threads (default: library default).")
    args = parser.parse_args()

    images = load_images(args)
    if not images:
        print("No readable images. Pass --image FILE or --dir DIR.")
        return 1

    providers = None
    if args.threads is not None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        # Rebuild backends with custom thread settings.
        spec = surface_model.load_spec()
        model_dir = surface_model.MODEL_DIR
        area = surface_model.OnnxRuntimeBackend(model_dir / spec["models"]["area"]["path"])
        material = surface_model.OnnxRuntimeBackend(model_dir / spec["models"]["material"]["path"])
        area.session = ort.InferenceSession(str(model_dir / spec["models"]["area"]["path"]),
                                             sess_options=options, providers=["CPUExecutionProvider"])
        material.session = ort.InferenceSession(str(model_dir / spec["models"]["material"]["path"]),
                                                sess_options=options, providers=["CPUExecutionProvider"])
        model = surface_model.SurfaceModel(area, material, spec)
    else:
        model = surface_model.SurfaceModel.from_onnxruntime(providers=providers)

    print(f"model_version: {surface_model.model_version(model.spec)}")
    print(f"backend: onnxruntime CPU, images: {len(images)}, iters: {args.iters}, warmup: {args.warmup}")

    splits = {"preprocess": [], "area": [], "material": [], "postprocess": [], "total": []}
    wall = []
    try:
        for i in range(args.warmup + args.iters):
            name, image = images[i % len(images)]
            start = time.perf_counter()
            result = model.analyze(image, timing=True)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i < args.warmup:
                continue
            for key in splits:
                splits[key].append(result["timing_ms"][key])
            wall.append(elapsed_ms)
    finally:
        model.close()

    print("\nPer-stage latency (ms), averaged over measured iterations:")
    for key in ("preprocess", "area", "material", "postprocess", "total"):
        vals = splits[key]
        print(f"  {key:<12} avg {statistics.mean(vals):7.1f}  "
              f"p50 {percentile(vals, .5):7.1f}  p95 {percentile(vals, .95):7.1f}  max {max(vals):7.1f}")

    target_ms = 2000
    avg_total = statistics.mean(splits["total"])
    p95_total = percentile(splits["total"], .95)
    print(f"\nEnd-to-end wall time: avg {statistics.mean(wall):.1f} ms, p95 {percentile(wall, .95):.1f} ms")
    print(f"2 s collection target: {'MET' if p95_total < target_ms else 'AT RISK'} "
          f"(p95 total {p95_total:.0f} ms vs {target_ms} ms)")

    # Drift check: compare the first and last fifths of measured totals.
    fifth = max(1, len(splits["total"]) // 5)
    early = statistics.mean(splits["total"][:fifth])
    late = statistics.mean(splits["total"][-fifth:])
    drift = (late - early) / early * 100 if early else 0.0
    print(f"Sustained-run drift (total): first {early:.0f} ms -> last {late:.0f} ms ({drift:+.1f}%)")
    print("A large positive drift under sustained load can indicate thermal throttling; "
          "confirm on the Pi with its thermal/memory counters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
