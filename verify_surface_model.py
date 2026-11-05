"""Verify the area-surface-v3 adapter on reference photos using ONNX Runtime CPU.

Runs the Area + Material pipeline on a set of images and prints the combined
observation for each. Use this to confirm the adapter reproduces expected
results on a PC before wiring the NPU backend.

    python verify_surface_model.py IMG_0088.jpg IMG_0089.jpg ...
    python verify_surface_model.py --dir /path/to/source   # all images in dir
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

import surface_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(args) -> list:
    paths = []
    if args.dir:
        directory = Path(args.dir)
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
    paths.extend(Path(p) for p in args.images)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the surface model adapter on CPU.")
    parser.add_argument("images", nargs="*", help="Image files to run.")
    parser.add_argument("--dir", help="Directory of images to run.")
    args = parser.parse_args()

    paths = collect_images(args)
    if not paths:
        print("No images given. Pass image paths or --dir DIR.")
        return 1

    model = surface_model.SurfaceModel.from_onnxruntime()
    print(f"model_version: {surface_model.model_version(model.spec)}")
    print(f"policy_version: {surface_model.policy_version(model.spec)}")
    print(f"images: {len(paths)}\n")

    latencies = []
    failures = 0
    try:
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"{path.name}: FAILED to read image")
                failures += 1
                continue
            start = time.perf_counter()
            result = model.analyze(image)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            summary = {
                "surface_label": result["surface_label"],
                "surface_confidence": round(result["surface_confidence"], 4)
                if result["surface_confidence"] is not None else None,
                "area_accept_ratio": round(result["area_accept_ratio"], 4),
                "material_accept_ratio": round(result["material_accept_ratio"], 4),
            }
            print(f"{path.name}: {json.dumps(summary)}  ({elapsed_ms:.0f} ms)")
    finally:
        model.close()

    if latencies:
        average = sum(latencies) / len(latencies)
        print(f"\nCPU latency per frame (Area+Material): "
              f"avg {average:.0f} ms, min {min(latencies):.0f} ms, max {max(latencies):.0f} ms")
    if failures:
        print(f"{failures} image(s) failed to load.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
