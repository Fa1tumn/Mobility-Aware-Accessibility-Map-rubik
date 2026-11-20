"""Capture one CSI camera frame to a JPEG, then release the camera and exit."""
import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import vision


def save_snapshot(output, camera_id=0, timeout=15.0, warmup=3.0):
    import cv2

    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    vision.CONFIG["camera_id"] = camera_id
    camera = None
    temporary = None
    try:
        camera = vision.open_camera()
        deadline = time.monotonic() + timeout + warmup
        ready_at = None
        print(f"Waiting for camera; discarding {warmup:g} seconds of startup frames...",
              flush=True)
        while time.monotonic() < deadline:
            frame = vision.capture_frame(camera)
            if frame is not None:
                # Start warmup when images arrive, not when GStreamer launches.
                # Drain the stream so the saved image is a fresh, later frame.
                if ready_at is None:
                    ready_at = frame["frame_monotonic"] + warmup
                if frame["frame_monotonic"] < ready_at:
                    continue
                success, jpeg = cv2.imencode(".jpg", frame["image"])
                if not success:
                    raise RuntimeError("Could not encode camera frame as JPEG.")
                # Replace only after the entire JPEG has been written.
                with tempfile.NamedTemporaryFile(
                    dir=output.parent, prefix=".camera-snapshot-", suffix=".jpg",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(jpeg.tobytes())
                os.replace(temporary, output)
                temporary = None
                height, width = frame["image"].shape[:2]
                brightness = float(cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY).mean())
                print(f"Saved {output} ({width}x{height}); mean brightness={brightness:.1f}/255",
                      flush=True)
                if brightness < 5:
                    print("Warning: image is nearly black. Check lens obstruction, lighting, "
                          "and camera selection; try --warmup 8.", file=sys.stderr)
                return output
            if camera["error"]:
                raise RuntimeError(camera["error"])
        raise TimeoutError(f"No frame ready after warmup within {timeout + warmup:g} seconds.")
    finally:
        try:
            vision.close_camera(camera)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="camera_snapshot.jpg",
                        help="JPEG output path; overwritten on each run")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Frame wait budget in seconds, excluding warmup (default: 15)")
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="Discard startup frames for this many seconds (default: 3)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if not math.isfinite(args.warmup) or args.warmup < 0:
        parser.error("--warmup must be a finite nonnegative number")
    try:
        save_snapshot(args.output, args.camera_id, args.timeout, args.warmup)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Snapshot failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
