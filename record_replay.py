"""Record GPS, IMU and camera streams on the Rubik Pi, or check a saved session."""
import argparse
import csv
import json
import math
import queue
import signal
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path


def csv_output(stack, directory, name, columns):
    handle = stack.enter_context((directory / name).open("x", newline="", buffering=1))
    writer = csv.writer(handle)
    writer.writerow(columns.split())
    return writer


def read_marks(marks):
    for line in sys.stdin:
        if line.strip():
            marks.put((time.monotonic(), line.strip()))


def wait_for_inputs(camera, sensor, receiver):
    import vision

    deadline = time.monotonic() + 60
    previous_missing = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        missing = []
        if camera is not None and vision.capture_frame(camera) is None:
            if camera["error"]:
                raise RuntimeError(camera["error"])
            missing.append("camera")
        with sensor["lock"]:
            if sensor["error"]:
                raise RuntimeError(sensor["error"])
            if not sensor["samples"] or now - sensor["samples"][-1]["sample_monotonic"] > 1:
                missing.append("IMU")
        with receiver["lock"]:
            gps = receiver["latest_gps"]
            if gps is None or now - gps["gps_monotonic"] > 3:
                missing.append("phone GPS")
        if not missing:
            return
        if missing != previous_missing:
            print("Waiting for: " + ", ".join(missing), flush=True)
            previous_missing = missing
        time.sleep(0.2)
    raise RuntimeError("No fresh input within 60 s: " + ", ".join(previous_missing or []))


def record(args):
    import imu
    import phone
    import vision

    folder = args.directory
    if folder.exists():
        raise FileExistsError(f"Choose a new session directory: {folder}")
    with ExitStack() as stack:
        gps = phone.start_phone()
        stack.callback(phone.stop_phone, gps)
        sensor = imu.start_imu(load_zero=False, record_session=True)
        stack.callback(imu.stop_imu, sensor)
        camera = None
        if args.mode == "route":
            camera = vision.open_camera()
            stack.callback(vision.close_camera, camera)
        print("Waiting for live inputs (up to 60 s). Start the phone GPS sender.", flush=True)
        wait_for_inputs(camera, sensor, gps)
        folder.mkdir(parents=True, exist_ok=False)
        if camera is not None:
            (folder / "frames").mkdir()

        imu_keys = "timestamp_s ax ay az gx gy gz roll_deg pitch_deg yaw_deg".split()
        imu_csv = csv_output(stack, folder, "imu.csv", "t_s " + " ".join(imu_keys))
        gps_csv = csv_output(stack, folder, "gps.csv", "t_s gps_time latitude longitude accuracy_m")
        marks_csv = csv_output(stack, folder, "marks.csv", "t_s label")
        if camera is not None:
            frames_csv = csv_output(stack, folder, "frames.csv", "t_s frame_time file")
        origin = time.monotonic()
        meta = dict(mode=args.mode, start_utc=datetime.now(timezone.utc).isoformat(),
                    start_monotonic=origin, target_fps=args.fps if camera else 0,
                    imu_hz=imu.CONFIG["sample_rate_hz"], status="recording")
        manifest = folder / "session.json"
        manifest.write_text(json.dumps(meta, indent=2))
        marks = queue.SimpleQueue()
        threading.Thread(target=read_marks, args=(marks,), daemon=True).start()
        print("RECORDING. Type a label and press Enter. Ctrl+C finishes the run.", flush=True)
        next_frame = next_report = origin
        deadline = origin + args.seconds if args.seconds else math.inf
        n_frames = n_imu = n_gps = 0
        latest = {}

        def drain(end):
            nonlocal n_imu, n_gps
            with sensor["lock"]:
                samples = sensor["session_samples"]
                sensor["session_samples"] = []
            for row in samples:
                stamp = row["sample_monotonic"]
                if origin <= stamp <= end:
                    imu_csv.writerow([stamp - origin, *(row.get(k) for k in imu_keys)])
                    n_imu += 1
                    latest["IMU"] = stamp
            with gps["lock"]:
                fixes = list(gps["gps_history"])
                gps["gps_history"].clear()
            for fix in fixes:
                stamp = fix["gps_monotonic"]
                if origin <= stamp <= end:
                    gps_csv.writerow([stamp - origin, fix["gps_time"], fix["latitude"],
                                      fix["longitude"], fix["accuracy_m"]])
                    n_gps += 1
                    latest["GPS"] = stamp
            while not marks.empty():
                stamp, label = marks.get()
                if origin <= stamp <= end:
                    marks_csv.writerow([stamp - origin, label])
                    print(f"Mark {stamp-origin:.2f}s  {label}", flush=True)

        status = "failed"
        # Finish the current file/CSV row before handling an operator stop.
        stop = threading.Event()
        previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
        stack.callback(signal.signal, signal.SIGINT, previous_handler)
        try:
            while not stop.is_set() and time.monotonic() < deadline:
                if sensor["error"]:
                    raise RuntimeError(sensor["error"])
                if camera is not None and time.monotonic() >= next_frame:
                    frame = vision.capture_frame(camera)
                    if camera["error"]:
                        raise RuntimeError(camera["error"])
                    if frame is not None and origin <= frame["frame_monotonic"] < deadline:
                        path = f"frames/{n_frames:07d}.jpg"
                        (folder / path).write_bytes(frame["jpeg_bytes"])
                        stamp = frame["frame_monotonic"]
                        frames_csv.writerow([stamp - origin, frame["frame_time"], path])
                        latest["camera"] = stamp
                        n_frames += 1
                    # Skip missed slots rather than recording a burst to catch up.
                    period = 1 / args.fps
                    next_frame += max(1, math.floor((time.monotonic() - next_frame) / period) + 1) * period
                drain(deadline)
                now = time.monotonic()
                if now >= next_report:
                    ages = " ".join(f"{key} age={now-latest[key]:.1f}s" if key in latest
                                    else f"{key} waiting" for key in
                                    (("camera", "IMU", "GPS") if camera else ("IMU", "GPS")))
                    print(f"{now-origin:.1f}s  frames={n_frames} imu={n_imu} gps={n_gps}  {ages}", flush=True)
                    next_report = now + 5
                time.sleep(0.01)
            status = "operator_stop" if stop.is_set() else "finished"
        finally:
            end = min(time.monotonic(), deadline)
            drain(end)
            meta.update(status=status, duration_s=end-origin, imu_timing_resets=sensor["timing_resets"])
            manifest.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"Saved to {folder}: frames={n_frames} imu={n_imu} gps={n_gps}")


def check(folder):
    meta = json.loads((folder / "session.json").read_text())
    duration = meta.get("duration_s")
    print(f"{meta['mode']}  status={meta['status']}  duration_s={duration}")
    ok = meta["status"] in ("operator_stop", "finished") and bool(duration)
    names = ["imu", "gps", "marks"]
    if meta["mode"] == "route":
        names.append("frames")
    for name in names:
        times = []
        bad_files = 0
        with (folder / f"{name}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                times.append(float(row["t_s"]))
                if name == "frames":
                    path = folder / row["file"]
                    if not path.is_file() or path.stat().st_size < 4:
                        bad_files += 1
                        continue
                    with path.open("rb") as jpg:
                        start = jpg.read(2)
                        jpg.seek(-2, 2)
                        bad_files += (start != b"\xff\xd8" or jpg.read(2) != b"\xff\xd9")
        if name == "marks":
            print(f"marks: {len(times)} (review labels in marks.csv)")
            continue
        ordered = sorted(times)
        gaps = [b-a for a, b in zip([0.0] + ordered, ordered + [duration or (max(times, default=0))])]
        rate = len(times) / duration if duration else 0
        print(f"{name}: {len(times)}  average={rate:.2f}/s  longest gap={max(gaps):.2f}s")
        if bad_files:
            print(f"Missing or incomplete JPEG files: {bad_files}")
        ok = ok and bool(times) and not bad_files
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record")
    capture.add_argument("directory", nargs="?", type=Path,
                         help="new session folder (default: recordings/campus_<UTC timestamp>)")
    capture.add_argument("--mode", choices=("route", "motion"), default="route",
                         help="motion saves IMU, GPS and labels only")
    capture.add_argument("--fps", type=int, choices=range(1, 31), default=5)
    capture.add_argument("--seconds", type=int, help="duration after all inputs are ready")
    inspect = commands.add_parser("check")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        return check(args.directory)
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.directory is None:
        args.directory = Path("recordings") / datetime.now(timezone.utc).strftime("campus_%Y%m%dT%H%M%SZ")
    print(f"Session directory: {args.directory}", flush=True)
    record(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
