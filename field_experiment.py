"""Shared field recorder for frames_test.py, imu_test.py and coast_test.py."""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from record_replay import csv_output


IMU_FIELDS = ("timestamp_s ax ay az gx gy gz roll_deg pitch_deg yaw_deg "
              "sample_monotonic attitude_slope_deg smoothed_slope_deg").split()
SURFACE_MARKS = ("red_1", "block_1", "red_2", "block_2", "route_end")


def session_name(value):
    if value.strip() in ("", ".", "..") or any(c in value for c in '/\\:\0'):
        raise argparse.ArgumentTypeError("Use a single folder name, for example block.")
    return value


def read_keys(fd, events):
    # Timestamp each mark when Enter is pressed.
    with os.fdopen(fd, "rb", buffering=0, closefd=False) as stream:
        for line in stream:
            events.put((time.monotonic(), line.decode("utf-8", errors="replace").strip()))


class FieldMarks:
    def __init__(self, stack, folder, experiment):
        self.writer = csv_output(stack, folder, "marks.csv", "t_s label")
        self.events = queue.SimpleQueue()
        self.index = 0
        self.release = None
        self.trials = (csv_output(stack, folder, "trials.csv", "trial release_s stop_s coast_s")
                       if experiment == "coast" else None)

    def drain(self, origin):
        while not self.events.empty():
            stamp, note = self.events.get()
            elapsed = stamp - origin
            if note:
                label = note
            elif self.trials is not None:
                trial = self.index // 2 + 1
                if self.release is None:
                    label = f"release_{trial}"
                    self.release = elapsed
                else:
                    label = f"stop_{trial}"
                    self.trials.writerow([trial, self.release, elapsed, elapsed - self.release])
                    self.release = None
                self.index += 1
            else:
                label = (SURFACE_MARKS[self.index] if self.index < len(SURFACE_MARKS)
                         else f"mark_{self.index + 1}")
                self.index += 1
            self.writer.writerow([elapsed, label])
            print(f"MARK {elapsed:.3f}s  {label}", flush=True)


def record(experiment, folder, name, stopping, summarize):
    import imu

    origin = time.monotonic()
    meta = dict(experiment=experiment, name=name, start_monotonic=origin,
                start_utc=datetime.now(timezone.utc).isoformat(),
                imu_hz=imu.CONFIG["sample_rate_hz"], status="starting")
    manifest = folder / "session.json"

    def save_status(status):
        meta.update(status=status, duration_s=time.monotonic() - origin)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(manifest)

    save_status("starting")
    counts = {"imu": 0, "frames": 0}
    ready = False
    try:
        with ExitStack() as stack:
            imu_csv = csv_output(stack, folder, "imu.csv", "t_s " + " ".join(IMU_FIELDS))
            marks = FieldMarks(stack, folder, experiment)
            if experiment == "frames":
                image_dir = folder / "frames"
                image_dir.mkdir()
                frame_csv = csv_output(stack, folder, "frames.csv", "t_s frame_time file")

                def save_frame(jpeg, stamp, utc):
                    filename = f"{counts['frames']:07d}.jpg"
                    (image_dir / filename).write_bytes(jpeg)
                    frame_csv.writerow([stamp - origin, utc, "frames/" + filename])
                    counts["frames"] += 1

            sensor = imu.start_imu(load_zero=False, record_session=True)

            def drain_imu():
                with sensor["lock"]:
                    samples = sensor["session_samples"]
                    sensor["session_samples"] = []
                for sample in samples:
                    imu_csv.writerow([sample["sample_monotonic"] - origin,
                                      *(sample.get(key) for key in IMU_FIELDS)])
                    counts["imu"] += 1

            # Stop sensors before saving the last samples.
            stack.callback(marks.drain, origin)
            stack.callback(drain_imu)
            stack.callback(imu.stop_imu, sensor)
            camera = None
            if experiment == "frames":
                import vision
                camera = vision.open_camera(on_jpeg=save_frame)
                stack.callback(vision.close_camera, camera)

            next_report = origin
            while not stopping.is_set():
                drain_imu()
                if sensor["error"] or (camera and camera["error"]):
                    raise RuntimeError(sensor["error"] or camera["error"])
                now = time.monotonic()
                if not ready and counts["imu"] and (camera is None or counts["frames"]):
                    ready = True
                    meta["ready_s"] = now - origin
                    save_status("recording")
                    threading.Thread(target=read_keys, args=(sys.stdin.fileno(), marks.events),
                                     daemon=True).start()
                    print("RECORDING. Ctrl+C saves and exits.", flush=True)
                    if experiment == "coast":
                        print("Enter at release, Enter at full stop. Repeat for each trial.", flush=True)
                    else:
                        print("Enter: red_1, block_1, red_2, block_2, route_end.", flush=True)
                if not ready and now - origin > 60:
                    raise RuntimeError("Timed out waiting for the first IMU sample or camera frame.")
                marks.drain(origin)
                if now >= next_report:
                    print(f"{now - origin:.1f}s  imu={counts['imu']}  frames={counts['frames']}",
                          flush=True)
                    next_report = now + 5
                stopping.wait(0.02)
        if sensor["error"] or (camera and camera["error"]):
            raise RuntimeError(sensor["error"] or camera["error"])
        meta["imu_timing_resets"] = sensor["timing_resets"]
        if summarize is not None and ready:
            summarize(folder)
        meta.update(counts)
        save_status("finished" if ready else "stopped_before_ready")
    except Exception as error:
        meta.update(counts, error=str(error))
        save_status("failed")
        raise
    print(f"Saved {counts['imu']} IMU samples and {counts['frames']} frames in {folder}", flush=True)


def run(experiment, description, *, summarize=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--name", required=True, type=session_name, help="surface or route folder name")
    parser.add_argument("--output", type=Path, default=Path("recordings"), help="base output directory")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = args.output / experiment / args.name / stamp
    folder.mkdir(parents=True, exist_ok=False)
    print(f"Session folder: {folder.resolve()}", flush=True)
    stopping = threading.Event()
    with ExitStack() as handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, lambda *_: stopping.set())
            handlers.callback(signal.signal, sig, previous)
        record(experiment, folder, args.name, stopping, summarize)
