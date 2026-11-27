import json
import math
import os
import queue
import signal
import sys
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = {
    "camera_id": 0,
}


CAMERA_CONFIG_PATH = Path(__file__).resolve().with_name("camera_config.json")


def load_camera_config():
    """Read the shared capture settings once at collection startup."""
    try:
        settings = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read {CAMERA_CONFIG_PATH}: {error}") from error

    types = {"exposure_metering": str, "max_exposure_us": int, "exposure_compensation": int,
             "save_sharp_frames": bool, "save_rejected_frames": bool, "sharp_frame_dir": str}
    if not isinstance(settings, dict):
        raise ValueError(f"{CAMERA_CONFIG_PATH} must contain a JSON object.")
    for key, expected in types.items():
        if type(settings.get(key)) is not expected:
            raise ValueError(f"{CAMERA_CONFIG_PATH}: {key} must be a {expected.__name__}.")
    if not settings["sharp_frame_dir"].strip():
        raise ValueError(f"{CAMERA_CONFIG_PATH}: sharp_frame_dir must be a nonempty path.")
    return settings


def camera_source_options(settings=None):
    """Build qtiqmmfsrc properties for center-weighted auto exposure with an optional shutter cap."""
    if settings is None:
        settings = load_camera_config()
    camera_options = [f"camera={CONFIG['camera_id']}"]
    exposure_compensation = os.environ.get("CAMERA_EXPOSURE_COMPENSATION") or settings["exposure_compensation"]
    if exposure_compensation:
        error_message = "CAMERA_EXPOSURE_COMPENSATION must be an integer from -12 to 12."
        try:
            exposure_steps = int(exposure_compensation)
        except ValueError as error:
            raise ValueError(error_message) from error
        if not -12 <= exposure_steps <= 12:
            raise ValueError(error_message)
        camera_options.append(f"exposure-compensation={exposure_steps}")

    metering = os.environ.get("CAMERA_EXPOSURE_METERING") or settings["exposure_metering"]
    if metering not in ("average", "center-weighted"):
        raise ValueError("CAMERA_EXPOSURE_METERING must be average or center-weighted.")
    camera_options.extend([f"exposure-metering={metering}", "exposure-mode=auto"])

    # Qualcomm's three-point AE table increases shutter time first, then gain.
    # Each exposure index step represents a factor of 1.03 in time * gain.
    message = "CAMERA_MAX_EXPOSURE_US must be 0 (factory AE) or 11 to 33333 microseconds."
    try:
        max_us = int(os.environ.get("CAMERA_MAX_EXPOSURE_US", settings["max_exposure_us"]))
    except ValueError as error:
        raise ValueError(message) from error
    if max_us:
        if not 11 <= max_us <= 33333:
            raise ValueError(message)
        max_ns = max_us * 1000
        time_index = math.log(max_ns / 10040, 1.03)
        gain_index = time_index + math.log(32, 1.03)
        camera_options.append(
            "custom-exposure-table=org.codeaurora.qcamera3.exposuretable,"
            "isValid=(boolean)true,sensitivityCorrectionFactor=(double)1.0,kneeCount=(int)3,"
            "gainKneeEntries=(double)<1.0,1.0,32.0>,"
            f"expTimeKneeEntries=(int)<10040,{max_ns},{max_ns}>,"
            "incrementPriorityKneeEntries=(int)<1,1,1>,"
            f"expIndexKneeEntries=(double)<0.0,{time_index:.4f},{gain_index:.4f}>,"
            "thresAntiBandingMinExpTimePct=(double)0.97;"
        )
    return camera_options


def open_camera(settings=None, *, on_jpeg=None):
    """Pass on_jpeg to save every complete JPEG."""
    camera_options = camera_source_options(settings)
    rate_and_queue = ["queue", "max-size-buffers=2", "!"]
    cv2 = np = None
    if on_jpeg is None:
        import cv2
        import numpy as np
        rate_and_queue = [
            "videorate", "!", "video/x-raw,framerate=30/1", "!",
            "queue", "max-size-buffers=2", "leaky=downstream", "!",
        ]
    command = [
        "gst-launch-1.0", "-q", "-e",
        "qtiqmmfsrc", *camera_options, "!",
        "video/x-raw,format=NV12,width=1280,height=720", "!",
        *rate_and_queue,
        "jpegenc", "quality=80", "!", "fdsink", "fd=1", "sync=false",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,  # Keep stderr out of the JPEG byte stream.
        bufsize=0,
        start_new_session=True,
    )

    camera = {
        "process": process,
        "frames": queue.Queue(maxsize=1),
        "stop_event": threading.Event(),
        "thread": None,
        "error": None,
        "on_jpeg": on_jpeg,
    }
    try:
        camera["thread"] = threading.Thread(
            target=_read_frames, args=(camera, cv2, np), daemon=True,
        )
        camera["thread"].start()
    except BaseException:
        close_camera(camera)
        raise
    return camera


def _read_frames(camera, cv2, np) -> None:
    buffer = bytearray()
    on_jpeg = camera.get("on_jpeg")
    try:
        # Read the remaining JPEGs on shutdown.
        while on_jpeg is not None or not camera["stop_event"].is_set():
            chunk = os.read(camera["process"].stdout.fileno(), 65536)
            if not chunk:
                if on_jpeg is not None and buffer:
                    camera["error"] = "GStreamer ended with an incomplete JPEG."
                elif not camera["stop_event"].is_set():
                    camera["error"] = "GStreamer image stream ended. Check stderr for details."
                break
            buffer.extend(chunk)

            while on_jpeg is not None or not camera["stop_event"].is_set():
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    # Preserve a trailing FF byte if the JPEG start marker is split.
                    buffer[:] = b"\xff" if buffer[-1:] == b"\xff" else b""
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    del buffer[:start]
                    if len(buffer) > 16 * 1024 * 1024:
                        if on_jpeg is not None:
                            raise ValueError("Incomplete JPEG exceeded 16 MiB.")
                        buffer.clear()
                        print("JPEG buffer is too large; discarded the incomplete frame.", file=sys.stderr)
                    break

                jpeg_bytes = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                if on_jpeg is not None:
                    on_jpeg(jpeg_bytes, time.monotonic(), datetime.now(timezone.utc).isoformat())
                    continue
                image = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR,
                )
                if image is None:
                    print("Failed to decode a JPEG frame; skipping it.", file=sys.stderr)
                    continue

                # Timestamp decoding, not exposure or the main loop queue read.
                frame = {
                    "image": image,
                    "jpeg_bytes": jpeg_bytes,
                    "frame_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "frame_monotonic": time.monotonic(),
                }

                # Discard the queued frame to prevent latency buildup.
                try:
                    camera["frames"].put_nowait(frame)
                except queue.Full:
                    try:
                        camera["frames"].get_nowait()
                    except queue.Empty:
                        pass
                    camera["frames"].put_nowait(frame)
    except Exception as error:
        if on_jpeg is not None or not camera["stop_event"].is_set():
            camera["error"] = f"Camera reader failed: {error}"


def capture_frame(camera):
    """Return the latest BGR frame, or None on error or a one-second timeout.

    The reader thread keeps only the newest frame (queue maxsize=1 with a
    leaky put), so this already returns the most recent decoded frame rather
    than a backlog. The frame carries its own capture timestamp, so a record
    always reflects when the frame was grabbed, not when inference finished.
    """
    if camera["stop_event"].is_set():
        return None
    if camera["error"] is not None:
        print(camera["error"], file=sys.stderr)
        return None
    try:
        return camera["frames"].get(timeout=1.0)
    except queue.Empty:
        print(camera["error"] or "No camera frame available. Check GStreamer output.", file=sys.stderr)
        return None


def close_camera(camera) -> None:
    if camera is None:
        return
    camera["stop_event"].set()
    process = camera["process"]
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
            if camera.get("on_jpeg") is not None and not camera["error"]:
                camera["error"] = "GStreamer timed out flushing the archive."
        except ProcessLookupError:
            process.wait(timeout=2)

    thread = camera["thread"]
    if thread is not None and thread.ident is not None:
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("Camera reader thread did not stop.")
    if process.stdout is not None:
        process.stdout.close()
