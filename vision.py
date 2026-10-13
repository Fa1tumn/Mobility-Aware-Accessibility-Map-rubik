import os
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone

CONFIG = {
    "camera_id": 0,
}


def open_camera():
    import cv2
    import numpy as np

    command = [
        "gst-launch-1.0", "-q", "-e",
        "qtiqmmfsrc", f"camera={CONFIG['camera_id']}", "!",
        "video/x-raw,format=NV12,width=1280,height=720", "!",
        "videorate", "!", "video/x-raw,framerate=30/1", "!",
        "queue", "max-size-buffers=2", "leaky=downstream", "!",
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
    try:
        while not camera["stop_event"].is_set():
            chunk = os.read(camera["process"].stdout.fileno(), 65536)
            if not chunk:
                if not camera["stop_event"].is_set():
                    camera["error"] = "GStreamer image stream ended. Check stderr for details."
                break
            buffer.extend(chunk)

            while not camera["stop_event"].is_set():
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    # Preserve a trailing FF byte if the JPEG start marker is split.
                    buffer[:] = b"\xff" if buffer[-1:] == b"\xff" else b""
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    del buffer[:start]
                    if len(buffer) > 16 * 1024 * 1024:
                        buffer.clear()
                        print("JPEG buffer is too large; discarded the incomplete frame.")
                    break

                jpeg_bytes = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                image = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR,
                )
                if image is None:
                    print("Failed to decode a JPEG frame; skipping it.")
                    continue

                # Timestamp decoding, not exposure or the main loop queue read.
                frame = {
                    "image": image,
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
        if not camera["stop_event"].is_set():
            camera["error"] = f"Camera reader failed: {error}"


def capture_frame(camera):
    """Return the latest BGR frame, or None on error or a one-second timeout."""
    if camera["stop_event"].is_set():
        return None
    if camera["error"] is not None:
        print(camera["error"])
        return None
    try:
        return camera["frames"].get(timeout=1.0)
    except queue.Empty:
        print(camera["error"] or "No camera frame available. Check GStreamer output.")
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
        except ProcessLookupError:
            process.wait(timeout=2)

    thread = camera["thread"]
    if thread is not None and thread.ident is not None:
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("Camera reader thread did not stop.")
    if process.stdout is not None:
        process.stdout.close()
