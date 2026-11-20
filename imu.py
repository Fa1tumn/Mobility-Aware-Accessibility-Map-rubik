import argparse
import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import time

import numpy as np

CONFIG = {
    "device_address": "C8:2C:72:8F:2B:2C",
    "notify_uuid": "0000ffe4-0000-1000-8000-00805f9a34fb",
    "write_uuid": "0000ffe9-0000-1000-8000-00805f9a34fb",
    "sample_rate_hz": 100,
    "mounting_calibrated": True,
    "window_s": 2.0,
    "slope_filter_tau_s": 0.25,  # Continuous EMA; ~0.75 s to settle 95% of a step.
}

CALIBRATION_PATH = Path(__file__).with_name("imu_calibration.json")
_level_gravity = None
_attitude_level_gravity = None
SLOPE_METHOD = "fused_attitude_ema_mean_v1"


def level_basis(gravity):
    """Keep sensor x as forward, projected onto the calibrated level plane."""
    up = np.asarray(gravity, dtype=float)
    if up.shape != (3,) or not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-6:
        raise ValueError("Invalid level gravity vector.")
    up = up / np.linalg.norm(up)
    forward = np.array([1.0, 0.0, 0.0]) - up[0] * up
    if np.linalg.norm(forward) < 0.1:
        raise ValueError("Sensor x axis must point forward, not vertically.")
    forward /= np.linalg.norm(forward)
    return np.array([forward, np.cross(up, forward), up])


def load_calibration():
    global _level_gravity, _attitude_level_gravity
    _level_gravity = None
    _attitude_level_gravity = None
    if not CALIBRATION_PATH.exists():
        return
    data = json.loads(CALIBRATION_PATH.read_text())
    if data.get("device_address") != CONFIG["device_address"]:
        raise ValueError("Saved IMU zero belongs to another device; run imu.py --zero again.")
    gravity = data["level_gravity"]
    level_basis(gravity)
    attitude_gravity = data.get("attitude_level_gravity")
    if attitude_gravity is not None:
        level_basis(attitude_gravity)
    _level_gravity = gravity
    _attitude_level_gravity = attitude_gravity
    print(f"Loaded IMU level zero from {CALIBRATION_PATH.name}.")
    if attitude_gravity is None:
        print("Using legacy acceleration mounting zero; run imu.py --zero on level ground to set fused-attitude zero.")


def save_level_zero(samples):
    """Save an explicit stationary, level reference; never auto-zero on startup."""
    global _level_gravity, _attitude_level_gravity
    if len(samples) < 160:
        raise ValueError("Need at least 160 recent samples to set zero.")
    acceleration = np.array([[s[k] for k in ("ax", "ay", "az")] for s in samples])
    gyro = np.array([[s[k] for k in ("gx", "gy", "gz")] for s in samples])
    if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(gyro)):
        raise ValueError("Invalid IMU samples; zero was not saved.")
    gravity = acceleration.mean(axis=0)
    if (np.max(acceleration.std(axis=0)) > 0.15
            or np.max(np.linalg.norm(gyro, axis=1)) > 0.05
            or not 7 < np.linalg.norm(gravity) < 12):
        raise ValueError("Keep the wheelchair stationary on level ground, then retry --zero.")
    level_basis(gravity)
    directions = [attitude_up(sample) for sample in samples]
    if any(direction is None for direction in directions):
        raise ValueError("Fused Roll/Pitch missing; zero was not saved.")
    directions = np.array(directions)
    attitude_gravity = directions.mean(axis=0)
    if np.max(np.linalg.norm(directions - attitude_gravity, axis=1)) > 0.02:
        raise ValueError("Fused attitude is not stable; keep still and retry --zero.")
    level_basis(attitude_gravity)
    data = {"device_address": CONFIG["device_address"], "level_gravity": gravity.tolist(),
            "attitude_level_gravity": attitude_gravity.tolist()}
    temporary = CALIBRATION_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(CALIBRATION_PATH)
    _level_gravity = gravity.tolist()
    _attitude_level_gravity = attitude_gravity.tolist()


def start_imu(load_zero=True, record_session=False) -> dict:
    tau = CONFIG["slope_filter_tau_s"]
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("slope_filter_tau_s must be finite and nonnegative.")
    if load_zero:
        load_calibration()
    imu_state = {
        "samples": deque(maxlen=2000),
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
        "clock_offset_s": None,
        "last_sample_time_s": None,
        "byte_buffer": bytearray(),
        "received_samples": 0,
        "error": None,
        "slope_filter": None,
        "timing_resets": 0,
        # Only the standalone monitor opts in. main.py never writes a report.
        "session_samples": [] if record_session else None,
    }
    if CONFIG["device_address"] is None or CONFIG["notify_uuid"] is None:
        print("IMU Bluetooth is not configured; IMU values will be NULL.")
        return imu_state
    thread = threading.Thread(target=_read_bluetooth, args=(imu_state,), daemon=True)
    imu_state["thread"] = thread
    thread.start()
    return imu_state


def decode_packet(data, timestamp_s) -> dict:
    if len(data) != 20 or data[:2] != b"\x55\x61":
        raise ValueError("Expected a 20-byte WT901BLE packet starting with 55 61.")
    values = struct.unpack("<9h", data[2:])
    acceleration_scale = 16 * 9.80665 / 32768
    gyro_scale = math.radians(2000) / 32768
    return {
        "timestamp_s": timestamp_s,
        "ax": values[0] * acceleration_scale,
        "ay": values[1] * acceleration_scale,
        "az": values[2] * acceleration_scale,
        "gx": values[3] * gyro_scale,
        "gy": values[4] * gyro_scale,
        "gz": values[5] * gyro_scale,
        "roll_deg": values[6] * 180 / 32768,
        "pitch_deg": values[7] * 180 / 32768,
        "yaw_deg": values[8] * 180 / 32768,
    }


def receive_data(imu_state, data):
    buffer = imu_state["byte_buffer"]
    buffer.extend(data)
    packets = []
    while len(buffer) >= 2:
        if buffer[0] != 0x55 or buffer[1] not in (0x61, 0x71):
            del buffer[0]
            continue
        if len(buffer) < 20:
            break
        packet = bytes(buffer[:20])
        del buffer[:20]
        if packet[1] == 0x61:
            packets.append(packet)
    if not packets:
        return

    sample_interval = 1 / CONFIG["sample_rate_hz"]
    timestamp_s = imu_state["last_sample_time_s"]
    if timestamp_s is None:
        timestamp_s = 0.0
    last_timestamp_s = timestamp_s + len(packets) * sample_interval
    receive_time = time.monotonic()
    with imu_state["lock"]:
        clock_offset = imu_state["clock_offset_s"]
        # Default packets have no device timestamp; reset the window after a large timing error.
        if clock_offset is None or abs(clock_offset + last_timestamp_s - receive_time) > 0.25:
            imu_state["samples"].clear()
            imu_state["slope_filter"] = None
            if clock_offset is not None:
                imu_state["timing_resets"] += 1
            imu_state["clock_offset_s"] = receive_time - last_timestamp_s
        else:
            # Follow slow clock drift without moving consecutive samples backwards.
            timing_error = receive_time - (clock_offset + last_timestamp_s)
            max_adjustment = sample_interval / 2
            adjustment = max(-max_adjustment, min(max_adjustment, timing_error * 0.1))
            imu_state["clock_offset_s"] += adjustment
    for index, packet in enumerate(packets):
        sample_time_s = timestamp_s + (index + 1) * sample_interval
        payload = decode_packet(packet, sample_time_s)
        if not add_sample(imu_state, payload):
            raise ValueError("Invalid or out-of-order IMU sample.")


def add_sample(imu_state: dict, payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = ("timestamp_s", "ax", "ay", "az", "gx", "gy", "gz")
    keys += tuple(key for key in ("roll_deg", "pitch_deg", "yaw_deg") if key in payload)
    sample = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            value = float(value)
        except OverflowError:
            return False
        if not math.isfinite(value):
            return False
        sample[key] = value
    if sample["timestamp_s"] < 0:
        return False
    receive_time = time.monotonic()
    with imu_state["lock"]:
        previous_time = imu_state["last_sample_time_s"]
        if previous_time is not None and sample["timestamp_s"] <= previous_time:
            return False
        if imu_state["clock_offset_s"] is None:
            imu_state["clock_offset_s"] = receive_time - sample["timestamp_s"]
        sample["sample_monotonic"] = sample["timestamp_s"] + imu_state["clock_offset_s"]
        _filter_attitude(imu_state, sample)
        imu_state["samples"].append(sample)
        if imu_state.get("session_samples") is not None:
            imu_state["session_samples"].append(sample.copy())
        imu_state["last_sample_time_s"] = sample["timestamp_s"]
        imu_state["received_samples"] += 1
        imu_state["error"] = None
    return True


def attitude_up(sample):
    """Earth up in sensor axes for the device's right-handed Z-Y-X Euler angles.

    R = Rz(yaw) Ry(pitch) Rx(roll); R.T @ [0,0,1] does not depend on yaw.
    Sensor x pointing uphill corresponds to negative device pitch.
    """
    try:
        roll, pitch = (sample[key] for key in ("roll_deg", "pitch_deg"))
        if any(isinstance(v, bool) or not math.isfinite(v) or abs(v) > 180
               for v in (roll, pitch)):
            return None
        roll, pitch = math.radians(roll), math.radians(pitch)
    except (KeyError, TypeError, ValueError):
        return None
    return np.array([-math.sin(pitch), math.sin(roll) * math.cos(pitch),
                     math.cos(roll) * math.cos(pitch)])


def attitude_slope_deg(sample):
    if not CONFIG["mounting_calibrated"]:
        return None
    up = attitude_up(sample)
    if up is None:
        return None
    reference = _attitude_level_gravity if _attitude_level_gravity is not None else _level_gravity
    if reference is not None:
        up = level_basis(reference) @ up
    denominator = math.hypot(up[1], up[2])
    if denominator < 1e-6:
        return None
    return math.degrees(math.atan2(up[0], denominator))


def _filter_attitude(imu_state, sample):
    """Filter once on receipt, across window boundaries, resetting after gaps."""
    angle = attitude_slope_deg(sample)
    sample["attitude_slope_deg"] = angle
    sample["smoothed_slope_deg"] = None
    if angle is None:
        imu_state["slope_filter"] = None
        return
    now = sample["sample_monotonic"]
    previous = imu_state.get("slope_filter")
    tau = CONFIG["slope_filter_tau_s"]
    if previous is not None and 0 < now - previous[0] <= 0.1 and tau > 0:
        alpha = -math.expm1(-(now - previous[0]) / tau)
        angle = previous[1] + alpha * (angle - previous[1])
    imu_state["slope_filter"] = (now, angle)
    sample["smoothed_slope_deg"] = angle


def _read_bluetooth(imu_state: dict) -> None:
    try:
        asyncio.run(_receive_bluetooth(imu_state))
    except Exception as error:
        message = str(error) or type(error).__name__
        with imu_state["lock"]:
            imu_state["error"] = message
        print(f"IMU Bluetooth error: {message}")


async def _receive_bluetooth(imu_state: dict) -> None:
    from bleak import BleakClient

    def on_data(sender, data):
        try:
            receive_data(imu_state, data)
        except ValueError as error:
            with imu_state["lock"]:
                changed = imu_state["error"] != str(error)
                imu_state["error"] = str(error)
            if changed:
                print(f"IMU packet error: {error}")

    rate_codes = {10: 0x06, 20: 0x07, 50: 0x08, 100: 0x09, 200: 0x0B}
    if CONFIG["sample_rate_hz"] not in rate_codes:
        raise ValueError("Set sample_rate_hz to 10, 20, 50, 100, or 200.")
    client = BleakClient(CONFIG["device_address"], timeout=10)
    try:
        print(f"Connecting to IMU {CONFIG['device_address']}...")
        await asyncio.wait_for(client.connect(), timeout=10)
        commands = [
            bytes.fromhex("FF AA 69 88 B5"),
            bytes([0xFF, 0xAA, 0x03, rate_codes[CONFIG["sample_rate_hz"]], 0x00]),
            bytes.fromhex("FF AA 1F 03 00"),
        ]
        for command in commands:
            if imu_state["stop_event"].is_set():
                return
            await asyncio.wait_for(
                client.write_gatt_char(CONFIG["write_uuid"], command, response=True), timeout=3
            )
            await asyncio.sleep(0.2)
        if imu_state["stop_event"].is_set():
            return
        await asyncio.wait_for(client.start_notify(CONFIG["notify_uuid"], on_data), timeout=3)
        print(f"IMU connected: output={CONFIG['sample_rate_hz']} Hz, bandwidth=42 Hz.")
        while not imu_state["stop_event"].is_set():
            if not client.is_connected:
                raise RuntimeError("IMU Bluetooth connection was lost.")
            await asyncio.sleep(0.05)
    finally:
        await asyncio.wait_for(client.disconnect(), timeout=2)


def get_window(imu_state: dict, frame_monotonic: float) -> list:
    start_time = frame_monotonic - CONFIG["window_s"]
    with imu_state["lock"]:
        if imu_state["error"] is not None:
            return []
        samples = []
        for sample in imu_state["samples"]:
            if start_time <= sample["sample_monotonic"] <= frame_monotonic:
                samples.append(sample.copy())
    if not samples or frame_monotonic - samples[-1]["sample_monotonic"] > 0.15:
        return []
    return samples


def analyze_samples(samples: list) -> dict:
    result = {"slope_percent": None, "vibration_frequency_avg_hz": None}
    if len(samples) < 32:
        return result
    timestamps = np.array([sample["sample_monotonic"] for sample in samples])
    intervals = np.diff(timestamps)
    duration = timestamps[-1] - timestamps[0]
    if duration < CONFIG["window_s"] * 0.8 or np.any(intervals <= 0):
        return result
    if intervals.max() > 0.1 or intervals.max() > 2.5 * np.median(intervals):
        return result
    result["slope_percent"] = calculate_slope(samples)
    result["vibration_frequency_avg_hz"] = calculate_vibration_frequency(samples, timestamps)
    return result


def snapshot_window(samples):
    """Capture the inputs and calibration used for later offline diagnosis."""
    return {
        "schema_version": 1,
        "slope_method": SLOPE_METHOD,
        "device_address": CONFIG["device_address"],
        "sample_rate_hz": CONFIG["sample_rate_hz"],
        "window_s": CONFIG["window_s"],
        "mounting_calibrated": CONFIG["mounting_calibrated"],
        "level_gravity": list(_level_gravity) if _level_gravity is not None else None,
        "attitude_level_gravity": list(_attitude_level_gravity) if _attitude_level_gravity is not None else None,
        "slope_filter_tau_s": CONFIG["slope_filter_tau_s"],
        "samples": [dict(sample) for sample in samples],
    }


def calculate_slope(samples: list):
    """Convert the window mean of continuously smoothed fused tilt to grade.

    Missing attitude never silently falls back to the accelerometer estimator.
    """
    if not samples or not CONFIG["mounting_calibrated"]:
        return None
    angles = [sample.get("smoothed_slope_deg") for sample in samples]
    if any(angle is None or not math.isfinite(angle) or abs(angle) >= 90 for angle in angles):
        return None
    return 100 * math.tan(math.radians(float(np.mean(angles))))


def calculate_vibration_frequency(samples: list, timestamps):
    if len(samples) < 2 or len(timestamps) != len(samples):
        return None
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None
    sample_rate = (len(samples) - 1) / duration
    if sample_rate < 80:
        return None
    magnitudes = []
    for sample in samples:
        magnitude = math.sqrt(sample["ax"] ** 2 + sample["ay"] ** 2 + sample["az"] ** 2)
        magnitudes.append(magnitude)
    uniform_times = np.linspace(timestamps[0], timestamps[-1], len(samples))
    signal = np.interp(uniform_times, timestamps, magnitudes)
    signal -= signal.mean()
    if np.sqrt(np.mean(signal ** 2)) < 0.02:
        return None
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    total_power = 0.0
    weighted_frequency = 0.0
    for frequency, strength in zip(frequencies, power):
        if 1 <= frequency <= 40:
            total_power += strength
            weighted_frequency += frequency * strength
    if total_power <= 1e-12:
        return None
    return float(weighted_frequency / total_power)


def stop_imu(imu_state: dict) -> None:
    if imu_state is None:
        return
    imu_state["stop_event"].set()
    thread = imu_state["thread"]
    if thread is not None and thread.ident is not None:
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("IMU Bluetooth thread did not stop.")


def write_session_report(path, state, windows, metadata, started_at, duration_s, termination):
    """Write the standalone monitor's full session, including its partial tail."""
    with state["lock"]:
        samples = list(state["session_samples"] or [])
        timing_resets = state["timing_resets"]
        last_error = state["error"]
    slopes = [row["slope_percent"] for row in windows if row["slope_percent"] is not None]
    report = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "termination": termination,
        "metadata": metadata,
        "summary": {
            "sample_count": len(samples), "window_count": len(windows),
            "valid_slope_windows": len(slopes), "timing_resets": timing_resets,
            "last_imu_error": last_error,
            "slope_min_percent": min(slopes) if slopes else None,
            "slope_max_percent": max(slopes) if slopes else None,
            "slope_mean_percent": float(np.mean(slopes)) if slopes else None,
        },
        # Monitor polls every second; these two-second windows overlap.
        "windows": windows,
        "samples": samples,
    }
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".imu-report-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"IMU report saved: {path} ({len(samples)} samples, {len(windows)} windows)")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Monitor IMU or save its level mounting reference.")
    parser.add_argument("--zero", action="store_true",
                        help="Keep the wheelchair stationary on level ground and save this pose as zero.")
    parser.add_argument("--report", help="Monitor JSON output path; default: imu_report_<UTC timestamp>.json")
    args = parser.parse_args(argv)
    if args.zero:
        print("Setting zero: wheelchair must be stationary on level ground; keep IMU fixed with x forward.")
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    report_path = args.report or f"imu_report_{started.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    state = None
    windows = []
    metadata = None
    termination = "completed"
    exit_code = 0
    try:
        state = start_imu(load_zero=not args.zero, record_session=not args.zero)
        metadata = snapshot_window([])
        metadata.pop("samples")
        metadata["session_start_monotonic"] = started_monotonic
        print(f"Slope estimator: {SLOPE_METHOD}; smoothing tau={CONFIG['slope_filter_tau_s']:g}s.")
        if not args.zero:
            print(f"Press Ctrl+C to stop and save all received IMU samples to {report_path}")
        zero_deadline = time.monotonic() + 30
        previous_count = 0
        previous_time = time.monotonic()
        while True:
            time.sleep(1)
            now = time.monotonic()
            samples = get_window(state, now)
            with state["lock"]:
                count = state["received_samples"]
                error = state["error"]
            if error is not None:
                raise RuntimeError(error)
            if args.zero and now >= zero_deadline:
                raise RuntimeError("Timed out waiting for enough recent samples; zero was not saved.")
            rate = (count - previous_count) / (now - previous_time)
            previous_count = count
            previous_time = now
            if not samples:
                print(f"Waiting for recent IMU samples. received_rate_hz={rate:.1f}")
            if args.zero:
                if len(samples) >= 160:
                    save_level_zero(samples)
                    print(f"IMU zero saved to {CALIBRATION_PATH}. Future runs load it automatically.")
                    break
                continue
            latest = samples[-1] if samples else {}
            output = {"received_rate_hz": round(rate, 1), "window_samples": len(samples),
                      "window_start_monotonic": now - CONFIG["window_s"],
                      "window_end_monotonic": now,
                      "window_end_utc": datetime.now(timezone.utc).isoformat()}
            for key in ("ax", "ay", "az", "gx", "gy", "gz", "roll_deg", "pitch_deg", "yaw_deg"):
                value = latest.get(key)
                output[key] = round(value, 4) if value is not None else None
            output.update(analyze_samples(samples))
            windows.append(output)
            print(json.dumps(output))
    except KeyboardInterrupt:
        termination = "interrupted"
        print("Stopping IMU.")
    except Exception as error:
        termination = f"error: {error}"
        exit_code = 1
        print(f"IMU monitor failed: {error}")
    finally:
        try:
            stop_imu(state)
        except Exception as error:
            termination += f"; cleanup error: {error}"
            exit_code = 1
            print(f"IMU cleanup failed: {error}")
        if state is not None and not args.zero:
            try:
                write_session_report(report_path, state, windows, metadata, started.isoformat(),
                                     time.monotonic() - started_monotonic, termination)
            except Exception as error:
                exit_code = 1
                print(f"Could not save IMU report: {error}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
