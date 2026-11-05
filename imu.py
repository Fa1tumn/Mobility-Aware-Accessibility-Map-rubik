import argparse
import asyncio
from collections import deque
import json
import math
from pathlib import Path
import struct
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
}

CALIBRATION_PATH = Path(__file__).with_name("imu_calibration.json")
_level_gravity = None


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
    global _level_gravity
    _level_gravity = None
    if not CALIBRATION_PATH.exists():
        return
    data = json.loads(CALIBRATION_PATH.read_text())
    if data.get("device_address") != CONFIG["device_address"]:
        raise ValueError("Saved IMU zero belongs to another device; run imu.py --zero again.")
    gravity = data["level_gravity"]
    level_basis(gravity)
    _level_gravity = gravity
    print(f"Loaded IMU level zero from {CALIBRATION_PATH.name}.")


def save_level_zero(samples):
    """Save an explicit stationary, level reference; never auto-zero on startup."""
    global _level_gravity
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
    data = {"device_address": CONFIG["device_address"], "level_gravity": gravity.tolist()}
    temporary = CALIBRATION_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(CALIBRATION_PATH)
    _level_gravity = gravity.tolist()


def start_imu(load_zero=True) -> dict:
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
        imu_state["samples"].append(sample)
        imu_state["last_sample_time_s"] = sample["timestamp_s"]
        imu_state["received_samples"] += 1
        imu_state["error"] = None
    return True


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


def calculate_slope(samples: list):
    if not samples or not CONFIG["mounting_calibrated"]:
        return None
    x_values = []
    y_values = []
    z_values = []
    for sample in samples:
        x_values.append(sample["ax"])
        y_values.append(sample["ay"])
        z_values.append(sample["az"])
    mean_x = float(np.mean(x_values))
    mean_y = float(np.mean(y_values))
    mean_z = float(np.mean(z_values))
    if _level_gravity is not None:
        mean_x, mean_y, mean_z = level_basis(_level_gravity) @ [mean_x, mean_y, mean_z]
    # Calculate directly without rejecting motion, vibration, or gravity deviations.
    horizontal_gravity = math.hypot(mean_y, mean_z)
    if horizontal_gravity == 0:
        return None
    return 100 * mean_x / horizontal_gravity


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor IMU or save its level mounting reference.")
    parser.add_argument("--zero", action="store_true",
                        help="Keep the wheelchair stationary on level ground and save this pose as zero.")
    args = parser.parse_args()
    if args.zero:
        print("Setting zero: wheelchair must be stationary on level ground; keep IMU fixed with x forward.")
    state = start_imu(load_zero=not args.zero)
    zero_deadline = time.monotonic() + 30
    previous_count = 0
    previous_time = time.monotonic()
    try:
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
                continue
            latest = samples[-1]
            if args.zero:
                if len(samples) >= 160:
                    save_level_zero(samples)
                    print(f"IMU zero saved to {CALIBRATION_PATH}. Future runs load it automatically.")
                    break
            output = {"received_rate_hz": round(rate, 1), "window_samples": len(samples)}
            for key in ("ax", "ay", "az", "gx", "gy", "gz"):
                output[key] = round(latest[key], 4)
            output.update(analyze_samples(samples))
            print(json.dumps(output))
    except KeyboardInterrupt:
        print("Stopping IMU.")
    finally:
        stop_imu(state)
