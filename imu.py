import asyncio
from collections import deque
import json
import math
import threading
import time

import numpy as np

CONFIG = {
    "device_address": None,
    "notify_uuid": None,
    "mounting_calibrated": False,
    "window_s": 2.0,
}


def start_imu() -> dict:
    imu_state = {
        "samples": deque(maxlen=2000),
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
        "clock_offset_s": None,
        "last_device_time_s": None,
        "error": None,
    }
    if CONFIG["device_address"] is None or CONFIG["notify_uuid"] is None:
        print("IMU Bluetooth is not configured; IMU values will be NULL.")
        return imu_state
    thread = threading.Thread(target=_read_bluetooth, args=(imu_state,), daemon=True)
    imu_state["thread"] = thread
    thread.start()
    return imu_state


def decode_packet(data) -> dict:
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Adapt decode_packet to the IMU device's Bluetooth frame format.") from error
    if not isinstance(payload, dict):
        raise ValueError("IMU packet must decode to a sample dictionary.")
    return payload


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
        previous_time = imu_state["last_device_time_s"]
        if previous_time is not None and sample["timestamp_s"] <= previous_time:
            return False
        if imu_state["clock_offset_s"] is None:
            imu_state["clock_offset_s"] = receive_time - sample["timestamp_s"]
        sample["sample_monotonic"] = sample["timestamp_s"] + imu_state["clock_offset_s"]
        imu_state["samples"].append(sample)
        imu_state["last_device_time_s"] = sample["timestamp_s"]
        imu_state["error"] = None
    return True


def _read_bluetooth(imu_state: dict) -> None:
    try:
        asyncio.run(_receive_bluetooth(imu_state))
    except Exception as error:
        with imu_state["lock"]:
            imu_state["error"] = str(error)
        print(f"IMU Bluetooth error: {error}")


async def _receive_bluetooth(imu_state: dict) -> None:
    from bleak import BleakClient

    def on_data(sender, data):
        try:
            payload = decode_packet(data)
            if not add_sample(imu_state, payload):
                raise ValueError("Invalid or out-of-order IMU sample.")
        except ValueError as error:
            with imu_state["lock"]:
                changed = imu_state["error"] != str(error)
                imu_state["error"] = str(error)
            if changed:
                print(f"IMU packet error: {error}")

    client = BleakClient(CONFIG["device_address"], timeout=3)
    try:
        await asyncio.wait_for(client.connect(), timeout=3)
        if imu_state["stop_event"].is_set():
            return
        await asyncio.wait_for(client.start_notify(CONFIG["notify_uuid"], on_data), timeout=3)
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
    gyro_squared = []
    for sample in samples:
        x_values.append(sample["ax"])
        y_values.append(sample["ay"])
        z_values.append(sample["az"])
        gyro_squared.append(sample["gx"] ** 2 + sample["gy"] ** 2 + sample["gz"] ** 2)
    mean_x = float(np.mean(x_values))
    mean_y = float(np.mean(y_values))
    mean_z = float(np.mean(z_values))
    gravity = math.sqrt(mean_x ** 2 + mean_y ** 2 + mean_z ** 2)
    if mean_z <= 0 or abs(gravity - 9.80665) >= 0.5:
        return None
    if math.sqrt(float(np.mean(gyro_squared))) >= 0.1:
        return None
    if max(np.std(x_values), np.std(y_values), np.std(z_values)) >= 0.5:
        return None
    horizontal_gravity = math.hypot(mean_y, mean_z)
    if horizontal_gravity <= 0.1:
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
        thread.join(timeout=6)
        if thread.is_alive():
            raise RuntimeError("IMU Bluetooth thread did not stop.")
