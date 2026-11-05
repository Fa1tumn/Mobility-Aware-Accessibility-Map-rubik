import asyncio
from collections import deque
import json
import math
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


def start_imu() -> dict:
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
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("IMU Bluetooth thread did not stop.")


if __name__ == "__main__":
    state = start_imu()
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
            rate = (count - previous_count) / (now - previous_time)
            previous_count = count
            previous_time = now
            if not samples:
                print(f"Waiting for recent IMU samples. received_rate_hz={rate:.1f}")
                continue
            latest = samples[-1]
            output = {"received_rate_hz": round(rate, 1), "window_samples": len(samples)}
            for key in ("ax", "ay", "az", "gx", "gy", "gz"):
                output[key] = round(latest[key], 4)
            output.update(analyze_samples(samples))
            print(json.dumps(output))
    except KeyboardInterrupt:
        print("Stopping IMU.")
    finally:
        stop_imu(state)
