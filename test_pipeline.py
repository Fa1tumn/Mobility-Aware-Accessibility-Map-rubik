import http.client
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

import imu
import inference
import main
import phone
import storage
import surface_model
import vision


class TestCameraProcess:
    def __init__(self, jpeg_bytes):
        read_fd, write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.input = os.fdopen(write_fd, "wb", buffering=0)
        self.input.write(jpeg_bytes)

    def poll(self):
        self.input.close()
        return 0


def inject_phone(phone_state, case):
    payload = {
        "latitude": case["latitude"],
        "longitude": case["longitude"],
        "accuracy_m": case["accuracy_m"],
    }
    port = phone_state["server"].server_address[1]
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("POST", "/gps", json.dumps(payload), {"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 200
        response.read()
    finally:
        client.close()
    return phone.get_latest_gps(phone_state)


def inject_camera(case):
    image = np.full((24, 32, 3), case["image_brightness"], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    process = TestCameraProcess(encoded.tobytes())
    with patch.object(vision.subprocess, "Popen", return_value=process):
        return vision.open_camera()


def inject_imu(imu_state, case, frame_monotonic):
    imu_state["clock_offset_s"] = frame_monotonic - 2.0
    angle = math.atan(case["slope_percent"] / 100)
    packets = []
    for index in range(200):
        timestamp = (index + 1) / 100
        payload = {
            "timestamp_s": timestamp,
            "ax": 9.80665 * math.sin(angle),
            "ay": 0.0,
            "az": 9.80665 * math.cos(angle) + 0.1 * math.sin(
                2 * math.pi * case["vibration_frequency_hz"] * timestamp
            ),
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
        }
        values = []
        for key in ("ax", "ay", "az"):
            values.append(round(payload[key] / 9.80665 * 32768 / 16))
        for key in ("gx", "gy", "gz"):
            values.append(round(math.degrees(payload[key]) * 32768 / 2000))
        data = b"\x55\x61" + struct.pack("<9h", *values, 0, 0, 0)
        packets.append(data)
    stream = b"".join(packets)
    with patch.object(imu.time, "monotonic", return_value=frame_monotonic):
        imu.receive_data(imu_state, stream[:11])
        assert not imu_state["samples"]
        imu.receive_data(imu_state, stream[11:])


def _logits_for_class(num_classes, height, width, class_id, confidence):
    """Build [1, C, H, W] logits whose softmax yields `confidence` for class_id
    uniformly across every pixel. Solve for the winning logit l given the rest
    at 0: confidence = e^l / (e^l + (C-1)), so l = ln((C-1) * p / (1 - p))."""
    probability = min(max(float(confidence), 1e-6), 1 - 1e-6)
    winning_logit = math.log((num_classes - 1) * probability / (1 - probability))
    logits = np.zeros((1, num_classes, height, width), dtype=np.float32)
    logits[0, class_id, :, :] = winning_logit
    return logits


class FakeBackend(surface_model.Backend):
    """Returns pre-built logits so tests control the surface decision without
    a real NPU. Mirrors the swappable-backend design of surface_model."""

    def __init__(self, logits, counter):
        self.logits = logits
        self.counter = counter

    def run(self, input_tensor):
        # Confirm the adapter feeds a normalized [1, 3, H, W] float32 tensor.
        assert input_tensor.dtype == np.float32
        assert input_tensor.shape[:2] == (1, 3)
        assert np.isfinite(input_tensor).all()
        self.counter.append(1)
        return self.logits


def build_fake_surface_model(case):
    """A SurfaceModel whose two heads vote the case's surface everywhere, so the
    Area sidewalk + Material thresholds both pass and surface_label matches."""
    spec = surface_model.load_spec()
    _, _, height, width = spec["input_shape"]
    area_labels = spec["models"]["area"]["labels"]
    material_labels = spec["models"]["material"]["labels"]
    sidewalk_id = area_labels.index("sidewalk")
    material_id = material_labels.index(case["surface_label"])
    # Area votes sidewalk at a high fixed confidence; Material votes the case
    # class at the case confidence. Both are >= 0.8 so the whole frame is kept.
    area_logits = _logits_for_class(len(area_labels), height, width, sidewalk_id, 0.95)
    material_logits = _logits_for_class(
        len(material_labels), height, width, material_id, case["surface_confidence"]
    )
    calls = []
    model = surface_model.SurfaceModel(
        FakeBackend(area_logits, calls), FakeBackend(material_logits, calls), spec
    )
    return model, calls


def inject_inference(image, case):
    model, calls = build_fake_surface_model(case)
    with patch.object(inference, "load_models", return_value={"surface": model}):
        models = inference.load_models()
        result = inference.analyze_image(image, models)
    assert len(calls) == 2  # Area then Material ran on the same frame.
    return result


def read_records(database_path):
    connection = sqlite3.connect(database_path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM observations ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def check_record(record, case, frame):
    assert record["record_time"] == frame["frame_time"]
    assert record["latitude"] == case["latitude"]
    assert record["longitude"] == case["longitude"]
    assert record["surface_label"] == case["surface_label"]
    assert abs(record["surface_confidence"] - case["surface_confidence"]) < 0.001
    # Whole frame votes sidewalk material, so both acceptance ratios are ~1.0.
    assert abs(record["area_accept_ratio"] - 1.0) < 0.001
    assert abs(record["material_accept_ratio"] - 1.0) < 0.001
    assert record["model_version"] == "lraspp_mobilenet_v3_large:area_e7+material_e10"
    assert record["policy_version"] == "area_sidewalk>=0.8&material>=0.8"
    assert abs(record["slope_percent"] - case["slope_percent"]) < 0.1
    assert abs(record["vibration_frequency_avg_hz"] - case["vibration_frequency_hz"]) < 0.1
    assert record["objects_supported"] == 0 and record["objects"] is None


def test_full_pipeline(database_path):
    print("\nSTAGE 1: run the full pipeline once")
    case = {
        "latitude": 37.5, "longitude": 127.0, "accuracy_m": 4.0,
        "surface_label": "asphalt", "surface_confidence": 0.9,
        "slope_percent": 10.0, "vibration_frequency_hz": 8,
        "image_brightness": 128,
    }
    phone_state = None
    imu_state = None
    camera = None
    try:
        with patch.dict(phone.CONFIG, {"host": "127.0.0.1", "port": 0}):
            phone_state = phone.start_phone()
        with patch.dict(imu.CONFIG, {"device_address": None, "notify_uuid": None}):
            imu_state = imu.start_imu()
        inject_phone(phone_state, case)
        camera = inject_camera(case)
        frame = vision.capture_frame(camera)
        assert frame is not None
        inject_imu(imu_state, case, frame["frame_monotonic"])
        camera["frames"].put(frame)

        fake_model, inference_calls = build_fake_surface_model(case)
        with patch.dict(storage.CONFIG, {"db_path": str(database_path)}), \
             patch.dict(imu.CONFIG, {"mounting_calibrated": True}), \
             patch.object(inference, "load_models", return_value={"surface": fake_model}), \
             patch.object(phone, "start_phone", return_value=phone_state), \
             patch.object(imu, "start_imu", return_value=imu_state), \
             patch.object(vision, "open_camera", return_value=camera), \
             patch.object(main.time, "sleep", side_effect=KeyboardInterrupt):
            main.main()
        assert len(inference_calls) == 2  # Area + Material ran once each.
        records = read_records(database_path)
        assert len(records) == 2
        assert records[0]["record_time"] == "old"
        assert records[0]["slope_percent"] is None
        assert records[0]["vibration_frequency_avg_hz"] is None
        check_record(records[1], case, frame)
        assert not phone_state["thread"].is_alive()
        assert not camera["thread"].is_alive()
        assert camera["stop_event"].is_set()
        assert imu_state["stop_event"].is_set()
    finally:
        vision.close_camera(camera)
        phone.stop_phone(phone_state)
        imu.stop_imu(imu_state)
    print("STAGE 1: PASS - main loop, module connections, migration, save, and cleanup.")


def test_module_case(case, connection, database_path):
    print(f"\nCASE {case['case_number']}: module inputs")
    print(json.dumps(case, indent=2))
    phone_state = None
    imu_state = None
    camera = None
    try:
        with patch.dict(phone.CONFIG, {"host": "127.0.0.1", "port": 0}):
            phone_state = phone.start_phone()
        gps = inject_phone(phone_state, case)
        assert gps["latitude"] == case["latitude"]
        assert gps["longitude"] == case["longitude"]
        assert gps["accuracy_m"] == case["accuracy_m"]
        print("phone.py: HTTP JSON -> GPS", json.dumps(gps))

        camera = inject_camera(case)
        frame = vision.capture_frame(camera)
        assert frame is not None
        assert frame["image"].shape == (24, 32, 3)
        assert np.allclose(frame["image"], case["image_brightness"], atol=1)
        print(f"vision.py: JPEG pipe -> BGR {frame['image'].shape}, time={frame['frame_time']}")

        with patch.dict(imu.CONFIG, {"device_address": None, "notify_uuid": None}):
            imu_state = imu.start_imu()
        inject_imu(imu_state, case, frame["frame_monotonic"])
        samples = imu.get_window(imu_state, frame["frame_monotonic"])
        assert len(samples) == 200
        with patch.dict(imu.CONFIG, {"mounting_calibrated": True}):
            imu_result = imu.analyze_samples(samples)
        assert abs(imu_result["slope_percent"] - case["slope_percent"]) < 0.1
        assert abs(imu_result["vibration_frequency_avg_hz"] - case["vibration_frequency_hz"]) < 0.1
        assert imu.get_window(imu_state, frame["frame_monotonic"] + 1.0) == []
        assert imu.analyze_samples([]) == {"slope_percent": None, "vibration_frequency_avg_hz": None}
        print("imu.py: 200 binary BLE packets -> metrics", json.dumps(imu_result))

        result = inject_inference(frame["image"], case)
        assert result["surface_label"] == case["surface_label"]
        assert abs(result["surface_confidence"] - case["surface_confidence"]) < 0.001
        print("inference.py: BGR + Area/Material logits -> surface", json.dumps(result))

        record = main.combine_data(frame, gps, result, imu_result)
        assert record is not None
        record_id = storage.save_record(connection, record)
        saved_record = read_records(database_path)[-1]
        assert record_id == case["case_number"]
        assert saved_record["id"] == record_id
        check_record(saved_record, case, frame)
    finally:
        vision.close_camera(camera)
        phone.stop_phone(phone_state)
        imu.stop_imu(imu_state)
    print("main.py -> storage.py: combined database record")
    print(json.dumps(saved_record, indent=2, allow_nan=False))
    print(f"CASE {case['case_number']}: PASS - all module outputs combined and saved.")


def test_module_integration(database_path):
    print("\nSTAGE 2: inject data into each module, then combine the outputs")
    generator = random.Random(42)
    with patch.dict(storage.CONFIG, {"db_path": str(database_path)}):
        connection = storage.open_database()
    try:
        for case_number in range(1, 6):
            case = {
                "case_number": case_number,
                "latitude": round(generator.uniform(37.49, 37.51), 6),
                "longitude": round(generator.uniform(126.99, 127.01), 6),
                "accuracy_m": round(generator.uniform(1.0, 8.0), 1),
                "surface_label": generator.choice(inference.LABELS),
                # Keep confidence clear of the 0.8 policy threshold (avoiding
                # float32 boundary flicker) so the material is accepted and
                # surface_label is preserved.
                "surface_confidence": round(generator.uniform(0.82, 0.99), 2),
                "slope_percent": round(generator.uniform(-12.0, 12.0), 1),
                "vibration_frequency_hz": generator.randint(4, 18),
                "image_brightness": generator.randint(30, 220),
            }
            test_module_case(case, connection, database_path)
        assert len(read_records(database_path)) == 5
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        storage.close_database(connection)
    print("STAGE 2: PASS - 5/5 module integrations.")


def test_policy_rejects_low_confidence():
    """Material below the 0.8 threshold yields unknown, not a stored label."""
    image = np.full((24, 32, 3), 128, dtype=np.uint8)
    low = inject_inference(image, {"surface_label": "cement", "surface_confidence": 0.5})
    assert low["surface_label"] == "unknown"
    assert low["surface_confidence"] is None
    assert low["area_accept_ratio"] == 1.0          # Area (0.95) still passes.
    assert low["material_accept_ratio"] == 0.0       # Material gate rejects all.
    high = inject_inference(image, {"surface_label": "cement", "surface_confidence": 0.9})
    assert high["surface_label"] == "cement"
    assert abs(high["material_accept_ratio"] - 1.0) < 0.001
    print("POLICY: PASS - material below 0.8 -> unknown; at/above -> accepted.")


def test_imu_clock_drift():
    packet = b"\x55\x61" + struct.pack("<9h", 0, 0, 2048, 0, 0, 0, 0, 0, 0)
    for actual_rate in (99.3, 100.7):
        with patch.dict(imu.CONFIG, {"device_address": None, "sample_rate_hz": 100}):
            state = imu.start_imu()
            try:
                previous_time = 0.0
                for batch in range(2500):
                    jitter = (0.0, 0.012, -0.008, 0.004)[batch % 4]
                    receive_time = 1000 + (batch + 1) * 4 / actual_rate + jitter
                    with patch.object(imu.time, "monotonic", return_value=receive_time):
                        imu.receive_data(state, packet * 4)
                    samples = imu.get_window(state, receive_time + 0.005)
                    assert samples, (actual_rate, batch)
                    assert samples[-1]["sample_monotonic"] > previous_time
                    previous_time = samples[-1]["sample_monotonic"]
                    if batch > 100:
                        assert len(samples) >= 180
                        assert samples[-1]["sample_monotonic"] - samples[0]["sample_monotonic"] > 1.6
                assert imu.get_window(state, receive_time + 0.2) == []
            finally:
                imu.stop_imu(state)
    print("IMU TIMING: PASS - slow clock drift, notification jitter, and stale-data rejection.")


def run_test():
    saved_database_path = Path(__file__).resolve().with_name("test_observations.sqlite3")
    print("Mode: software test; device input and the NPU/model backends are simulated.")
    print("HTTP, camera pipe reading, IMU calculations, the Area+Material adapter, and SQLite are real.")
    test_imu_clock_drift()
    test_policy_rejects_low_confidence()
    with tempfile.TemporaryDirectory(prefix="rubik-pipeline-test-") as directory:
        smoke_database_path = Path(directory) / "smoke.sqlite3"
        integration_database_path = Path(directory) / "integration.sqlite3"
        # Legacy schema without the IMU/area/material columns, to exercise the
        # automatic column migration in storage.open_database().
        connection = sqlite3.connect(smoke_database_path)
        try:
            connection.execute("""CREATE TABLE observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, record_time TEXT,
                latitude REAL, longitude REAL, surface_label TEXT,
                surface_confidence REAL, objects_supported INTEGER, objects TEXT
            )""")
            connection.execute("INSERT INTO observations (record_time) VALUES (?)", ("old",))
            connection.commit()
        finally:
            connection.close()
        test_full_pipeline(smoke_database_path)
        test_module_integration(integration_database_path)
        connection = sqlite3.connect(integration_database_path)
        saved_connection = sqlite3.connect(saved_database_path)
        try:
            connection.backup(saved_connection)
        finally:
            saved_connection.close()
            connection.close()
    print("\nTEST RESULT: PASS - full pipeline + 5/5 module integrations.")
    print(f"Saved test database: {saved_database_path}")


if __name__ == "__main__":
    run_test()
