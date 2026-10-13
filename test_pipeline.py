import http.client
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import imu
import inference
import main
import phone
import storage
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
        data = json.dumps(payload).encode()
        assert imu.add_sample(imu_state, imu.decode_packet(data))


def simulate_qnn(command, case):
    input_list = Path(command[command.index("--input_list") + 1])
    tensor = np.fromfile(input_list.parent / "image.raw", dtype=np.float32)
    assert tensor.size == 3 * 640 * 640
    assert np.isfinite(tensor).all()
    assert np.allclose(tensor, case["image_brightness"] / 255, atol=1 / 255)
    output_dir = Path(command[command.index("--output_dir") + 1]) / "Result_0"
    output_dir.mkdir(parents=True)
    for index, (name, shape) in enumerate(inference.OUTPUT_SHAPES.items()):
        values = np.zeros(shape, dtype=np.float32)
        if name == "boxes":
            values[0, 0] = [320, 320, 640, 640]
        elif name == "class_scores":
            class_id = inference.LABELS.index(case["surface_label"])
            values[0, 0, class_id] = case["surface_confidence"]
        elif name == "mask_coefficients":
            values[0, 0, 0] = 1.0
        elif name == "prototypes":
            values[0, :, :, 0] = 1.0
        values.tofile(output_dir / f"output_{index}.raw")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def inject_inference(image, case, model_path):
    def qnn_output(command, **kwargs):
        return simulate_qnn(command, case)

    models = {"seg_model": model_path, "object_model": None}
    with patch.object(inference.subprocess, "run", side_effect=qnn_output) as qnn:
        result = inference.analyze_image(image, models)
    assert qnn.call_count == 1
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
    assert abs(record["surface_confidence"] - case["surface_confidence"]) < 0.000001
    assert abs(record["slope_percent"] - case["slope_percent"]) < 0.1
    assert abs(record["vibration_frequency_avg_hz"] - case["vibration_frequency_hz"]) < 0.1
    assert record["objects_supported"] == 0 and record["objects"] is None


def test_full_pipeline(database_path, model_path):
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

        def qnn_output(command, **kwargs):
            return simulate_qnn(command, case)

        models = {"seg_model": model_path, "object_model": None}
        with patch.dict(storage.CONFIG, {"db_path": str(database_path)}), \
             patch.dict(imu.CONFIG, {"mounting_calibrated": True}), \
             patch.object(inference, "load_models", return_value=models), \
             patch.object(inference.subprocess, "run", side_effect=qnn_output) as qnn, \
             patch.object(phone, "start_phone", return_value=phone_state), \
             patch.object(imu, "start_imu", return_value=imu_state), \
             patch.object(vision, "open_camera", return_value=camera), \
             patch.object(main.time, "sleep", side_effect=KeyboardInterrupt):
            main.main()
        assert qnn.call_count == 1
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


def test_module_case(case, connection, database_path, model_path):
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
        print("imu.py: 200 JSON samples -> metrics", json.dumps(imu_result))

        result = inject_inference(frame["image"], case, model_path)
        assert result["surface_label"] == case["surface_label"]
        assert abs(result["surface_confidence"] - case["surface_confidence"]) < 0.000001
        print("inference.py: BGR + simulated QNN outputs -> surface", json.dumps(result))

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


def test_module_integration(database_path, model_path):
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
                "surface_confidence": round(generator.uniform(0.75, 0.99), 2),
                "slope_percent": round(generator.uniform(-12.0, 12.0), 1),
                "vibration_frequency_hz": generator.randint(4, 18),
                "image_brightness": generator.randint(30, 220),
            }
            test_module_case(case, connection, database_path, model_path)
        assert len(read_records(database_path)) == 5
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        storage.close_database(connection)
    print("STAGE 2: PASS - 5/5 module integrations.")


def run_test():
    saved_database_path = Path(__file__).resolve().with_name("test_observations.sqlite3")
    print("Mode: software test; device input, model loading, and QNN execution are simulated.")
    print("HTTP, camera pipe reading, IMU calculations, inference processing, and SQLite are real.")
    with tempfile.TemporaryDirectory(prefix="rubik-pipeline-test-") as directory:
        smoke_database_path = Path(directory) / "smoke.sqlite3"
        integration_database_path = Path(directory) / "integration.sqlite3"
        model_path = Path(directory) / "surface.bin"
        model_path.write_bytes(b"simulated-model")
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
        test_full_pipeline(smoke_database_path, model_path)
        test_module_integration(integration_database_path, model_path)
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
