"""Regression tests for partial observations and mocked Supabase uploads."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
import storage
import upload_supabase


class PartialDeviceTests(unittest.TestCase):
    def run_pipeline(self, failure=None, iterations=1, upload_failure=False):
        frame = {"frame_monotonic": 100.0, "frame_time": "2026-11-05T00:00:00Z", "image": object()}
        gps = {"gps_monotonic": 100.0, "accuracy_m": 2.0, "latitude": 37.5, "longitude": 127.0}
        result = {"surface_label": "asphalt", "surface_confidence": 0.9, "objects_supported": False, "objects": None}
        metrics = {"slope_percent": 5.0, "vibration_frequency_avg_hz": 8.0}
        operations = {
            "models": (main.inference, "load_models", {}),
            "camera_start": (main.vision, "open_camera", {}),
            "camera_read": (main.vision, "capture_frame", frame),
            "gps_start": (main.phone, "start_phone", {}),
            "gps_read": (main.phone, "get_latest_gps", gps),
            "imu_start": (main.imu, "start_imu", {}),
            "imu_read": (main.imu, "get_window", []),
            "imu_analysis": (main.imu, "analyze_samples", metrics),
            "inference": (main.inference, "analyze_image", result),
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            database = str(Path(directory) / "observations.sqlite3")
            stack.enter_context(patch.dict(storage.CONFIG, {"db_path": database}))
            stack.enter_context(patch.dict(main.os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SECRET_KEY": "test-key"}))
            stack.enter_context(patch.object(main.time, "monotonic", return_value=100.0))
            stack.enter_context(patch.object(main.time, "sleep", side_effect=[None] * (iterations - 1) + [KeyboardInterrupt]))
            for name, (module, method, value) in operations.items():
                kwargs = {"side_effect": RuntimeError("device failed")} if failure == name else {"return_value": value}
                if failure == name + "_none":
                    kwargs = {"return_value": None}
                stack.enter_context(patch.object(module, method, **kwargs))
            for module, method in ((main.vision, "close_camera"), (main.phone, "stop_phone"), (main.imu, "stop_imu")):
                stack.enter_context(patch.object(module, method))
            batches = []

            def fake_upload(records, *args):
                # Exercise real request serialization without sending any network traffic.
                batches.append(json.loads(json.dumps(records, allow_nan=False)))
                if upload_failure and len(batches) == 1:
                    raise RuntimeError("offline")
                return len(records)

            stack.enter_context(patch.object(upload_supabase, "upload_records", side_effect=fake_upload))
            main.main()
            records = upload_supabase.read_records(database)
        return records, batches

    def test_each_device_failure_still_saves_and_uploads(self):
        for failure, fields in (
            ("models", ("surface_label", "surface_confidence", "objects_supported", "objects")),
            ("camera_start", ("surface_label", "surface_confidence", "objects_supported", "objects")),
            ("camera_read", ("surface_label", "surface_confidence", "objects_supported", "objects")),
            ("camera_read_none", ("surface_label", "surface_confidence", "objects_supported", "objects")),
            ("inference", ("surface_label", "surface_confidence", "objects_supported", "objects")),
            ("gps_start", ("latitude", "longitude")),
            ("gps_read", ("latitude", "longitude")),
            ("gps_read_none", ("latitude", "longitude")),
            ("imu_start", ("slope_percent", "vibration_frequency_avg_hz")),
            ("imu_read", ("slope_percent", "vibration_frequency_avg_hz")),
            ("imu_analysis", ("slope_percent", "vibration_frequency_avg_hz")),
        ):
            with self.subTest(failure=failure):
                records, batches = self.run_pipeline(failure)
                self.assertEqual(len(records), 1)
                self.assertEqual(records, batches[0])
                for field in fields:
                    self.assertIsNone(records[0][field])
                expected = {"latitude": 37.5, "longitude": 127.0, "surface_label": "asphalt", "surface_confidence": 0.9, "objects_supported": False, "slope_percent": 5.0, "vibration_frequency_avg_hz": 8.0}
                for field, value in expected.items():
                    if field not in fields:
                        self.assertEqual(records[0][field], value)
                self.assertTrue(records[0]["record_time"])

    def test_upload_failure_retries_saved_records(self):
        records, batches = self.run_pipeline("gps_read", iterations=2, upload_failure=True)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(batches[0]), 1)
        self.assertEqual(batches[1], records)

    def test_invalid_gps_only_clears_coordinates(self):
        frame = {"frame_monotonic": 100.0, "frame_time": "now"}
        for timestamp, accuracy in ((90.0, 2.0), (100.0, 20.0)):
            record = main.combine_data(frame, {"gps_monotonic": timestamp, "accuracy_m": accuracy, "latitude": 1, "longitude": 2}, {"surface_label": "asphalt"}, {"slope_percent": 3})
            self.assertIsNone(record["latitude"])
            self.assertIsNone(record["longitude"])
            self.assertEqual(record["surface_label"], "asphalt")
            self.assertEqual(record["slope_percent"], 3)

    def test_all_missing_still_has_timestamp(self):
        record = main.combine_data(None, None, None)
        self.assertTrue(record.pop("record_time"))
        self.assertTrue(all(value is None for value in record.values()))


if __name__ == "__main__":
    unittest.main()
