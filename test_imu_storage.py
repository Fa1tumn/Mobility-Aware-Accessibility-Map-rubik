"""Raw IMU recording, replay, and atomic observation writes in temporary SQLite."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imu
from inspect_imu_windows import summarize_window
import main
import storage
import upload_supabase


class ImuStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = str(Path(directory.name) / "observations.sqlite3")
        with patch.dict(storage.CONFIG, {"db_path": self.database}):
            self.connection = storage.open_database()
        self.addCleanup(self.connection.close)
        self.record = main.combine_data(None, None, None, {"slope_percent": 0.0})

    def test_raw_round_trip_and_replay_do_not_depend_on_current_calibration(self):
        with patch.object(imu, "_level_gravity", [0, 0, 9.80665]):
            samples = [dict(timestamp_s=i / 100, sample_monotonic=100 + i / 100,
                            ax=8 if i < 20 else 0, ay=0, az=9.80665,
                            gx=0, gy=0, gz=0) for i in range(200)]
            payload = imu.snapshot_window(samples)
        # A prior observation without raw samples must remain valid.
        first = storage.save_record(self.connection, self.record)
        second = storage.save_record(self.connection, self.record, payload)
        identifier, encoded = self.connection.execute("SELECT observation_id, payload FROM imu_windows").fetchone()
        self.assertEqual(identifier, second)
        self.assertNotEqual(identifier, first)
        saved = json.loads(encoded)
        self.assertEqual(saved, payload)
        with patch.object(imu, "_level_gravity", [1, 0, 9]), \
                patch.dict(imu.CONFIG, {"mounting_calibrated": False}):
            stats = summarize_window(saved)
        self.assertEqual(stats["samples"], 200)
        self.assertEqual(stats["median_percent"], 0)
        self.assertGreater(stats["mean_percent"], 8)
        self.assertGreater(stats["accel_std_m_s2"], 0)
        self.assertEqual(stats["gyro_rms_rad_s"], 0)
        # Standalone uploads read observation fields only.
        records = upload_supabase.read_records(self.database)
        self.assertEqual(len(records), 2)
        self.assertNotIn("payload", records[-1])
        self.assertNotIn("samples", records[-1])

    def test_failed_raw_write_rolls_back_observation(self):
        prior = storage.save_record(self.connection, self.record)
        with self.assertRaises(ValueError):
            storage.save_record(self.connection, self.record, {"samples": [float("nan")]})
        self.assertEqual(self.connection.execute("SELECT id FROM observations").fetchall(), [(prior,)])
        self.assertEqual(self.connection.execute("SELECT count(*) FROM imu_windows").fetchone()[0], 0)

    def test_empty_window_and_snapshot_copy(self):
        payload = imu.snapshot_window([])
        self.assertIsNone(summarize_window(payload)["median_percent"])
        identifier = storage.save_record(self.connection, self.record, payload)
        self.connection.execute("DELETE FROM observations WHERE id=?", (identifier,))
        self.connection.commit()
        self.assertEqual(self.connection.execute("SELECT count(*) FROM imu_windows").fetchone()[0], 0)
        samples = [{"ax": 1}]
        copied = imu.snapshot_window(samples)
        samples[0]["ax"] = 2
        self.assertEqual(copied["samples"][0]["ax"], 1)


if __name__ == "__main__":
    unittest.main()
