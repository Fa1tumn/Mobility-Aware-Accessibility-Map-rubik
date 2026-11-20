"""Standalone reporting must retain the full session without affecting imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imu


def payload(index):
    return dict(timestamp_s=index / 100, ax=0, ay=0, az=9.80665,
                gx=0, gy=0, gz=0, roll_deg=0, pitch_deg=0, yaw_deg=0)


class ReportTests(unittest.TestCase):
    def setUp(self):
        for name in ("_level_gravity", "_attitude_level_gravity"):
            context = patch.object(imu, name, None)
            context.start()
            self.addCleanup(context.stop)

    def state(self, count=200, record=True):
        with patch.dict(imu.CONFIG, {"device_address": None}):
            state = imu.start_imu(load_zero=False, record_session=record)
        for index in range(count):
            self.assertTrue(imu.add_sample(state, payload(index)))
        return state

    def test_ctrl_c_saves_full_session_and_partial_tail(self):
        state = self.state(2200)
        calls = [0]
        def sleep(_):
            calls[0] += 1
            if calls[0] == 2:
                imu.add_sample(state, payload(2200))
                raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(imu, "start_imu", return_value=state) as start, \
                    patch.object(imu, "get_window", return_value=list(state["samples"])[-200:]), \
                    patch.object(imu.time, "sleep", side_effect=sleep):
                self.assertEqual(imu.main(["--report", str(output)]), 0)
            start.assert_called_once_with(load_zero=True, record_session=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["termination"], "interrupted")
            self.assertEqual(len(report["samples"]), 2201)
            self.assertEqual(report["samples"][0]["timestamp_s"], 0)
            self.assertEqual(report["samples"][-1]["timestamp_s"], 22)
            self.assertEqual(report["summary"]["sample_count"], 2201)
            self.assertEqual(report["summary"]["valid_slope_windows"], 1)
            self.assertEqual(report["windows"][0]["slope_percent"], 0)
            self.assertEqual(report["metadata"]["slope_method"], imu.SLOPE_METHOD)
            self.assertEqual(list(Path(directory).glob(".imu-report-*")), [])

    def test_connection_error_still_saves_received_samples(self):
        state = self.state()
        state["error"] = "connection lost"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            with patch.object(imu, "start_imu", return_value=state), \
                    patch.object(imu.time, "sleep", return_value=None):
                self.assertEqual(imu.main(["--report", str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["samples"]), 200)
            self.assertEqual(report["summary"]["last_imu_error"], "connection lost")
            self.assertIn("connection lost", report["termination"])

    def test_library_mode_has_no_session_archive_or_report(self):
        with patch.object(imu, "write_session_report") as writer:
            state = self.state(record=False)
            imu.stop_imu(state)
        self.assertIsNone(state["session_samples"])
        writer.assert_not_called()

    def test_zero_mode_only_saves_calibration(self):
        state = self.state(record=False)
        with patch.object(imu, "start_imu", return_value=state) as start, \
                patch.object(imu, "get_window", return_value=list(state["samples"])), \
                patch.object(imu, "save_level_zero") as zero, \
                patch.object(imu, "write_session_report") as writer, \
                patch.object(imu.time, "sleep", return_value=None):
            self.assertEqual(imu.main(["--zero"]), 0)
        start.assert_called_once_with(load_zero=False, record_session=False)
        zero.assert_called_once()
        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
