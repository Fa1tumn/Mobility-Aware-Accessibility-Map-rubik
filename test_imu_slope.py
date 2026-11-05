import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imu
import numpy as np


class SlopeTests(unittest.TestCase):
    def setUp(self):
        zero_patch = patch.object(imu, "_level_gravity", None)
        zero_patch.start()
        self.addCleanup(zero_patch.stop)

    def test_zero_persists_and_preserves_signed_slopes(self):
        gravity = np.array([0.7135, -0.4453, 9.9216])
        samples = [dict(ax=gravity[0], ay=gravity[1], az=gravity[2],
                        gx=0, gy=0, gz=0) for _ in range(200)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(imu, "CALIBRATION_PATH", Path(directory) / "zero.json"), \
                patch.dict(imu.CONFIG, {"mounting_calibrated": True}):
            imu.save_level_zero(samples)
            self.assertAlmostEqual(imu.calculate_slope(samples), 0)
            imu._level_gravity = None
            imu.load_calibration()
            self.assertAlmostEqual(imu.calculate_slope(samples), 0)
            basis = imu.level_basis(gravity)
            for degrees in (-10, 10):
                angle = math.radians(degrees)
                measured = basis.T @ (9.80665 * np.array([math.sin(angle), 0, math.cos(angle)]))
                slope = imu.calculate_slope([dict(zip(("ax", "ay", "az"), measured))])
                self.assertAlmostEqual(slope, 100 * math.tan(angle))

    def test_failed_zero_does_not_replace_saved_reference(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(imu, "CALIBRATION_PATH", Path(directory) / "zero.json"):
            samples = [dict(ax=0.7, ay=-0.4, az=9.9, gx=0, gy=0, gz=0)] * 200
            imu.save_level_zero(samples)
            saved = imu.CALIBRATION_PATH.read_text()
            moving = [dict(s, gx=0.5) for s in samples]
            vibrating = [dict(s, ax=(-1) ** i) for i, s in enumerate(samples)]
            for invalid in (samples[:10], moving, vibrating):
                with self.assertRaises(ValueError):
                    imu.save_level_zero(invalid)
                self.assertEqual(imu.CALIBRATION_PATH.read_text(), saved)
            with patch.dict(imu.CONFIG, {"device_address": "different"}):
                with self.assertRaises(ValueError):
                    imu.load_calibration()

    def test_motion_and_gravity_deviations_do_not_reject_slope(self):
        cases = {
            "rotation": [(1, 0, 9.80665, 1)] * 200,
            "gravity_deviation": [(1, 0, 15, 0)] * 200,
            "vibration": [(1 + (-1) ** i, 0, 9.80665, 0) for i in range(200)],
            "negative_z": [(1, 0, -9.80665, 0)] * 200,
            "small_denominator": [(1, 0, 0.05, 0)] * 200,
        }
        with patch.dict(imu.CONFIG, {"mounting_calibrated": True}):
            for name, values in cases.items():
                with self.subTest(name=name):
                    samples = [dict(sample_monotonic=i / 100, ax=x, ay=y, az=z,
                                    gx=g, gy=0, gz=0)
                               for i, (x, y, z, g) in enumerate(values)]
                    mean_x = sum(v[0] for v in values) / len(values)
                    mean_y = sum(v[1] for v in values) / len(values)
                    mean_z = sum(v[2] for v in values) / len(values)
                    self.assertAlmostEqual(
                        imu.analyze_samples(samples)["slope_percent"],
                        100 * mean_x / math.hypot(mean_y, mean_z),
                    )

    def test_empty_samples_and_zero_denominator_remain_null(self):
        with patch.dict(imu.CONFIG, {"mounting_calibrated": True}):
            self.assertIsNone(imu.calculate_slope([]))
            self.assertIsNone(imu.calculate_slope([{"ax": 1, "ay": 0, "az": 0}]))


if __name__ == "__main__":
    unittest.main()
