import math
import unittest
from unittest.mock import patch

import imu


class SlopeTests(unittest.TestCase):
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
