import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import imu
import numpy as np


def sample_for_up(up, timestamp=0.0):
    up = np.asarray(up, dtype=float)
    return dict(ax=up[0], ay=up[1], az=up[2], gx=0, gy=0, gz=0,
                timestamp_s=timestamp, sample_monotonic=timestamp,
                roll_deg=math.degrees(math.atan2(up[1], up[2])),
                pitch_deg=math.degrees(math.atan2(-up[0], math.hypot(up[1], up[2]))),
                yaw_deg=0)


def filtered(samples, state=None):
    state = {} if state is None else state
    for sample in samples:
        imu._filter_attitude(state, sample)
    return samples


class SlopeTests(unittest.TestCase):
    def setUp(self):
        for name in ('_level_gravity', '_attitude_level_gravity'):
            context = patch.object(imu, name, None)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict(imu.CONFIG, {'mounting_calibrated': True, 'slope_filter_tau_s': 0.25})
        context.start()
        self.addCleanup(context.stop)

    def test_packet_decodes_signed_angles_in_degrees(self):
        data = b'\x55\x61' + struct.pack('<9h', 0, 0, 2048, 0, 0, 0, 16384, -8192, -32768)
        sample = imu.decode_packet(data, 1)
        self.assertEqual((sample['roll_deg'], sample['pitch_deg'], sample['yaw_deg']), (90, -45, -180))
        with patch.dict(imu.CONFIG, {'device_address': None}):
            state = imu.start_imu(load_zero=False)
        self.assertTrue(imu.add_sample(state, sample))
        self.assertEqual(state['samples'][-1]['roll_deg'], 90)
        self.assertAlmostEqual(state['samples'][-1]['attitude_slope_deg'], 45)
        self.assertFalse(imu.add_sample(state, dict(sample, timestamp_s=2, pitch_deg=float('nan'))))

    def test_zero_persists_and_preserves_signed_slopes_and_mounting_roll(self):
        gravity = np.array([0.7135, -0.4453, 9.9216])
        fused_zero = np.array([0.2, 2.0, 9.5])
        samples = [dict(sample_for_up(fused_zero, i / 100),
                        ax=gravity[0], ay=gravity[1], az=gravity[2]) for i in range(200)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(imu, 'CALIBRATION_PATH', Path(directory) / 'zero.json'):
            imu.save_level_zero(samples)
            self.assertAlmostEqual(imu.calculate_slope(filtered(samples)), 0)
            imu._level_gravity = imu._attitude_level_gravity = None
            imu.load_calibration()
            self.assertAlmostEqual(imu.calculate_slope(filtered(samples)), 0)
            basis = imu.level_basis(fused_zero)
            for degrees in (-10, 10):
                angle = math.radians(degrees)
                measured = basis.T @ np.array([math.sin(angle), 0, math.cos(angle)])
                for yaw in (-179, 0, 179):
                    sample = dict(sample_for_up(measured), yaw_deg=yaw)
                    self.assertAlmostEqual(imu.calculate_slope(filtered([sample])), 100 * math.tan(angle))

    def test_legacy_zero_is_accepted_without_automatic_recalibration(self):
        import json
        gravity = [0.7, -0.4, 9.9]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(imu, 'CALIBRATION_PATH', Path(directory) / 'zero.json'):
            text = json.dumps({'device_address': imu.CONFIG['device_address'], 'level_gravity': gravity})
            imu.CALIBRATION_PATH.write_text(text)
            imu.load_calibration()
            self.assertIsNone(imu._attitude_level_gravity)
            self.assertAlmostEqual(imu.calculate_slope(filtered([sample_for_up(gravity)])), 0)
            self.assertEqual(imu.CALIBRATION_PATH.read_text(), text)

    def test_failed_zero_preserves_saved_reference(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(imu, 'CALIBRATION_PATH', Path(directory) / 'zero.json'):
            samples = [sample_for_up([0.7, -0.4, 9.9], i / 100) for i in range(200)]
            imu.save_level_zero(samples)
            saved = imu.CALIBRATION_PATH.read_text()
            invalid = [samples[:10], [dict(s, gx=0.5) for s in samples],
                       [dict(s, ax=(-1) ** i) for i, s in enumerate(samples)],
                       [{k: v for k, v in s.items() if k != 'pitch_deg'} for s in samples],
                       [dict(s, pitch_deg=20 * (-1) ** i) for i, s in enumerate(samples)]]
            for values in invalid:
                with self.assertRaises(ValueError):
                    imu.save_level_zero(values)
                self.assertEqual(imu.CALIBRATION_PATH.read_text(), saved)
            with patch.dict(imu.CONFIG, {'device_address': 'different'}):
                with self.assertRaises(ValueError):
                    imu.load_calibration()

    def test_raw_acceleration_spikes_do_not_drive_fused_slope(self):
        samples = []
        for i in range(200):
            sample = sample_for_up([1, 0, 10], i / 100)
            sample.update(ax=30 * (-1) ** i, ay=10, az=30 if i % 5 else 70)
            samples.append(sample)
        result = imu.analyze_samples(filtered(samples))
        self.assertAlmostEqual(result['slope_percent'], 10)
        self.assertIsNotNone(result['vibration_frequency_avg_hz'])

    def test_filter_continues_across_windows_and_resets_after_gap(self):
        state = {}
        first = filtered([sample_for_up([0, 0, 10], i / 100) for i in range(200)], state)
        second = filtered([sample_for_up([1, 0, 10], 2 + i / 100) for i in range(200)], state)
        self.assertEqual(imu.calculate_slope(first), 0)
        self.assertGreater(second[0]['smoothed_slope_deg'], 0)
        self.assertLess(second[0]['smoothed_slope_deg'], 1)
        self.assertAlmostEqual(second[-1]['smoothed_slope_deg'], math.degrees(math.atan(.1)), places=2)
        self.assertGreater(imu.calculate_slope(second), 8)
        self.assertLess(imu.calculate_slope(second), 10)
        after_gap = filtered([sample_for_up([-1, 0, 10], 5)], state)
        self.assertAlmostEqual(imu.calculate_slope(after_gap), -10)

    def test_missing_attitude_and_bad_coverage_remain_null(self):
        self.assertIsNone(imu.calculate_slope([]))
        self.assertIsNone(imu.calculate_slope(filtered([dict(ax=1, ay=0, az=10, sample_monotonic=0)])))
        self.assertIsNone(imu.calculate_slope(filtered([dict(sample_for_up([0, 0, 10]), pitch_deg=float('nan'))])))
        self.assertIsNone(imu.calculate_slope(filtered([sample_for_up([1, 0, 0])])))
        partial = filtered([sample_for_up([1, 0, 10], i / 100) for i in range(137)])
        self.assertIsNone(imu.analyze_samples(partial)['slope_percent'])
        with patch.dict(imu.CONFIG, {'mounting_calibrated': False}):
            self.assertIsNone(imu.calculate_slope(filtered([sample_for_up([1, 0, 10])])))

    def test_roll_wrap_does_not_create_angle_spike(self):
        with patch.object(imu, '_attitude_level_gravity', [0, 0, -1]):
            samples = [dict(sample_for_up([0, 0, -1], i / 100), roll_deg=179 if i % 2 else -179)
                       for i in range(200)]
            self.assertAlmostEqual(imu.calculate_slope(filtered(samples)), 0)


if __name__ == '__main__':
    unittest.main()
