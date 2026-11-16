"""Temporal regression tests; no hardware, real sleeps, or network required."""
from collections import deque
import threading
import unittest
from unittest.mock import patch

import camera_inference as ci
import phone


def result(label, confidence=0.9, ratio=0.6):
    return {"surface_label": label, "surface_confidence": confidence,
            "area_accept_ratio": ratio, "material_accept_ratio": ratio / 2,
            "model_version": "test", "policy_version": "test",
            "objects_supported": False, "objects": None}


class AggregationTests(unittest.TestCase):
    def test_vote_confidence_and_ratios(self):
        rows = [result("cement", 0.8, 0.4), result("asphalt", 0.99, 1), result("cement", 0.9, 0.4)]
        merged = ci.aggregate_results(rows)
        self.assertEqual(merged["surface_label"], "cement")
        self.assertAlmostEqual(merged["surface_confidence"], 0.85)
        self.assertAlmostEqual(merged["area_accept_ratio"], 0.6)
        self.assertEqual(rows[-1]["surface_confidence"], 0.9)

    def test_unknown_votes_empty_and_ties(self):
        self.assertIsNone(ci.aggregate_results([]))
        rows = [result("unknown", None), result("unknown", None), result("cement")]
        merged = ci.aggregate_results(rows)
        self.assertEqual(merged["surface_label"], "unknown")
        self.assertIsNone(merged["surface_confidence"])
        self.assertEqual(ci.aggregate_results([result("cement", 0.8), result("asphalt")])["surface_label"], "asphalt")

    def test_fixed_boundaries_late_inference_stale_frames_and_failures(self):
        now = [100.0]
        # The first inference finishes in the next window but must stay in
        # [100,102). Duplicate/stale frames must never vote again.
        frames = iter([(100.5, 102.2, "cement"), (100.5, 102.2, "cement"),
                       (102.0, 102.4, "asphalt"), (103.0, 103.2, "fail"),
                       (104.0, 104.2, "cement")])
        active = [None]

        def capture(camera):
            stamp, finish, label = next(frames)
            now[0] = max(now[0], stamp)
            active[0] = (finish, label)
            return {"frame_monotonic": stamp, "image": label}

        def analyze(*args):
            finish, label = active[0]
            now[0] = finish
            if label == "fail":
                raise RuntimeError("bad frame")
            return result(label)

        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", side_effect=capture), \
             patch.object(ci.inference, "analyze_image", side_effect=analyze):
            windows = ci.iter_windows({}, {}, 2)
            first, second = next(windows), next(windows)
            windows.close()
        self.assertEqual((first["start_monotonic"], first["end_monotonic"]), (100, 102))
        self.assertEqual(first["result"]["surface_label"], "cement")
        self.assertEqual(first["frame_count"], 1)
        self.assertEqual((second["start_monotonic"], second["end_monotonic"]), (102, 104))
        self.assertEqual(second["result"]["surface_label"], "asphalt")
        self.assertEqual(second["frame_count"], 2)
        self.assertEqual(second["inference_count"], 1)

    def test_long_inference_keeps_gaps_empty(self):
        now = [100.0]
        def analyze(*args):
            now[0] = 106.5
            return result("cement")
        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", return_value={"frame_monotonic": 100.5, "image": object()}), \
             patch.object(ci.inference, "analyze_image", side_effect=analyze):
            windows = ci.iter_windows({}, {})
            first, second, third = next(windows), next(windows), next(windows)
            windows.close()
        self.assertEqual(first["inference_count"], 1)
        self.assertEqual(second["end_monotonic"], 104)
        self.assertEqual(third["end_monotonic"], 106)
        self.assertIsNone(second["result"])
        self.assertIsNone(third["result"])

    def test_unavailable_devices_emit_empty_window(self):
        now = [100.0]
        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.time, "sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)):
            windows = ci.iter_windows(None, None)
            window = next(windows)
            windows.close()
        self.assertIsNone(window["result"])
        self.assertEqual(window["frame_count"], 0)
        self.assertEqual(window["end_monotonic"], 102)

    def test_interrupt_flushes_partial_window(self):
        now = [100.0]
        def capture(camera):
            if now[0] > 100:
                raise KeyboardInterrupt
            now[0] = 100.5
            return {"frame_monotonic": 100.5, "image": object()}
        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", side_effect=capture), \
             patch.object(ci.inference, "analyze_image", return_value=result("cement")):
            windows = list(ci.iter_windows({}, {}))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["end_monotonic"], 100.5)
        self.assertEqual(windows[0]["inference_count"], 1)

    def test_invalid_duration(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                next(ci.iter_windows(None, None, value))

    def test_gps_uses_history_near_boundary(self):
        state = {"lock": threading.Lock(), "gps_history": deque([
            {"gps_monotonic": 101.9, "latitude": 1},
            {"gps_monotonic": 104.0, "latitude": 2},
        ])}
        gps = phone.get_gps_at(state, 102.0)
        self.assertEqual(gps["latitude"], 1)
        gps["latitude"] = 9
        self.assertEqual(state["gps_history"][0]["latitude"], 1)


if __name__ == "__main__":
    unittest.main()
