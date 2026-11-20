"""Temporal regression tests; no hardware, real sleeps, or network required."""
from collections import deque
import threading
import unittest
from unittest.mock import patch

from pathlib import Path

import numpy as np

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



class FrameQualityTests(unittest.TestCase):
    def test_frame_problem(self):
        h, w = 720, 1280
        textured = np.random.default_rng(0).integers(40, 200, (h, w, 3), dtype=np.uint8)
        self.assertEqual(ci.frame_problem(np.full((h, w, 3), 255, np.uint8)), "overexposed")
        self.assertEqual(ci.frame_problem(np.zeros((h, w, 3), np.uint8)), "underexposed")
        self.assertEqual(ci.frame_problem(np.full((h, w, 3), 128, np.uint8)), "flat")
        self.assertIsNone(ci.frame_problem(textured))
        self.assertIsNone(ci.frame_problem("not an image"))     # 테스트용 가짜 프레임은 검사하지 않음

    def test_rejected_frames_are_not_inferred(self):
        now = [100.0]
        white = np.full((720, 1280, 3), 255, np.uint8)
        frames = iter([100.2, 100.6])

        def capture(camera):
            stamp = next(frames, None)
            if stamp is None:
                raise KeyboardInterrupt
            now[0] = stamp
            return {"frame_monotonic": stamp, "image": white}

        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", side_effect=capture), \
             patch.object(ci.inference, "analyze_image") as analyze:
            windows = list(ci.iter_windows({}, {}))
        analyze.assert_not_called()
        self.assertEqual(windows[0]["frame_count"], 2)
        self.assertEqual(windows[0]["inference_count"], 0)
        self.assertEqual(windows[0]["rejected_frames"], {"overexposed": 2})
        self.assertEqual((windows[0]["rejected_count"], windows[0]["rejected_reasons"]), (2, "overexposed:2"))
        self.assertIsNone(windows[0]["result"])

    def test_same_frame_has_same_hash(self):
        img = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        self.assertEqual(ci.frame_stats(img)["hash"], ci.frame_stats(img.copy())["hash"])
        self.assertNotEqual(ci.frame_stats(img)["hash"], ci.frame_stats(255 - img)["hash"])

    def test_sampler_saves_first_rejected_per_window_and_periodic_ok(self):
        import json
        import tempfile
        white = np.full((72, 128, 3), 255, np.uint8)
        ok = np.random.default_rng(2).integers(40, 200, (72, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            s = ci.FrameSampler(tmp, every_s=30)
            frame = lambda img, t: {"image": img, "frame_monotonic": t, "frame_time": f"t{t}"}
            self.assertIsNotNone(s.maybe_save(frame(white, 1), 0, ci.frame_stats(white), "overexposed"))
            self.assertIsNone(s.maybe_save(frame(white, 1.5), 0, ci.frame_stats(white), "overexposed"))
            self.assertIsNotNone(s.maybe_save(frame(white, 2.5), 1, ci.frame_stats(white), "overexposed"))
            self.assertIsNotNone(s.maybe_save(frame(ok, 3), 1, ci.frame_stats(ok), None, result("blocks")))
            self.assertIsNone(s.maybe_save(frame(ok, 20), 9, ci.frame_stats(ok), None, result("blocks")))
            self.assertIsNotNone(s.maybe_save(frame(ok, 34), 16, ci.frame_stats(ok), None, result("blocks")))
            lines = [json.loads(x) for x in open(f"{tmp}/samples.jsonl", encoding="utf-8")]
            self.assertEqual([x["problem"] for x in lines], ["overexposed", "overexposed", None, None])
            self.assertEqual(lines[2]["result"]["surface_label"], "blocks")
            self.assertEqual(len(list(Path(tmp).glob("*.jpg"))), 4)


if __name__ == "__main__":
    unittest.main()
