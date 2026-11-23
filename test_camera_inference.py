"""Temporal regression tests; no hardware, real sleeps, or network required."""
from collections import deque
import argparse
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from pathlib import Path

import numpy as np

import camera_inference as ci
import phone


def setUpModule():
    directory = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(directory.cleanup)
    config_path = Path(directory.name) / "camera_config.json"
    config_path.write_text(json.dumps({
        "exposure_metering": "center-weighted", "exposure_compensation": 0, "max_exposure_us": 0,
        "save_sharp_frames": False, "save_rejected_frames": False, "sharp_frame_dir": "sharp_frames",
    }), encoding="utf-8")
    config_patch = patch.object(ci.vision, "CAMERA_CONFIG_PATH", config_path)
    config_patch.start()
    unittest.addModuleCleanup(config_patch.stop)
    env = patch.dict(os.environ, {"SHARP_FRAME_DIR": directory.name})
    env.start()
    unittest.addModuleCleanup(env.stop)


def result(label, confidence=0.9, ratio=0.6):
    return {"surface_label": label, "surface_confidence": confidence,
            "area_accept_ratio": ratio, "material_accept_ratio": ratio / 2,
            "model_version": "test", "policy_version": "test",
            "objects_supported": False, "objects": None}


def textured_frame():
    return np.random.default_rng(42).integers(40, 200, (72, 128, 3), dtype=np.uint8)


class CameraOptionsTests(unittest.TestCase):
    def test_default_auto_exposure_and_metering(self):
        with patch.dict(os.environ, {}, clear=True):
            opts = ci.vision.camera_source_options()
        self.assertIn("exposure-mode=auto", opts)
        self.assertIn("exposure-metering=center-weighted", opts)
        self.assertFalse(any(opt.startswith("custom-exposure-table=") for opt in opts))

    def test_json_settings_and_per_run_overrides(self):
        settings = dict(ci.vision.load_camera_config(), exposure_metering="average",
                        max_exposure_us=8000, exposure_compensation=2, save_sharp_frames=True,
                        save_rejected_frames=True, sharp_frame_dir="archive")
        parser = argparse.ArgumentParser()
        ci.add_frame_archive_options(parser)
        args = parser.parse_args(["--no-save-rejected-frames"])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / "camera_config.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            with patch.object(ci.vision, "CAMERA_CONFIG_PATH", path):
                opts = ci.vision.camera_source_options()
                self.assertIn("exposure-metering=average", opts)
                self.assertIn("exposure-compensation=2", opts)
                self.assertIn("expTimeKneeEntries=(int)<10040,8000000,8000000>", " ".join(opts))
                with patch.dict(os.environ, {"CAMERA_MAX_EXPOSURE_US": "0"}):
                    self.assertFalse(any("custom-exposure-table" in v for v in ci.vision.camera_source_options()))
                good = textured_frame()
                frames = [{"image": image, "frame_monotonic": stamp}
                          for image, stamp in [(np.zeros_like(good), 101), (good, 101.1)]]
                with patch.object(ci.time, "monotonic", return_value=100), \
                     patch.object(ci.vision, "capture_frame", side_effect=[*frames, KeyboardInterrupt()]):
                    list(ci.iter_windows({}, None, **vars(args)))
                self.assertEqual(len(list((Path(tmp) / "archive").glob("*.jpg"))), 1)
                self.assertFalse((Path(tmp) / "archive" / "rejected").exists())
                settings["save_sharp_frames"] = "false"
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "save_sharp_frames"):
                    ci.vision.load_camera_config()

    def test_explicit_shutter_cap_preserves_auto_metering(self):
        with patch.dict(os.environ, {"CAMERA_MAX_EXPOSURE_US": "4000"}, clear=True):
            opts = ci.vision.camera_source_options()
        self.assertIn("exposure-mode=auto", opts)
        self.assertIn("exposure-metering=center-weighted", opts)
        table, = [v for v in opts if v.startswith("custom-exposure-table=")]
        self.assertIn("expTimeKneeEntries=(int)<10040,4000000,4000000>", table)

    def test_factory_curve_and_metering_overrides(self):
        with patch.dict(os.environ, {"CAMERA_MAX_EXPOSURE_US": "0",
                                     "CAMERA_EXPOSURE_METERING": "average",
                                     "CAMERA_EXPOSURE_COMPENSATION": "-8"}):
            opts = ci.vision.camera_source_options()
        self.assertIn("exposure-metering=average", opts)
        self.assertIn("exposure-compensation=-8", opts)
        self.assertFalse(any(opt.startswith("custom-exposure-table=") for opt in opts))
        with patch.dict(os.environ, {"CAMERA_MAX_EXPOSURE_US": "-1"}), self.assertRaises(ValueError):
            ci.vision.camera_source_options()


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
            return {"frame_monotonic": stamp, "image": textured_frame()}

        def analyze(*args):
            finish, label = active[0]
            now[0] = finish
            if label == "fail":
                raise RuntimeError("bad frame")
            return result(label)

        with patch.object(ci, "FRAMES_PER_GROUP", 1), \
             patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
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
        with patch.object(ci, "FRAMES_PER_GROUP", 1), \
             patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", return_value={"frame_monotonic": 100.5, "image": textured_frame()}), \
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
            return {"frame_monotonic": 100.5, "image": textured_frame()}
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

    def test_sharpest_road_selects_whole_frame_and_flushes_at_boundary(self):
        import cv2

        sharp = textured_frame()
        soft = sharp.copy()
        road = np.s_[36:, 32:96]
        soft[road] = cv2.GaussianBlur(sharp[road], (11, 11), 3)
        # High contrast outside the road region must not change its score.
        sharp[:36] = 100
        self.assertGreater(ci.road_sharpness(sharp), ci.road_sharpness(soft))
        bad = np.full_like(sharp, 255)
        candidates = iter([(100.1, soft), (100.2, sharp), (100.3, bad),
                           (101.9, soft), (102.0, sharp)])
        now = [100.0]

        def capture(camera):
            item = next(candidates, None)
            if item is None:
                raise KeyboardInterrupt
            now[0], image = item
            return {"frame_monotonic": now[0], "image": image}

        with patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(ci.vision, "capture_frame", side_effect=capture), \
             patch.object(ci.inference, "analyze_image", return_value=result("blocks")) as analyze:
            first, second = ci.iter_windows({}, {})
        self.assertEqual((first["frame_count"], first["inference_count"]), (4, 2))
        self.assertEqual(first["rejected_frames"], {"overexposed": 1})
        self.assertEqual((second["frame_count"], second["inference_count"]), (1, 1))
        calls = analyze.call_args_list
        self.assertIs(calls[0].args[0], sharp)
        self.assertIs(calls[1].args[0], soft)
        self.assertIs(calls[2].args[0], sharp)

    def test_archive_saves_each_rejection_and_winner_even_if_inference_fails(self):
        import cv2

        image = textured_frame()
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        self.assertTrue(ok)
        payload = jpeg.tobytes()
        now = [100.0]
        dark = np.zeros_like(image)
        bright = np.full_like(image, 255)
        flat = np.full_like(image, 128)
        # One mixed group, one all-rejected group and a partial group.
        def capture(camera):
            sample = next(samples, None)
            if sample is None:
                raise KeyboardInterrupt
            now[0] += 0.03
            return {"image": sample, "jpeg_bytes": payload if sample is image else None,
                    "frame_monotonic": now[0]}

        for options in ({}, {"save_sharp_frames": True}, {"save_rejected_frames": True},
                        {"save_sharp_frames": True, "save_rejected_frames": True}):
            sharp_enabled = options.get("save_sharp_frames", False)
            rejected_enabled = options.get("save_rejected_frames", False)
            now[0] = 100.0
            samples = iter([dark, image, bright, dark, flat, dark, image])
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(os.environ, {"SHARP_FRAME_DIR": tmp}), \
                 patch.object(ci.time, "monotonic", side_effect=lambda: now[0]), \
                 patch.object(ci.vision, "capture_frame", side_effect=capture), \
                 patch.object(ci.inference, "analyze_image", side_effect=RuntimeError("failed")):
                window, = ci.iter_windows({}, {}, **options)
                self.assertEqual(window["inference_count"], 0)
                counts = {"underexposed": 3, "overexposed": 1, "flat": 1}
                self.assertEqual(window["rejected_frames"], counts)
                for reason, count in counts.items():
                    rejected = list((Path(tmp) / "rejected" / reason).glob("*.jpg"))
                    self.assertEqual(len(rejected), count if rejected_enabled else 0)
                    for path in rejected:
                        self.assertEqual(ci.frame_problem(cv2.imread(str(path))), reason)
                files = list(Path(tmp).glob("*.jpg"))
                self.assertEqual(len(files), 2 if sharp_enabled else 0)
                for path in files:
                    self.assertEqual(path.suffix, ".jpg")
                    self.assertEqual(path.read_bytes(), payload)

    def test_same_frame_has_same_hash(self):
        img = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        self.assertEqual(ci.frame_stats(img)["hash"], ci.frame_stats(img.copy())["hash"])
        self.assertNotEqual(ci.frame_stats(img)["hash"], ci.frame_stats(255 - img)["hash"])

    def test_sampler_saves_first_rejected_per_window_and_periodic_ok(self):
        import json
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
