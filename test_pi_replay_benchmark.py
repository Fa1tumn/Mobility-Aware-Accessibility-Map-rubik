"""pi_replay_benchmark: 합성 녹화와 가짜 모델로 사진 수 계산과 출력 형식을 확인한다(모델·카메라 불필요)."""
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import pi_replay_benchmark as bench

FPS = 10
SECONDS = 3.0


class FakeSurface:
    """한 번 분석에 delay_s만큼 걸리는 가짜 모델."""

    def __init__(self, delay_s=0.03):
        self.delay_s = delay_s
        self.calls = 0

    def analyze(self, image, timing=False):
        self.calls += 1
        t0 = time.perf_counter()
        time.sleep(self.delay_s)
        total = (time.perf_counter() - t0) * 1000
        result = {"surface_label": "blocks", "surface_confidence": 0.9}
        if timing:
            result["timing_ms"] = {"preprocess": 1.0, "area": 1.0, "material": 1.0, "postprocess": 1.0,
                                   "total": round(total, 1)}
        return result

    def close(self):
        pass


def make_recording(root):
    """밝기와 무늬가 있는 사진을 FPS장/초로 SECONDS초 동안 만든 녹화 폴더."""
    (root / "frames").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(int(FPS * SECONDS)):
        image = rng.integers(40, 200, size=(72, 128, 3), dtype=np.uint8)
        name = f"frames/{i:07d}.jpg"
        cv2.imwrite(str(root / name), image)
        rows.append({"t_s": i / FPS + 0.01, "frame_time": "", "file": name})
    with (root / "frames.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["t_s", "frame_time", "file"])
        writer.writeheader()
        writer.writerows(rows)
    with (root / "imu.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["t_s", "gx", "gy", "gz"])
        writer.writeheader()
        for k in range(int(SECONDS * 100)):
            writer.writerow({"t_s": k / 100, "gx": 0.01, "gy": 0.0, "gz": 0.0})
    (root / "session.json").write_text(json.dumps({"mode": "route", "status": "finished", "duration_s": SECONDS,
                                                   "target_fps": FPS, "start_utc": "2026-11-27T00:00:00Z"}))
    return bench.read_rows(root / "frames.csv")


class ReplayBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recording = Path(self.tmp.name) / "rec"
        self.rows = make_recording(self.recording)
        self.models = {"surface": FakeSurface()}
        self.meter, self.sampler = bench.Meter(), bench.SystemSampler(0.2)
        self.meter.patch()
        self.sampler.start()

    def tearDown(self):
        self.sampler.stop()
        self.meter.unpatch()
        self.tmp.cleanup()

    def run_policy(self, policy, interval_s=1.0):
        gyro = bench.gyro_lookup(self.recording) if policy == "quality" else None
        return bench.measure(self.recording, self.rows, self.models, self.meter, self.sampler, policy=policy,
                             interval_s=interval_s, start_s=0.0, duration_s=2.0, source_fps=None, warmup=1,
                             gyro_rms=gyro)

    def check_counts(self, windows, summary):
        self.assertEqual(len(windows), 2)
        for w in windows:
            self.assertEqual(w["unprocessed"], w["delivered"] - w["inferred"])
            self.assertLessEqual(w["read"], w["delivered"])
            self.assertGreaterEqual(w["lag_ms"], 0)
        # 측정 2초 동안 10장/초 → 구간 경계 근처 한두 장 차이만 허용
        self.assertAlmostEqual(summary["frames"]["delivered"], 20, delta=2)
        self.assertEqual(summary["frames"]["unprocessed"],
                         summary["frames"]["delivered"] - summary["frames"]["inferred"])

    def test_timed_one_inference_per_window(self):
        windows, inferences, _, summary = self.run_policy("timed")
        self.check_counts(windows, summary)
        self.assertEqual([w["inferred"] for w in windows], [1, 1])
        self.assertEqual(len(inferences), 2)                 # 워밍업 호출은 빠진다
        self.assertTrue(all(w["file"].startswith("frames/") for w in windows))
        self.assertTrue(summary["fits_interval"])
        self.assertEqual(summary["stage_ms"]["area"]["mean"], 1.0)

    def test_quality_measures_checks(self):
        windows, _, _, summary = self.run_policy("quality")
        self.check_counts(windows, summary)
        self.assertEqual([w["inferred"] for w in windows], [1, 1])
        self.assertGreater(summary["quality_check_ms_per_call"]["n"], 0)

    def test_live_uses_main_pipeline(self):
        windows, _, _, summary = self.run_policy("live")
        self.check_counts(windows, summary)
        # iter_windows는 3장마다 한 번 분석한다
        self.assertTrue(all(w["inferred"] >= w["read"] // 3 for w in windows))
        self.assertLess(summary["result_lag_ms"]["max"], 500)   # 마지막 구간이 사진을 기다리며 멈추지 않음

    def test_slow_model_drops_frames_and_misses_interval(self):
        self.models["surface"].delay_s = 0.6
        windows, _, _, summary = self.run_policy("timed", interval_s=0.5)
        self.assertGreater(summary["frames"]["dropped_unread"], 0)
        self.assertFalse(summary["fits_interval"])

    def test_live_load_runs_devices_and_reports(self):
        import queue
        import sys
        import threading
        import types
        closed = []

        def open_camera(_config):
            cam = {"frames": queue.Queue(maxsize=1), "error": None}

            def feed():
                while "camera" not in closed:
                    try:
                        cam["frames"].put({"image": None}, timeout=0.1)
                    except queue.Full:
                        pass
                    time.sleep(1 / 30)
            threading.Thread(target=feed, daemon=True).start()
            return cam
        fake_imu = types.SimpleNamespace(start_imu=lambda: {"last_sample_time_s": 1.0},
                                         stop_imu=lambda s: closed.append("imu"))
        fake_phone = types.SimpleNamespace(start_phone=lambda: (_ for _ in ()).throw(OSError("포트 사용 중")),
                                           stop_phone=lambda s: closed.append("gps"))
        originals = (bench.vision.open_camera, bench.vision.close_camera, bench.vision.load_camera_config)
        bench.vision.open_camera = open_camera
        bench.vision.close_camera = lambda c: closed.append("camera")
        bench.vision.load_camera_config = lambda: {}
        sys.modules["imu"], sys.modules["phone"] = fake_imu, fake_phone
        try:
            load = bench.LiveLoad()
            load.start()
            windows, _, _, summary = bench.measure(
                self.recording, self.rows, self.models, self.meter, self.sampler, policy="timed", interval_s=1.0,
                start_s=0.0, duration_s=2.0, source_fps=None, warmup=1, gyro_rms=None, live_load=load)
            load.stop()
        finally:
            bench.vision.open_camera, bench.vision.close_camera, bench.vision.load_camera_config = originals
            sys.modules.pop("imu"), sys.modules.pop("phone")
        ll = summary["live_load"]
        self.assertGreater(ll["camera_fps"], 10)            # 가짜 카메라 30fps를 실제로 비웠다
        self.assertTrue(ll["imu_receiving"])
        self.assertFalse(ll["gps_receiving"])
        self.assertIn("gps", ll["errors"])                   # 켜지 못한 장비는 오류로 남고 측정은 계속
        self.assertEqual(sorted(closed), ["camera", "imu"])
        self.assertEqual([w["inferred"] for w in windows], [1, 1])

    def test_markdown_and_pc_comparison(self):
        _, _, _, summary = self.run_policy("timed")
        results = {"runs": [{"method": "A", "interval_s": 1.0, "windows": [
            {"index": 0, "frame": {"file": "frames/0000009.jpg"}, "prediction": {"surface_label": "blocks"}},
            {"index": 1, "frame": {"file": "frames/0000000.jpg"}, "prediction": {"surface_label": "cement"}}]}]}
        (self.recording / "four_method_replay").mkdir()
        (self.recording / "four_method_replay" / "results.json").write_text(json.dumps(results))
        windows = [{"index": 0, "file": "frames/0000009.jpg", "surface_label": "blocks"},
                   {"index": 1, "file": "frames/0000019.jpg", "surface_label": "blocks"}]
        pc = bench.compare_with_pc(self.recording, "timed", 1.0, 0.0, windows, "models/area-surface-v3")
        self.assertEqual(pc, {"pc_method": "A", "compared_windows": 2, "same_frame": 1, "same_label": 1,
                              "same_label_when_same_frame": 1})
        summary["pc_comparison"] = pc
        report = {"recording": "rec", "started_utc": "t", "start_s": 0, "duration_s": 2, "source_fps": None,
                  "cooldown_s": 0, "runs": [summary],
                  "environment": {"hostname": "h", "machine": "m", "cpu_count": 1, "backend": "torch",
                                  "torch_threads": 2, "model_dir": "d", "surface_region": None}}
        text = bench.markdown(report)
        self.assertIn("| timed | 1 | 2 | 1.00 |", text)
        self.assertIn("## PC replay와 비교", text)


if __name__ == "__main__":
    unittest.main()
