"""Wheel-ahead region policy: synthetic logits and fake backends; no models/hardware."""
import copy
import unittest

import numpy as np

import surface_model
from surface_model import SurfaceModel, apply_policy, wheel_ahead_slices

H, W = 360, 640
AREA = ["unannotated_background", "sidewalk", "roadway", "alley", "bike_lane", "braille_guide_blocks", "caution_zone"]
MATERIAL = ["blocks", "cement", "asphalt", "urethane", "soil_stone", "other"]
SPEC = {
    "architecture": "lraspp_mobilenet_v3_large", "input_shape": [1, 3, H, W], "color_order": "RGB",
    "normalization": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "models": {"area": {"labels": AREA, "epoch": 7}, "material": {"labels": MATERIAL, "epoch": 5}},
    "policy": {"area_sidewalk_threshold": 0.8, "material_threshold": 0.8},
}


def logits(class_map, n, strength=10.0):
    out = np.zeros((1, n, H, W), dtype=np.float32)
    for c in range(n):
        out[0, c][class_map == c] = strength
    return out


def spec(region=None):
    s = copy.deepcopy(SPEC)
    if region:
        s["policy"]["region"] = region
    return s


class FakeBackend:
    def __init__(self, out):
        self.out, self.calls = out, 0

    def run(self, tensor):
        self.calls += 1
        return self.out

    def close(self):
        pass


def red_lane_scene():
    """Asphalt under the wheels (lower-center), blocks everywhere else.

    The Area head is confident sidewalk only on the blocks, as on the 11/25
    dark red lane, so the whole-frame sidewalk policy reports blocks."""
    material = np.zeros((H, W), dtype=np.int64)          # blocks
    rows, cols = wheel_ahead_slices(H, W)
    material[rows, cols] = MATERIAL.index("asphalt")
    area = np.full((H, W), AREA.index("sidewalk"), dtype=np.int64)
    area[rows, cols] = AREA.index("bike_lane")
    return logits(area, len(AREA)), logits(material, len(MATERIAL))


class WheelAheadTests(unittest.TestCase):
    def test_region_matches_road_sharpness(self):
        rows, cols = wheel_ahead_slices(720, 1280)
        self.assertEqual((rows.start, rows.stop, cols.start, cols.stop), (360, 720, 320, 960))

    def test_default_policy_unchanged(self):
        area, material = red_lane_scene()
        result = apply_policy(area, material, spec())
        self.assertEqual(result["surface_label"], "blocks")
        self.assertEqual(result["policy_version"], "area_sidewalk>=0.8&material>=0.8")
        self.assertIsNotNone(result["area_accept_ratio"])

    def test_wheel_ahead_votes_only_lower_center(self):
        area, material = red_lane_scene()
        result = apply_policy(area, material, spec("wheel_ahead"))
        self.assertEqual(result["surface_label"], "asphalt")
        self.assertIsNone(result["area_accept_ratio"])
        self.assertAlmostEqual(result["material_accept_ratio"], 1.0, places=6)
        self.assertEqual(result["policy_version"], "wheel_ahead(lower_half,center_half)&material>=0.8")

    def test_wheel_ahead_accepts_no_area_logits(self):
        _, material = red_lane_scene()
        self.assertEqual(apply_policy(None, material, spec("wheel_ahead"))["surface_label"], "asphalt")

    def test_wheel_ahead_low_confidence_is_unknown(self):
        _, material = red_lane_scene()
        result = apply_policy(None, material * 0.0, spec("wheel_ahead"))   # uniform -> conf 1/6
        self.assertEqual(result["surface_label"], "unknown")
        self.assertIsNone(result["surface_confidence"])
        self.assertEqual(result["material_accept_ratio"], 0.0)

    def test_unknown_region_rejected(self):
        _, material = red_lane_scene()
        with self.assertRaises(ValueError):
            apply_policy(None, material, spec("whole_frame"))

    def test_analyze_skips_area_model_in_wheel_ahead(self):
        area, material = red_lane_scene()
        for region, area_calls, label in ((None, 1, "blocks"), ("wheel_ahead", 0, "asphalt")):
            area_backend, material_backend = FakeBackend(area), FakeBackend(material)
            model = SurfaceModel(area_backend, material_backend, spec(region))
            result = model.analyze(np.zeros((H, W, 3), dtype=np.uint8), timing=True)
            self.assertEqual(result["surface_label"], label)
            self.assertEqual(area_backend.calls, area_calls)
            self.assertEqual(material_backend.calls, 1)

    def test_inference_env_override(self):
        import inference
        from unittest.mock import patch
        area, material = red_lane_scene()
        fake = SurfaceModel(FakeBackend(area), FakeBackend(material), spec())
        with patch.dict(inference.CONFIG, {"backend": "onnxruntime", "surface_region": "wheel_ahead"}), \
             patch.object(surface_model.SurfaceModel, "from_onnxruntime", return_value=fake):
            models = inference.load_models()
        self.assertEqual(models["surface"].spec["policy"]["region"], "wheel_ahead")
        with patch.dict(inference.CONFIG, {"backend": "onnxruntime", "surface_region": "typo"}):
            with self.assertRaises(ValueError):
                inference.load_models()


if __name__ == "__main__":
    unittest.main()
