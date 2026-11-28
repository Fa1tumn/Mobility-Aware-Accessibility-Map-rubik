"""Area + Material segmentation adapter for the area-surface-v3 package.

Two LR-ASPP MobileNetV3 Large models run on the same frame in sequence:

    frame -> preprocess -> [Area backend] -> area logits
                        -> [Material backend] -> material logits
                        -> policy -> surface result

The three concerns are deliberately separated so the execution backend can be
swapped without touching image handling or the decision policy:

    * preprocess / postprocess / policy  -- pure NumPy, identical everywhere.
    * Backend                            -- runs one model's logits.
      - TorchBackend:       bundled .pt checkpoints on CPU or a torch device.
      - OnnxRuntimeBackend: CPU/PC verification without an NPU.
      - QnnBackend:         Rubik Pi NPU via qnn-net-run (added later).

The policy matches models/area-surface-v3/model-spec.json: a pixel is accepted
as a material observation only where the Area head predicts `sidewalk` with
confidence >= area_sidewalk_threshold AND the Material head's top class
confidence >= material_threshold. The dominant accepted material becomes
`surface_label`; if no pixel qualifies the label is `unknown`. Stairs and
damage heads are out of scope for this package.

Optional region policy (policy["region"], default "sidewalk"):

    "sidewalk"     -- the rule above (Area sidewalk >= threshold, whole frame).
    "wheel_ahead"  -- ignore the Area head and vote only over the lower-center
                      region the wheels are about to roll over: the bottom half
                      of the frame x the middle half of its width, the same
                      region camera_inference.road_sharpness() scores. Pixels
                      there count when Material confidence >= material_threshold.
                      The Area model is not run, area_accept_ratio is None, and
                      material_accept_ratio is the accepted share of that region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "area-surface-v3"
SPEC_PATH = MODEL_DIR / "model-spec.json"

REGION_SIDEWALK = "sidewalk"
REGION_WHEEL_AHEAD = "wheel_ahead"
REGIONS = (REGION_SIDEWALK, REGION_WHEEL_AHEAD)


def policy_region(spec: dict) -> str:
    region = spec["policy"].get("region", REGION_SIDEWALK)
    if region not in REGIONS:
        raise ValueError(f"Unknown policy region {region!r}; expected one of {REGIONS}.")
    return region


def wheel_ahead_slices(height: int, width: int) -> tuple:
    """Lower half x middle half of the frame (same as road_sharpness)."""
    return slice(height // 2, height), slice(width // 4, max(width // 4 + 1, 3 * width // 4))


def load_spec(spec_path: Path = SPEC_PATH) -> dict:
    spec_path = Path(spec_path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Model spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for key in ("architecture", "input_shape", "color_order", "normalization", "models", "policy"):
        if key not in spec:
            raise ValueError(f"model-spec.json is missing '{key}'.")
    if spec["color_order"].upper() != "RGB":
        raise ValueError("Adapter assumes RGB input; spec reports a different color order.")
    for task in ("area", "material"):
        if task not in spec["models"]:
            raise ValueError(f"model-spec.json has no '{task}' model entry.")
    return spec


def model_version(spec: dict) -> str:
    """Stable identifier stored alongside each record for traceability."""
    if spec.get("version"):
        return spec["version"]
    area = spec["models"]["area"]
    material = spec["models"]["material"]
    return (
        f"{spec['architecture']}:area_e{area.get('epoch', '?')}"
        f"+material_e{material.get('epoch', '?')}"
    )


def policy_version(spec: dict) -> str:
    policy = spec["policy"]
    if policy_region(spec) == REGION_WHEEL_AHEAD:
        return f"wheel_ahead(lower_half,center_half)&material>={policy['material_threshold']}"
    return (
        f"area_sidewalk>={policy['area_sidewalk_threshold']}"
        f"&material>={policy['material_threshold']}"
    )


# -- preprocessing ---------------------------------------------------------

def preprocess(image, spec: dict) -> np.ndarray:
    """BGR uint8 frame -> normalized float32 tensor [1, 3, H, W], RGB order.

    The training and reference-export pipeline resized with PIL's bilinear
    resampling, and this model is sensitive to the exact resize kernel:
    OpenCV's bilinear produces a materially different result (near-total
    'other' collapse). We therefore reproduce the PIL resize here to stay
    consistent with how the weights were trained. The frame is accepted as a
    BGR array (camera/cv2 convention) and converted to RGB without cv2.

    PIL is imported lazily so the policy/postprocess code can be unit tested
    without it present.
    """
    from PIL import Image

    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Image is empty. Check camera or image loading.")

    _, _, height, width = spec["input_shape"]
    # BGR (HWC) -> RGB (HWC) via channel reversal, then PIL bilinear resize.
    rgb_full = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    pil = Image.fromarray(rgb_full, mode="RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    rgb = np.asarray(pil)

    norm = spec["normalization"]
    scale = float(norm["scale"])
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(3, 1, 1)

    tensor = rgb.astype(np.float32) * scale          # to [0, 1]
    tensor = tensor.transpose(2, 0, 1)               # HWC -> CHW
    tensor = (tensor - mean) / std                   # ImageNet normalization
    return np.ascontiguousarray(tensor[None], dtype=np.float32)


# -- postprocessing / policy ----------------------------------------------

def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def decode_logits(logits: np.ndarray) -> tuple:
    """[1, C, H, W] logits -> (class_id [H, W], confidence [H, W]) via softmax."""
    probabilities = _softmax(np.asarray(logits, dtype=np.float32)[0], axis=0)
    class_id = probabilities.argmax(axis=0).astype(np.int32)
    confidence = probabilities.max(axis=0).astype(np.float32)
    return class_id, confidence


def _vote(material_class, material_confidence, accepted, material_labels):
    if not accepted.any():
        return "unknown", None
    accepted_class = material_class[accepted]
    accepted_conf = material_confidence[accepted]
    counts = np.bincount(accepted_class, minlength=len(material_labels))
    dominant = int(counts.argmax())
    # Mean confidence of the pixels backing the chosen material.
    return material_labels[dominant], float(accepted_conf[accepted_class == dominant].mean())


def apply_wheel_ahead_policy(material_logits: np.ndarray, spec: dict) -> dict:
    """Vote over the lower-center region only; the Area head is not used."""
    material_labels = spec["models"]["material"]["labels"]
    material_threshold = float(spec["policy"]["material_threshold"])
    material_class, material_confidence = decode_logits(material_logits)
    rows, cols = wheel_ahead_slices(*material_class.shape)
    region_class = material_class[rows, cols]
    region_conf = material_confidence[rows, cols]
    accepted = region_conf >= material_threshold
    surface_label, surface_confidence = _vote(region_class, region_conf, accepted, material_labels)
    return {
        "surface_label": surface_label,
        "surface_confidence": surface_confidence,
        "area_accept_ratio": None,
        "material_accept_ratio": float(accepted.mean()) if accepted.size else 0.0,
        "model_version": model_version(spec),
        "policy_version": policy_version(spec),
        "objects_supported": False,
        "objects": None,
    }


def apply_policy(area_logits, material_logits: np.ndarray, spec: dict) -> dict:
    """Combine the two heads into one surface observation per the spec policy."""
    if policy_region(spec) == REGION_WHEEL_AHEAD:
        return apply_wheel_ahead_policy(material_logits, spec)
    area_labels = spec["models"]["area"]["labels"]
    material_labels = spec["models"]["material"]["labels"]
    policy = spec["policy"]
    area_threshold = float(policy["area_sidewalk_threshold"])
    material_threshold = float(policy["material_threshold"])

    try:
        sidewalk_id = area_labels.index("sidewalk")
    except ValueError:
        raise ValueError("Area labels must include 'sidewalk' for the material policy.")

    area_class, area_confidence = decode_logits(area_logits)
    material_class, material_confidence = decode_logits(material_logits)
    total_pixels = int(area_class.size)

    # Area stage: pixels confidently predicted as sidewalk.
    area_accepted = (area_class == sidewalk_id) & (area_confidence >= area_threshold)
    area_accept_ratio = float(area_accepted.mean()) if total_pixels else 0.0

    # Material stage: within accepted sidewalk, keep confident material pixels.
    material_accepted = area_accepted & (material_confidence >= material_threshold)
    material_accept_ratio = float(material_accepted.mean()) if total_pixels else 0.0

    surface_label, surface_confidence = _vote(
        material_class, material_confidence, material_accepted, material_labels)

    return {
        "surface_label": surface_label,
        "surface_confidence": surface_confidence,
        "area_accept_ratio": area_accept_ratio,
        "material_accept_ratio": material_accept_ratio,
        "model_version": model_version(spec),
        "policy_version": policy_version(spec),
        "objects_supported": False,
        "objects": None,
    }


# -- execution backends ----------------------------------------------------

class Backend:
    """Runs one model and returns its raw logits [1, C, H, W]."""

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OnnxRuntimeBackend(Backend):
    """CPU (or any ORT provider) execution for PC-side verification."""

    def __init__(self, model_path, providers=None, *, session_options=None):
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return np.asarray(outputs[0], dtype=np.float32)


class TorchBackend(Backend):
    """Run one bundled LR-ASPP checkpoint without downloading backbone weights."""

    def __init__(self, model_path, labels, task, device="cpu"):
        import torch
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        from numpy._core.multiarray import _reconstruct as numpy_reconstruct

        # NumPy 1.26 exposes this import but reports numpy.core in __module__.
        # These checkpoints record numpy._core (NumPy 2). PyTorch 2.5 keys its
        # allowlist by __module__/__name__ and cannot accept explicit-name tuples.
        # A local wrapper maps that exact checkpoint name to the installed
        # NumPy implementation without changing NumPy's function globally.
        def _reconstruct(*args):
            return numpy_reconstruct(*args)

        _reconstruct.__module__ = "numpy._core.multiarray"

        self.torch = torch
        self.device = torch.device(device)
        # The training checkpoints include NumPy RNG state. Allow only the
        # required array types; keep PyTorch's restricted weights-only loader.
        with torch.serialization.safe_globals([
            _reconstruct, numpy_reconstruct, np.ndarray, np.dtype,
            type(np.dtype("uint32")),
        ]):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("names") != labels:
            raise ValueError(f"{task} checkpoint labels differ from model-spec.json")
        if checkpoint.get("config", {}).get("task") != task:
            raise ValueError(f"Expected a {task} checkpoint: {model_path}")
        self.epoch = checkpoint["epoch"]
        self.model = lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=len(labels),
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(input_tensor).to(self.device)
            return self.model(tensor)["out"].cpu().numpy()

    def close(self) -> None:
        self.model = None


class SurfaceModel:
    """Area + Material pipeline bound to a pair of backends and the spec."""

    def __init__(self, area_backend: Backend, material_backend: Backend, spec: dict):
        self.area_backend = area_backend
        self.material_backend = material_backend
        self.spec = spec

    @classmethod
    def from_onnxruntime(cls, model_dir=MODEL_DIR, providers=None, *, session_options=None) -> "SurfaceModel":
        model_dir = Path(model_dir)
        spec = load_spec(model_dir / "model-spec.json")
        area_path = model_dir / spec["models"]["area"]["path"]
        material_path = model_dir / spec["models"]["material"]["path"]
        return cls(
            OnnxRuntimeBackend(area_path, providers, session_options=session_options),
            OnnxRuntimeBackend(material_path, providers, session_options=session_options),
            spec,
        )

    @classmethod
    def from_checkpoints(cls, model_dir=MODEL_DIR, device="cpu", threads=2) -> "SurfaceModel":
        import torch

        if threads < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(threads)
        model_dir = Path(model_dir)
        spec = load_spec(model_dir / "model-spec.json")
        if spec["architecture"] != "lraspp_mobilenet_v3_large":
            raise ValueError("Unsupported checkpoint architecture")
        backends = {}
        try:
            for task in ("area", "material"):
                backends[task] = TorchBackend(
                    model_dir / "checkpoints" / f"{task}-best.pt",
                    spec["models"][task]["labels"], task, device,
                )
                spec["models"][task]["epoch"] = backends[task].epoch
        except BaseException:
            for backend in backends.values():
                backend.close()
            raise
        return cls(backends["area"], backends["material"], spec)

    def analyze(self, image, timing: bool = False) -> dict:
        """BGR frame -> combined surface observation (same frame, in sequence).

        When timing is True, the result carries a 'timing_ms' dict with the
        preprocess / area / material / postprocess / total wall-clock split,
        for measuring the on-device CPU cost of the two-model pipeline.
        With policy region "wheel_ahead" the Area model is skipped (area = 0 ms).
        """
        use_area = policy_region(self.spec) != REGION_WHEEL_AHEAD
        if not timing:
            input_tensor = preprocess(image, self.spec)
            area_logits = self.area_backend.run(input_tensor) if use_area else None
            material_logits = self.material_backend.run(input_tensor)
            return apply_policy(area_logits, material_logits, self.spec)

        from time import perf_counter

        t0 = perf_counter()
        input_tensor = preprocess(image, self.spec)
        t1 = perf_counter()
        area_logits = self.area_backend.run(input_tensor) if use_area else None
        t2 = perf_counter()
        material_logits = self.material_backend.run(input_tensor)
        t3 = perf_counter()
        result = apply_policy(area_logits, material_logits, self.spec)
        t4 = perf_counter()
        result["timing_ms"] = {
            "preprocess": round((t1 - t0) * 1000, 1),
            "area": round((t2 - t1) * 1000, 1),
            "material": round((t3 - t2) * 1000, 1),
            "postprocess": round((t4 - t3) * 1000, 1),
            "total": round((t4 - t0) * 1000, 1),
        }
        return result

    def close(self) -> None:
        self.area_backend.close()
        self.material_backend.close()
