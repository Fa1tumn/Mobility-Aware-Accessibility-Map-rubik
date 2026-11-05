"""Area + Material segmentation adapter for the area-surface-v3 package.

Two LR-ASPP MobileNetV3 Large models run on the same frame in sequence:

    frame -> preprocess -> [Area backend] -> area logits
                        -> [Material backend] -> material logits
                        -> policy -> surface result

The three concerns are deliberately separated so the execution backend can be
swapped without touching image handling or the decision policy:

    * preprocess / postprocess / policy  -- pure NumPy, identical everywhere.
    * Backend                            -- runs one model's logits.
      - OnnxRuntimeBackend: CPU/PC verification without an NPU.
      - QnnBackend:         Rubik Pi NPU via qnn-net-run (added later).

The policy matches models/area-surface-v3/model-spec.json: a pixel is accepted
as a material observation only where the Area head predicts `sidewalk` with
confidence >= area_sidewalk_threshold AND the Material head's top class
confidence >= material_threshold. The dominant accepted material becomes
`surface_label`; if no pixel qualifies the label is `unknown`. Stairs and
damage heads are out of scope for this package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "area-surface-v3"
SPEC_PATH = MODEL_DIR / "model-spec.json"


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
    area = spec["models"]["area"]
    material = spec["models"]["material"]
    return (
        f"{spec['architecture']}:area_e{area.get('epoch', '?')}"
        f"+material_e{material.get('epoch', '?')}"
    )


def policy_version(spec: dict) -> str:
    policy = spec["policy"]
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


def apply_policy(area_logits: np.ndarray, material_logits: np.ndarray, spec: dict) -> dict:
    """Combine the two heads into one surface observation per the spec policy."""
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

    surface_label = "unknown"
    surface_confidence = None
    if material_accepted.any():
        accepted_class = material_class[material_accepted]
        accepted_conf = material_confidence[material_accepted]
        counts = np.bincount(accepted_class, minlength=len(material_labels))
        dominant = int(counts.argmax())
        surface_label = material_labels[dominant]
        # Mean confidence of the pixels backing the chosen material.
        surface_confidence = float(accepted_conf[accepted_class == dominant].mean())

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

    def __init__(self, model_path, providers=None):
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return np.asarray(outputs[0], dtype=np.float32)


class SurfaceModel:
    """Area + Material pipeline bound to a pair of backends and the spec."""

    def __init__(self, area_backend: Backend, material_backend: Backend, spec: dict):
        self.area_backend = area_backend
        self.material_backend = material_backend
        self.spec = spec

    @classmethod
    def from_onnxruntime(cls, model_dir=MODEL_DIR, providers=None) -> "SurfaceModel":
        model_dir = Path(model_dir)
        spec = load_spec(model_dir / "model-spec.json")
        area_path = model_dir / spec["models"]["area"]["path"]
        material_path = model_dir / spec["models"]["material"]["path"]
        return cls(
            OnnxRuntimeBackend(area_path, providers),
            OnnxRuntimeBackend(material_path, providers),
            spec,
        )

    def analyze(self, image) -> dict:
        """BGR frame -> combined surface observation (same frame, in sequence)."""
        input_tensor = preprocess(image, self.spec)
        area_logits = self.area_backend.run(input_tensor)
        material_logits = self.material_backend.run(input_tensor)
        return apply_policy(area_logits, material_logits, self.spec)

    def close(self) -> None:
        self.area_backend.close()
        self.material_backend.close()
