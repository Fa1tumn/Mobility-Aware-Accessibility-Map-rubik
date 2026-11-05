"""Surface inference entry point used by main.py.

This wraps the area-surface-v3 adapter (surface_model.py): Area + Material
LR-ASPP MobileNetV3 models run on the same frame in sequence, combined by the
spec policy. The execution backend is selectable:

    backend = "onnxruntime"  -> CPU/PC, verified against the reference outputs.
    backend = "qnn"          -> Rubik Pi NPU via qnn-net-run (added after the
                                 on-device SDK/runtime versions are confirmed).

main.py keeps its existing contract: load_models() once, then analyze_image()
per frame returning a dict. The returned dict now also carries material
confidence, the Area/Material acceptance ratios, and the model/policy versions.
Stairs and damage heads are out of scope for this package.
"""

from pathlib import Path

import surface_model

CONFIG = {
    "backend": "onnxruntime",                      # "onnxruntime" or "qnn"
    "model_dir": str(surface_model.MODEL_DIR),
    # onnxruntime backend
    "onnx_providers": None,                         # None -> ["CPUExecutionProvider"]
    # qnn backend (Rubik Pi); set once models are compiled to .dlc/.bin
    "qnn_area_model": None,
    "qnn_material_model": None,
}

# Surface material classes, kept for callers that referenced inference.LABELS.
LABELS = ["blocks", "cement", "asphalt", "urethane", "soil_stone", "other"]


def load_models() -> dict:
    """Build the Area+Material pipeline for the configured backend."""
    backend = CONFIG["backend"]
    model_dir = Path(CONFIG["model_dir"]).expanduser().resolve()

    if backend == "onnxruntime":
        model = surface_model.SurfaceModel.from_onnxruntime(
            model_dir=model_dir, providers=CONFIG["onnx_providers"],
        )
    elif backend == "qnn":
        # Deferred: QnnBackend is wired after on-device SDK/runtime checks and
        # NPU compilation. Kept explicit so misconfiguration fails loudly.
        raise NotImplementedError(
            "QNN backend is not connected yet. Compile the models to .dlc/.bin "
            "on the Pi and add QnnBackend before setting backend='qnn'."
        )
    else:
        raise ValueError(f"Unknown inference backend: {backend!r}")

    return {"surface": model}


def analyze_image(image, models: dict) -> dict:
    """BGR frame + loaded models -> combined surface observation dict."""
    return models["surface"].analyze(image)
