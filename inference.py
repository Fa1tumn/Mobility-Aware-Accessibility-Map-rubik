from pathlib import Path
import shutil
import subprocess
import tempfile

CONFIG = {
    "seg_model_path": None,     # QNN .dlc or HTP .bin from the raw-classes export.
    # "object_model_path": None,
    "confidence": 0.25,
    "nms_iou": 0.45,
}

# Fixed contract for export_best_qnn.py --raw-classes.
LABELS = ["blocks", "cement", "asphalt", "urethane", "soil_stone", "other", "damaged"]
OUTPUT_SHAPES = {
    "boxes": (1, 8400, 4),              # Center x/y, width/height in 640x640 input pixels.
    "class_scores": (1, 8400, 7),       # Class probabilities in [0, 1].
    "mask_coefficients": (1, 8400, 32),
    "prototypes": (1, 160, 160, 32),    # NHWC mask prototypes.
}

# Class order must match the trained detector.
# OBJECT_LABELS = ["vehicle", "construction", "tent"]
# OBJECT_OUTPUT_SHAPES = {
#     "boxes": (1, 8400, 4),
#     "class_scores": (1, 8400, 3),
# }


def load_models() -> dict:
    """Validate model paths; qnn-net-run loads them on each inference."""
    if CONFIG["seg_model_path"] is None:
        raise ValueError("Set seg_model_path to a compatible QNN model (.dlc or .bin).")

    model_path = Path(CONFIG["seg_model_path"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"NPU model not found: {model_path}")
    if model_path.suffix not in (".dlc", ".bin"):
        raise ValueError("Surface model must be a compatible QNN .dlc or cached .bin.")
    models = {"seg_model": model_path, "object_model": None}
    # if CONFIG["object_model_path"] is not None:
    #     models["object_model"] = Path(CONFIG["object_model_path"]).expanduser().resolve()

    if shutil.which("qnn-net-run") is None:
        raise RuntimeError("qnn-net-run not found. Check the Pi's QNN environment.")
    if not Path("/usr/lib/libQnnHtp.so").is_file():
        raise RuntimeError("libQnnHtp.so not found. NPU inference is unavailable.")
    if model_path.suffix == ".dlc" and not Path("/usr/lib/libQnnModelDlc.so").is_file():
        raise RuntimeError("libQnnModelDlc.so not found. Cannot load DLC models.")
    return models


def analyze_image(image, models: dict) -> dict:
    if image is None or image.size == 0:
        raise ValueError("Image is empty. Check camera or image loading.")

    outputs = _run_npu(image, models["seg_model"], OUTPUT_SHAPES)
    surface_label, surface_confidence = _get_surface(outputs)

    objects_supported = False
    objects = None
    # objects_supported = models["object_model"] is not None
    # if objects_supported:
    #     object_outputs = _run_npu(image, models["object_model"], OBJECT_OUTPUT_SHAPES)
    #     objects = _get_objects(object_outputs)

    result = {
        "surface_label": surface_label,
        "surface_confidence": surface_confidence,
        "objects_supported": objects_supported,
        "objects": objects,
    }
    return result


def _run_npu(image, model_path, output_shapes) -> dict:
    """Both models expect RGB float32 input in [0, 1], shaped [1, 3, 640, 640]."""
    import cv2
    import numpy as np

    resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    with tempfile.TemporaryDirectory(prefix="rubik-npu-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / "image.raw"
        tensor.tofile(input_path)
        input_list = work_dir / "inputs.txt"
        input_list.write_text(f"images:={input_path}\n", encoding="utf-8")
        output_dir = work_dir / "results"

        model_path = Path(model_path)
        command = ["qnn-net-run", "--backend", "/usr/lib/libQnnHtp.so"]
        if model_path.suffix == ".dlc":
            command += ["--model", "/usr/lib/libQnnModelDlc.so", "--dlc_path", str(model_path)]
        elif model_path.suffix == ".bin":
            command += ["--retrieve_context", str(model_path)]
        else:
            raise ValueError("NPU inference accepts only .dlc or .bin models.")
        command += [
            "--input_list", str(input_list), "--output_dir", str(output_dir),
            "--perf_profile", "high_performance",
        ]

        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0:
            message = (completed.stdout + completed.stderr)[-2000:]
            raise RuntimeError(f"NPU inference failed with exit code {completed.returncode}:\n{message}")

        outputs = {}
        for output_index, (name, shape) in enumerate(output_shapes.items()):
            paths = list(output_dir.rglob(f"{name}.raw"))

            if not paths:
                paths = list(output_dir.rglob(f"output_{output_index}.raw"))
            if len(paths) != 1:
                raise RuntimeError(f"Expected one {name}.raw file, found {len(paths)}.")
            values = np.fromfile(paths[0], dtype=np.float32)
            if values.size != np.prod(shape) or not np.isfinite(values).all():
                raise RuntimeError(f"{name} output has an unexpected shape or contains non-finite values.")
            outputs[name] = values.reshape(shape)
        return outputs


def _get_candidates(outputs) -> tuple:
    import cv2
    import numpy as np

    boxes = outputs["boxes"][0]
    class_scores = outputs["class_scores"][0]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    kept = []
    for class_id in range(class_scores.shape[1]):
        indices = np.flatnonzero(
            (class_ids == class_id) & (scores > CONFIG["confidence"])
            & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        )
        if len(indices) == 0:
            continue
        selected = boxes[indices].copy()
        selected[:, :2] -= selected[:, 2:] / 2  # Convert center x/y to top-left x/y for NMS.
        keep = cv2.dnn.NMSBoxes(
            selected.tolist(), scores[indices].tolist(),
            CONFIG["confidence"], CONFIG["nms_iou"],
        )
        for position in np.asarray(keep).reshape(-1):
            kept.append(int(indices[position]))
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept[:300], scores, class_ids


def _get_surface(outputs) -> tuple:
    import cv2
    import numpy as np

    boxes = outputs["boxes"][0]
    coefficients = outputs["mask_coefficients"][0]
    prototypes = outputs["prototypes"][0]
    kept, scores, class_ids = _get_candidates(outputs)

    surface_label = "unknown"
    surface_confidence = None
    largest_area = 0
    rows, columns = np.ogrid[:160, :160]
    for index in kept:
        mask = prototypes @ coefficients[index]
        center_x, center_y, width, height = boxes[index] / 4  # Scale input coordinates to the 160x160 prototype grid.
        x1, x2 = center_x - width / 2, center_x + width / 2
        y1, y2 = center_y - height / 2, center_y + height / 2
        inside_box = (columns >= x1) & (columns < x2) & (rows >= y1) & (rows < y2)
        mask = np.where(inside_box, mask, 0)
        mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)
        # sigmoid(mask) > 0.5 is equivalent to mask > 0.
        area = int((mask > 0).sum())
        if area > largest_area:
            largest_area = area
            surface_label = LABELS[int(class_ids[index])]
            surface_confidence = float(scores[index])
    return surface_label, surface_confidence


# def _get_objects(outputs) -> list:
#     kept, scores, class_ids = _get_candidates(outputs)
#     objects = []
#     for index in kept:
#         objects.append({
#             "label": OBJECT_LABELS[int(class_ids[index])],
#             "confidence": float(scores[index]),
#         })
#     return objects
