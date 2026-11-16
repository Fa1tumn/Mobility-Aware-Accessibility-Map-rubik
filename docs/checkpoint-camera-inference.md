# Dual-checkpoint camera inference and two-second aggregation

The default backend loads `models/area-surface-v3/checkpoints/area-best.pt` and
`material-best.pt` directly and runs PyTorch inference on the CPU. Both models
process the same camera image using the model package's RGB conversion,
PIL bilinear resize to 640×360, ImageNet normalization, and threshold policy.
Only pixels with an Area sidewalk confidence of at least 0.8 and a Material
confidence of at least 0.8 contribute to the surface classification.

## Setup and execution

On Rubik Pi, use a GStreamer environment that supports `qtiqmmfsrc`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-checkpoints.txt
# The full pipeline also requires bleak for the BLE IMU:
.venv/bin/pip install bleak

# Check the camera and both models; stop after ten windows.
# This does not use GPS, IMU, the database, or cloud uploads.
.venv/bin/python camera_inference.py --windows 10 --timing --output /tmp/surface-windows.jsonl

# Full collection: save each window to SQLite using main.py's Supabase configuration.
.venv/bin/python main.py

# Collect locally with uploads explicitly disabled.
SUPABASE_URL='' .venv/bin/python main.py
```

Without `--output`, `camera_inference.py` writes one JSON object per line to stdout.
`--camera-id 0` selects the CSI camera; `--window-seconds 2` sets the standalone
script's window duration; `--windows 0` runs continuously. Diagnostics go to
stderr. In `main.py`, `CONFIG["interval_s"]` sets the window duration and defaults
to 2 seconds. If you change it, adjust the IMU's `window_s` to match.

CPU inference uses two PyTorch compute threads by default. Set `TORCH_THREADS`
to change this; `TORCH_DEVICE` defaults to `cpu`. The existing ONNX backend
remains available:

```bash
.venv/bin/python camera_inference.py --backend onnxruntime --windows 10 --timing
INFERENCE_BACKEND=onnxruntime .venv/bin/python main.py
```

Weights use PyTorch's restricted `weights_only=True` loader, with additional
array types allowlisted for the NumPy RNG state stored in the checkpoints.
Class order, task type, and model architecture are validated; model epochs come
from the actual checkpoints. No pretrained backbone download or NPU is used.
References: [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
and [torchvision LR-ASPP API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html).

## Timing and aggregation rules

- Starting at collection startup, the monotonic clock defines consecutive,
  non-overlapping two-second `[start, end)` windows. Frame membership uses the
  camera decode time, not the exposure time or inference completion time.
- Inference processes the latest frames as often as possible within each window.
  The camera queue holds only one frame, dropping older frames when the CPU
  cannot keep up. The actual frame count depends on device performance;
  processing every frame of a 30 FPS camera stream is not guaranteed.
- Each successful inference casts one vote, including `unknown`. Ties are broken
  by summed confidence, then by lexicographic label order. The output confidence
  is the mean confidence of frames voting for the winning class. Both pixel
  acceptance ratios are averaged across all successful frames.
- Failed inference casts no vote. Visual fields are NULL when no inference
  succeeds, and the label is `unknown` when inference succeeds without eligible
  pixels. Empty windows still produce a record to distinguish downtime from
  normal operation without a recognition result.
- `record_time` is the window end in UTC. GPS selects the sample from its history
  closest to the window end, requiring a time difference of at most 1 second and
  reported accuracy of at most 10 meters. IMU analysis uses the two seconds of
  samples ending at the same boundary.
- In-flight inference can finish before its window is delivered. Slow inference
  or synchronous uploads can delay delivery and reduce subsequent frame counts;
  old frames are never moved into a new window. Long stalls can exhaust sensor
  history buffers, leaving the corresponding fields NULL.
- Pressing Ctrl+C during capture or inference flushes completed results from the
  current window, so the final window may be shorter than two seconds.
  Interruptions during saving or uploading follow the main program's existing
  shutdown and cleanup behavior.

SQLite automatically adds `window_start_time`, `window_end_time`, `frame_count`,
and `inference_count`. The first two store window boundaries; the latter two
store the number of attempted frames and successful inferences, respectively.
These diagnostic fields stay local. Uploads retain the existing observation
fields, so this change requires no Supabase table migration. `timing_ms` reports
mean inference durations across successful frames in a window and is written
to logs or the standalone JSONL output.

## Validation

```bash
.venv/bin/python -m unittest test_camera_inference test_main_failures test_imu_slope test_upload_supabase
.venv/bin/python -c 'import test_pipeline; test_pipeline.run_test("/tmp/pipeline-test.sqlite3")'
```

Software tests use simulated clocks and devices with temporary SQLite databases;
they do not upload to the real Supabase service.
In a brief device test on 2026-11-16, three two-second windows completed inference
on 1, 3, and 2 frames, respectively. Mean per-frame inference times for both models
were approximately 1.03, 0.89, and 0.86 seconds in those windows. This test confirms
that the pipeline runs; it does not establish surface classification accuracy or
long-term thermal stability. Comparing random-input logits from the two
checkpoints against the existing ONNX models gave maximum absolute differences
of approximately `8.6e-6` and `7.6e-6`, respectively.

During the same device test, `main.py` saved three windows containing 2, 2, and 3
frames to a temporary SQLite database with cloud uploads explicitly disabled.
All 22 unit regression tests and the full software integration test passed.
The IMU connection timed out after installing bleak, and no phone GPS data was
received, so joint collection with real sensors still requires verification once
the devices are connected. Missing sensors do not prevent visual results from
being saved.
