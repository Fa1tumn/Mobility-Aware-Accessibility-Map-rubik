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

### Wheel-ahead region (optional)

`SURFACE_REGION=wheel_ahead` judges the material only in the region the wheels
are about to roll over: the bottom half of the frame x the middle half of its
width, the same region used for sharpness. The Area model is not run, so
`area_accept_ratio` is stored as null, `material_accept_ratio` is the confident
share of that region, and `policy_version` becomes
`wheel_ahead(lower_half,center_half)&material>=0.8`. Unset (default) keeps the
whole-frame sidewalk rule.

```bash
SURFACE_REGION=wheel_ahead INFERENCE_BACKEND=onnxruntime MODEL_DIR="$PWD/models/finetuned-v3-onnx" .venv/bin/python main.py
```

On the 11/25 drive (2-second records, one fixed frame each), the 2nd-round
material model scored 340/343 blocks and 18/69 non-block records with this
region, against 323/343 and 10/69 with the whole-frame sidewalk rule. On a PC
CPU the per-frame time fell from 132 ms to 70 ms; Pi timing is not yet measured.

Weights use PyTorch's restricted `weights_only=True` loader, with additional
array types allowlisted for the NumPy RNG state stored in the checkpoints.
The loader also maps the checkpoints' NumPy 2 `numpy._core` reconstruction
name to the installed NumPy implementation, supporting existing NumPy 1.26 /
PyTorch 2.5 device environments without disabling the restricted loader.
Class order, task type, and model architecture are validated; model epochs come
from the actual checkpoints. No pretrained backbone download or NPU is used.
References: [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
and [torchvision LR-ASPP API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html).

## Road capture defaults

The CSI camera uses center-weighted metering and its factory automatic exposure
curve for collection and snapshots. The camera adjusts shutter time and gain as
lighting changes. Longer exposures in dim light can increase motion blur.

Edit [camera_config.json](../camera_config.json) and restart collection to apply
changes. Set `max_exposure_us` to `0` for the factory automatic exposure curve,
or to a value from 11 to 33333 for a shutter cap in microseconds (4000 is 4 ms).
A positive cap also applies in low light. `exposure_metering` accepts
`center-weighted` or `average`. `exposure_compensation` accepts integer steps
from -12 to 12. The environment variables `CAMERA_MAX_EXPOSURE_US`,
`CAMERA_EXPOSURE_METERING` and `CAMERA_EXPOSURE_COMPENSATION` override the
corresponding JSON values.

Set `save_sharp_frames` and `save_rejected_frames` independently to JSON
`true` or `false`. Both ship disabled. Both collection entry points accept
per-run overrides, including `--no-save-sharp-frames` and
`--no-save-rejected-frames`.

- `--save-sharp-frames` saves the full JPEG selected from each group to
  `sharp_frames/` before inference. Filenames contain the UTC decode time and
  lower-center road sharpness score.
- `--save-rejected-frames` saves each frame that fails the quality check to
  `sharp_frames/rejected/<reason>/`. Reasons are `underexposed`, `overexposed`,
  and `flat`.

Set `sharp_frame_dir` to choose the archive root. Relative JSON paths resolve
from the config file. `SHARP_FRAME_DIR` overrides this path for a run. The separate diagnostic sampler is enabled with `FRAME_SAMPLE_DIR`.

## Timing and aggregation rules

- Starting at collection startup, the monotonic clock defines consecutive,
  non-overlapping two-second `[start, end)` windows. Frame membership uses the
  camera decode time, not the exposure time or inference completion time.
- Each group of three observed frames in the same window contributes up to one
  full image for inference. After the full-frame brightness and contrast checks,
  the frame with the strongest mean squared Sobel gradient (Tenengrad) in the
  lower-center road region wins. The region spans the middle half of the width
  and the bottom half of the height. Ties select the newer frame. A group with
  eligible frames also contributes when the window ends before three arrive.
  An all-rejected group contributes no vote. Sampled winners include
  `road_sharpness` in `samples.jsonl`.
- The camera keeps only the latest decoded image during inference. Frame counts
  cover observed candidates. Rejection counts cover quality failures, and
  inference counts cover successful calls on group winners.
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

## Diagnosing slope measurements

On the next start, `main.py` creates a local `imu_windows` table in
`observations.sqlite3`. Each row links to `observations.id` and stores the window's
individual accelerometer/gyro samples, device Roll/Pitch/Yaw, corrected and
smoothed tilt, sample timestamps, saved level references, sampling configuration,
and slope method (`fused_attitude_ema_mean_v1`) as JSON. The
observation and its inputs commit together. Windows with insufficient samples
can still have raw inputs even when their saved slope is NULL. Inputs unavailable
because the IMU failed have no raw row. This is a record of the samples returned
for each observation window, not a continuous archive of every BLE packet.

Raw inputs stay local; Supabase continues receiving observation summaries.
Old observations are preserved and have no retroactively reconstructed raw data.
Storage usage grows with collection time; there is no automatic raw-data deletion.

After collecting a short test route, compare recent windows with:

```bash
/home/ubuntu/test/.venv/bin/python -B inspect_imu_windows.py --limit 30
```

The report shows the saved slope, fused-angle replay, historical acceleration
mean and median estimates,
acceleration-magnitude standard deviation (m/s²), and angular-rate RMS (rad/s).
Candidate estimates use the recorded calibration, but do not apply the window
coverage checks; a candidate value does not make a NULL observation valid.
These diagnostics help investigate impacts and rotation; they do not prove road
grade or remove sustained acceleration. Use a rigid mount with sensor x forward,
and repeat `imu.py --zero` while stationary on level ground after moving the mount.

The fused-angle estimator runs continuously with a 0.25-second exponential
smoothing time constant and converts the two-second mean tilt to percent grade.
It never substitutes the old acceleration estimator when angles are unavailable.
Device fusion and smoothing do not guarantee correct road grade during motion.

For a standalone experiment, stop `main.py`, set the zero once on level ground,
then run the IMU monitor:

```bash
/home/ubuntu/test/.venv/bin/python -B imu.py --zero
/home/ubuntu/test/.venv/bin/python -B imu.py
```

Ctrl+C saves `imu_report_<UTC timestamp>.json` containing all accepted session
samples, calibration/settings, window results and summary statistics. Only the
standalone monitor buffers a full session and creates this report. `main.py`
continues saving SQLite/Supabase observations without extra report files.

## Validation

```bash
.venv/bin/python -m unittest test_camera_inference test_main_failures test_imu_slope test_imu_report test_imu_storage test_upload_supabase
# Load both bundled checkpoints and run a synthetic frame (requires PyTorch).
.venv/bin/python -m unittest test_surface_model
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
