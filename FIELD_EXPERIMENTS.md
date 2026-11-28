# Rubik Pi field experiments

Run one recorder at a time from the project root using the Pi's collection environment. Wait for `RECORDING` before moving. Press Ctrl+C at the end and wait for the recorder to print the saved counts.

Each run creates `recordings/<experiment>/<name>/<UTC timestamp>/`. Reuse a name for repeated runs or pass `--output /path/to/disk` to change the base directory. All three recorders save locally. Only the campus photo experiment requires phone GPS.

## Campus photos, IMU and GPS

On the same network as the Pi, start the phone GPS sender and keep it sending fixes to `http://<Pi LAN IP>:8000/gps`. The recorder prints `RECORDING` once the IMU is ready, an image has arrived, and a GPS fix has been received within the last three seconds.

```bash
python frames_test.py --name block
```

The recorder saves every complete JPEG received from the camera pipeline, including blurry and dark frames. Photos go in `recordings/frames/block/<UTC timestamp>/frames/`. The same session contains `frames.csv` with image paths and timestamps, and `imu.csv` with all received IMU samples.

`gps.csv` saves each phone fix with its Pi receive time (`gps_time`, UTC), latitude, longitude, and reported accuracy in metres. Use `t_s` to align these fixes with photos and IMU samples. Watch the `gps` count during collection. After stopping, run `python record_replay.py check <session-directory>` to inspect sample counts and the longest gap in each stream.

To combine the campus survey and the surface test, use `--name campus` and follow the marking sequence below. The name applies to the whole run. Marks identify sections within it.

## IMU surface test

```bash
python imu_test.py --name red_block
```

1. Hold the wheelchair still on level ground for 30 seconds. Type `baseline_start` and `baseline_end` at the start and end. If the protocol includes three firm pushes, do them after the baseline and before the steady-speed passes.
2. Press Enter as the front wheels enter each section: red path, block path, red path, block path. The recorder writes `red_1`, `block_1`, `red_2`, and `block_2` to `marks.csv`.
3. If you continue recording, press Enter when leaving the last block section to mark `route_end`. To add a note at any point, type it and press Enter. Notes leave the next automatic label available.

Skip the marks if you only need the IMU data.

For repeated surface runs, keep the mount, rider and load fixed. Record the pass speed or the time over a measured distance. Add notes for turns and stops so those periods can be identified later.

## Wheelchair coast test

```bash
python coast_test.py --name red
```

Mount the IMU with its positive X axis pointing forward. Choose a straight, level path with about 10 m of clear space beyond the release line. Use the same rider, direction, and a similar release speed on both surfaces.

1. Start with a 30-second stationary baseline. For each trial, pause for 3 seconds about 4 m behind the line, push up to walking speed, and hold that speed over the last 1 to 2 m.
2. Press Enter when the pusher lets go. Let the wheelchair roll freely with the rider sitting still. Press Enter again at a complete stop, then remain still for 3 seconds.
3. Repeat for each trial on this surface. End the run with Ctrl+C before starting the block path run with `python coast_test.py --name block`.

`marks.csv` records each release and stop. `trials.csv` lists the paired events and coast duration. At the end of the run, `coast_summary.csv` adds the estimated mean deceleration in m/s^2. The calculation subtracts the time-weighted mean X acceleration during coasting from its stationary mean 0.5 to 2.5 seconds after the stop. It assumes the chair's pitch changes little. Incomplete IMU coverage or a sample gap over 0.1 seconds leaves the estimate blank.

## Timestamps and units

All three experiments save the files below. The IMU is ready as soon as the first motion sample arrives. The configured IMU output is 100 Hz with 42 Hz bandwidth.

| File | Measurements |
| --- | --- |
| `imu.csv` | Each motion sample has XYZ acceleration including gravity (m/s²), XYZ angular velocity (rad/s), roll, pitch and yaw (degrees), and the original 20-byte packet in `raw_hex`. `t_s` estimates the sample time and `timing_epoch` increments on each clock reset. |
| `imu_ble.csv` | Each notification has a Pi receive time in `t_s`, byte count, counts of completed motion and register packets, discarded bytes, and bytes buffered for the next notification. Join by `ble_notification_id` to find which notification completed each motion packet. |
| `imu_registers.csv` | Device quaternion components `q0` through `q3` in rows for register `0x51`, with the Pi receive time in `t_s` and the full reply in `raw_hex`. The quaternion is polled separately from the 100 Hz motion stream. |

The default motion packets have no device timestamp or sample counter. Use the notification log to inspect arrival gaps, batching and framing. Identical consecutive packets can also result from sensor quantization.

The recorder waits 100 ms after a successful quaternion request. The observed rate also depends on BLE latency. The terminal shows how many quaternion replies arrived. If a register write fails, polling stops and the error is recorded in `session.json`. Motion capture continues while the BLE connection is available.

### Optional register diagnostics

Pass `--imu-diagnostics` to any of the three commands to add the measurements below to `imu_registers.csv`. The recorder first queries the magnetometer chip ID, then polls the quaternion and four other blocks in rotation. `session.json` saves the count for each block under `imu_register_replies`. A count of zero means that no reply for that address arrived.

| Starting register | Columns in `imu_registers.csv` |
| --- | --- |
| `0x72` | `mag_chip_id` identifies the magnetometer. |
| `0x3A` | `mag_x_raw`, `mag_y_raw`, `mag_z_raw` are signed magnetometer counts. When the device reports a recognized chip ID, `mag_x_ut`, `mag_y_ut`, `mag_z_ut` give the field in µT. |
| `0x51` | `q0` through `q3` are the four dimensionless quaternion components in device order. |
| `0x41` | `d0_status_raw` through `d3_status_raw` are the returned port words. Their meaning depends on port mode and firmware. The same reply covers pressure and height on devices with a barometer. |
| `0x40` | `temperature_c` is module temperature in °C. |
| `0x64` | `battery_v` is the reported battery voltage in V. |

The datasheet gives a barometer range of 300–1100 hPa. The model comparison omits pressure from this model's outputs. When the returned pressure falls within that range the recorder fills `pressure_pa` and `barometric_height_m` using the standard register layout. Otherwise both cells stay empty and the raw reply is preserved. A zero-filled port block can also come from firmware with unimplemented registers. [Model datasheet](https://wit-motion.yuque.com/wumwnr/docs/fudgtt) · [BLE protocol](https://wit-motion.yuque.com/wumwnr/docs/gpare3) · [Standard register definitions](https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/wit-standard-communication-protocol) · [Magnetometer conversion](https://wit-motion.yuque.com/wumwnr/docs/pemnbmhv5olkvuuq)

Frame, GPS, IMU, and mark times in `t_s` use seconds from the same Pi monotonic clock origin. `session.json` records the UTC start, readiness time, and final status. The camera timestamps each complete JPEG on receipt. IMU sample times use the existing BLE timing estimate. Acceleration is in m/s^2 and includes gravity. Angular velocity is in rad/s and orientation angles are in degrees.
