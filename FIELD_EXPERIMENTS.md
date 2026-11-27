# Rubik Pi field experiments

Run one recorder at a time from the project root using the Pi's collection environment. Wait for `RECORDING` before moving. Press Ctrl+C at the end and wait for the recorder to print the saved counts.

Each run creates `recordings/<experiment>/<name>/<UTC timestamp>/`. Reuse a name for repeated runs or pass `--output /path/to/disk` to change the base directory. All three recorders save to local storage and can run without phone GPS or inference models.

## Campus photos and IMU

```bash
python frames_test.py --name block
```

The recorder saves every complete JPEG received from the camera pipeline, including blurry and dark frames. Photos go in `recordings/frames/block/<UTC timestamp>/frames/`. The same session contains `frames.csv` with image paths and timestamps, and `imu.csv` with all received IMU samples.

To combine the campus survey and the surface test, use `--name campus` and follow the marking sequence below. The name applies to the whole run. Marks identify sections within it.

## IMU surface test

```bash
python imu_test.py --name red_block
```

1. Hold the wheelchair still on level ground for 30 seconds, then give it about three firm forward pushes. The recorder saves samples throughout.
2. Press Enter as the front wheels enter each section: red path, block path, red path, block path. The recorder writes `red_1`, `block_1`, `red_2`, and `block_2` to `marks.csv`.
3. If you continue recording, press Enter when leaving the last block section to mark `route_end`. To add a note at any point, type it and press Enter. Notes leave the next automatic label available.

Skip the marks if you only need the IMU data.

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

Frame, IMU, and mark times in `t_s` use seconds from the same Pi monotonic clock origin. `session.json` records the UTC start, readiness time, and final status. The camera timestamps each complete JPEG on receipt. IMU sample times use the existing BLE timing estimate. Acceleration is in m/s^2 and includes gravity. Angular velocity is in rad/s and orientation angles are in degrees.
