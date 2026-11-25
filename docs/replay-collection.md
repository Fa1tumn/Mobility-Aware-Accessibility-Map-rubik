# Rubik Pi campus recording guide

Record one campus route for all four model and frame-selection conditions at 2-, 4-, and 8-second intervals. The Pi automatically saves timestamped JPEG frames, continuous IMU samples, and phone GPS fixes. Keep the camera mount and exposure settings fixed. Retain blurry and underexposed frames for replay.

## 现在开始采集

打开 IMU，在同一网络上的手机启动 GPS 发送端，向 `http://<Pi 的局域网 IP>:8000/gps` 发送定位。在当前 Pi 的项目目录运行：

```bash
/home/ubuntu/test/.venv/bin/python -u record_replay.py record
```

终端会显示本次的新目录 `recordings/campus_<UTC 时间>`。看到 `RECORDING` 再出发。默认每秒保存 5 张候选图像，连续保存 IMU 和手机发来的 GPS，每 5 秒显示三路计数和距最近一条数据的时间。结束时在该终端按 Ctrl+C，然后运行 `python3 record_replay.py check <本次目录>` 检查数据。采集可以全程离线保存，无需模型或云服务。

## Run a 30-second field check

1. Mount the camera so the lower center of the frame shows the ground just ahead of the wheels. Secure the IMU and note its orientation. Close any previous process using the camera, BLE IMU, or port 8000. Disconnect the phone's Bluetooth client from the IMU so the Pi can connect.
2. On the Pi, open the project directory containing `record_replay.py`. Use the Python environment that ran the camera previously. Replace the first line below if that environment lives elsewhere.

   ```bash
   PY=.venv/bin/python
   "$PY" -c 'import cv2, numpy, bleak'
   gst-inspect-1.0 qtiqmmfsrc >/dev/null
   hostname -I
   df -h .
   ```

3. Connect the phone and Pi to the same network. Enable phone location and point the existing GPS sender at `http://<pi-lan-ip>:8000/gps`. Send JSON via HTTP POST with numeric `latitude`, `longitude`, and `accuracy_m` values from each fix. Aim for one fix per second. The recorder starts the receiver.
4. Run the check outdoors. The script waits up to 60 seconds for fresh inputs. Start moving when the terminal prints `RECORDING`. Use a new directory name for each recording.

   ```bash
   "$PY" record_replay.py record recordings/smoke --seconds 30
   "$PY" record_replay.py check recordings/smoke
   du -sh recordings/smoke
   ```

5. Confirm that all three counts increase. At the default targets of 5 frames/s and 100 IMU samples/s, a 30-second recording should yield roughly 150 images and 3,000 IMU samples. A 1 Hz phone sender should produce about 30 fixes. Open several JPEGs to check framing and exposure. Check that coordinates in `gps.csv` fall on campus and review `accuracy_m`. Estimate the storage needed by scaling the trial size to the planned route duration, and allow some headroom.

## Record the campus route

```bash
"$PY" record_replay.py record recordings/campus_eval
```

Plan a 10–15 minute route across several available materials. Include smooth travel, slow rolling, turns, and naturally bumpy sections. Where possible, spend more than eight seconds on each material so the longest interval can sample it. Press Ctrl+C at the endpoint to finish. The session can run from start to finish without anyone entering labels. If a teammate wants to flag a hard-to-identify surface, they can type a short field note in the recording terminal and press Enter. The recorder timestamps it in `marks.csv`.

## Check the route and copy it to the Mac

```bash
"$PY" record_replay.py check recordings/campus_eval
du -sh recordings/campus_eval
```

Review each stream's count, measured rate, and longest gap, including gaps at the beginning and end. At 5 frames/s, a two-second window should usually have about ten candidate frames. If camera gaps repeatedly exceed two seconds or IMU or GPS updates stall for long stretches, check the connections on site and repeat the affected segment if needed. Open several JPEGs to check that the road surface is visible and confirm that the GPS coordinates follow the route. A recording with zero operator marks is valid.

From the Mac project directory, copy the full route folder. Replace the example account, host, and path with the Pi's actual values. Run `pwd` in the Pi project directory to get the absolute path.

```bash
mkdir -p recordings
scp -r pi-user@pi-host:/absolute/project/path/recordings/campus_eval recordings/
python3 record_replay.py check recordings/campus_eval
```

## Label the reference segments on the Mac

Before inspecting model predictions, have one reviewer open the original JPEGs in the order listed in `frames.csv`. When the material under the lower center of the image changes, start a new segment. Record one row per segment with its ID, first and last frame times (`t_s`), and observed material. Use the GPS track and nearby frames to locate a boundary. All four conditions share this reference.

Use the material names `blocks`, `cement`, `asphalt`, `urethane`, `soil_stone`, and `other`. If the material is ambiguous across nearby frames, label the segment `uncertain` and report how many segments could not be verified. If the team also needs fine-tuning data, collect different segments in a separate `campus_train` session and reserve the entire `campus_eval` route for evaluation. A training pipeline that requires pixel masks needs separate pixel-level annotations.

## Align the streams for replay

- `session.json` records the UTC start time, Pi monotonic clock origin, mode, target rates, duration, and exit status. In every CSV, `t_s` means seconds since that session's origin. The Pi timestamps frames after JPEG decoding and GPS fixes when the phone's POST arrives. IMU sample times are estimated from BLE batch arrivals and the configured sample rate. Transmission and decoding delays limit alignment precision.
- `frames.csv` maps frame times to files in `frames/`. Compute image sharpness in the lower-center road region and estimate shake from nearby samples in `imu.csv`. To simulate live selection, use only information available at decision time. Acceleration includes gravity and is measured in m/s², angular velocity in rad/s, and orientation in degrees. Align streams using `t_s`. The IMU's `timestamp_s` is a sample clock generated by the receiver.
- `gps.csv` retains coordinates and reported accuracy. To follow the project's existing matching policy, take the nearest fix within one second of the frame and require a reported accuracy of 10 m or better. Mark missing matches as uncovered.

## Compare the four conditions

Run the offline comparison from the project root. The script compares all four methods over complete 2-second windows in a finished route recording.

```bash
/home/ubuntu/test/.venv/bin/python -B compare_replay.py recordings/campus_20261125T075508Z
```

The default output is `four_method_replay/` inside the recording. To repeat a run in a new location, pass `--output DIR`.
Open `index.html` for the English prediction timeline, GPS maps, and a chronological photo gallery for each method and interval.
`results.json` records every complete window, source frame, quality metrics, matched GPS fix, model prediction, and label transition.
The `changes/` directory contains the original JPEG selected when the label first switches, including transitions to and from `unknown`.
The first prediction establishes the starting label. A skipped window has no prediction and the next detection is compared with the last observed label.

For each interval, use the same complete time windows from the evaluation route and one inference per window for all four conditions. A and B use the same frame nearest the scheduled time. C and D use the same frame selected by IMU and image quality. Apply the same rule to windows with insufficient data and report how many were affected. Report the actual inference count for each interval.

For map accuracy, divide the number of reference segments assigned the correct material by the total number of manually verified reference segments. Fix the segment boundaries, the GPS assignment rule, and the rule for combining multiple predictions before comparing conditions. For example, use a majority vote and mark ties as unresolved. Keep segments with missing fixes or predictions in the denominator. Report the proportion of segments that received any mapped prediction alongside accuracy, so missed short segments remain visible.

Compare A with B and C with D for the fine-tuning effect. Compare A with C and B with D for the frame-selection effect. Compare A with D for the combined change.
