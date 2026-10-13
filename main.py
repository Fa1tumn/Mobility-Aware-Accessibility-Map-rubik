import time

import phone
import imu
import vision
import inference
import storage

CONFIG = {
    "interval_s": 1.0,            # Delay after each iteration, not a guaranteed sampling interval.
    "max_gps_time_diff_s": 1.0,
    "max_gps_accuracy_m": 10.0,
}


def combine_data(frame: dict, gps: dict, result: dict, imu_result=None):
    if frame is None or gps is None or result is None:
        return None

    time_diff_s = frame["frame_monotonic"] - gps["gps_monotonic"]
    if abs(time_diff_s) > CONFIG["max_gps_time_diff_s"]:
        return None
    if gps["accuracy_m"] > CONFIG["max_gps_accuracy_m"]:
        return None

    if imu_result is None:
        imu_result = {"slope_percent": None, "vibration_frequency_avg_hz": None}
    # Preserve the frame's read time as the record timestamp.
    record = {
        "record_time": frame["frame_time"],
        "latitude": gps["latitude"],
        "longitude": gps["longitude"],
        "surface_label": result["surface_label"],
        "surface_confidence": result["surface_confidence"],
        "slope_percent": imu_result["slope_percent"],
        "vibration_frequency_avg_hz": imu_result["vibration_frequency_avg_hz"],
        "objects_supported": result["objects_supported"],
        "objects": result["objects"],
    }
    return record


def main() -> None:
    phone_state = None
    imu_state = None
    camera = None
    connection = None
    try:
        models = inference.load_models()
        connection = storage.open_database()
        phone_state = phone.start_phone()
        imu_state = imu.start_imu()
        camera = vision.open_camera()
        print("Pipeline started. Press Ctrl+C to stop.")

        # Snapshot GPS before inference so new updates cannot change this record.
        while True:
            frame = vision.capture_frame(camera)
            gps = phone.get_latest_gps(phone_state)
            if frame is None:
                if camera["error"] is not None:
                    raise RuntimeError(camera["error"])
                print("Waiting for camera frames.")
                time.sleep(CONFIG["interval_s"])
                continue
            if gps is None:
                print("Waiting for phone GPS. Send data to the Pi's /gps endpoint.")
                time.sleep(CONFIG["interval_s"])
                continue

            samples = imu.get_window(imu_state, frame["frame_monotonic"])
            imu_result = imu.analyze_samples(samples)
            result = inference.analyze_image(frame["image"], models)
            record = combine_data(frame, gps, result, imu_result)
            if record is None:
                print("Skipped: GPS/frame time difference or reported accuracy exceeds the configured limit.")
            else:
                record_id = storage.save_record(connection, record)
                print(f"Saved record {record_id}: time={record['record_time']}, surface={record['surface_label']}")
            time.sleep(CONFIG["interval_s"])

    except KeyboardInterrupt:
        print("Ctrl+C received. Stopping.")
    except Exception as error:
        print(f"Pipeline error: {error}")
        raise
    finally:
        # Close each resource even if another cleanup fails.
        if camera is not None:
            try:
                vision.close_camera(camera)
            except Exception as error:
                print(f"Failed to close camera: {error}")
        if phone_state is not None:
            try:
                phone.stop_phone(phone_state)
            except Exception as error:
                print(f"Failed to stop GPS receiver: {error}")
        if imu_state is not None:
            try:
                imu.stop_imu(imu_state)
            except Exception as error:
                print(f"Failed to stop IMU receiver: {error}")
        if connection is not None:
            try:
                storage.close_database(connection)
            except Exception as error:
                print(f"Failed to close database: {error}")


if __name__ == "__main__":
    main()
