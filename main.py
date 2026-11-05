import os
import time
from datetime import datetime, timezone

import phone
import imu
import vision
import inference
import storage
import upload_supabase

CONFIG = {
    "interval_s": 1.0,            # Delay after each iteration, not a guaranteed sampling interval.
    "max_gps_time_diff_s": 1.0,
    "max_gps_accuracy_m": 10.0,
}


def device_output(name, operation, *args):
    """Isolate device startup/read/inference errors from the rest of the pipeline."""
    try:
        return operation(*args)
    except Exception as error:
        print(f"{name} unavailable; its values will be NULL: {error}")
        return None


def combine_data(frame, gps, result, imu_result=None, reference_monotonic=None):
    if reference_monotonic is None:
        reference_monotonic = frame["frame_monotonic"] if frame is not None else time.monotonic()
    if gps is not None:
        time_diff_s = reference_monotonic - gps["gps_monotonic"]
        if abs(time_diff_s) > CONFIG["max_gps_time_diff_s"] or gps["accuracy_m"] > CONFIG["max_gps_accuracy_m"]:
            gps = None

    gps = gps or {}
    result = result or {}
    imu_result = imu_result or {}
    return {
        "record_time": frame["frame_time"] if frame is not None else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "latitude": gps.get("latitude"),
        "longitude": gps.get("longitude"),
        "surface_label": result.get("surface_label"),
        "surface_confidence": result.get("surface_confidence"),
        "slope_percent": imu_result.get("slope_percent"),
        "vibration_frequency_avg_hz": imu_result.get("vibration_frequency_avg_hz"),
        "objects_supported": result.get("objects_supported"),
        "objects": result.get("objects"),
    }


def main() -> None:
    phone_state = None
    imu_state = None
    camera = None
    connection = None
    try:
        supabase_url = os.environ.get("SUPABASE_URL", "https://vptvvtxqqocpaepziqmz.supabase.co/rest/v1/")
        api_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        upload_enabled = bool(supabase_url and api_key)
        pending_uploads = []
        if not upload_enabled:
            print("Supabase upload disabled: set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY). Records will be saved locally.")
        models = device_output("Inference models", inference.load_models)
        connection = storage.open_database()
        phone_state = device_output("GPS receiver", phone.start_phone)
        imu_state = device_output("IMU receiver", imu.start_imu)
        camera = device_output("Camera", vision.open_camera)
        print("Pipeline started. Press Ctrl+C to stop.")

        while True:
            frame = device_output("Camera", vision.capture_frame, camera) if camera is not None else None
            reference_monotonic = frame["frame_monotonic"] if frame is not None else time.monotonic()
            # Snapshot GPS before inference so new updates cannot change this record.
            gps = device_output("GPS", phone.get_latest_gps, phone_state) if phone_state is not None else None
            imu_result = None
            if imu_state is not None:
                samples = device_output("IMU", imu.get_window, imu_state, reference_monotonic)
                if samples is not None:
                    imu_result = device_output("IMU analysis", imu.analyze_samples, samples)
            result = None
            if frame is not None and models is not None:
                result = device_output("Image inference", inference.analyze_image, frame["image"], models)
            record = combine_data(frame, gps, result, imu_result, reference_monotonic)
            record_id = storage.save_record(connection, record)
            print(f"Saved record {record_id}: time={record['record_time']}, surface={record['surface_label']}")
            if upload_enabled:
                pending_uploads.append({"id": record_id, **record})
                try:
                    upload_supabase.upload_records(pending_uploads, supabase_url, api_key)
                except Exception as error:
                    message = str(error).replace(api_key, "[redacted]")
                    print(f"Supabase upload failed; records remain in SQLite and will be retried next iteration: {message}")
                else:
                    pending_uploads.clear()
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
