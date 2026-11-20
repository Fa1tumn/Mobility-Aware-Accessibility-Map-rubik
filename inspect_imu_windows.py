"""Compare saved fused slope with historical acceleration estimators, read-only."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

import imu


def summarize_window(payload):
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported IMU window schema")
    samples = payload["samples"]
    stats = {"samples": len(samples), "mean_percent": None, "median_percent": None,
             "fused_percent": None,
             "accel_std_m_s2": None, "gyro_rms_rad_s": None}
    if not samples:
        return stats
    acceleration = np.array([[s[k] for k in ("ax", "ay", "az")] for s in samples])
    gyro = np.array([[s[k] for k in ("gx", "gy", "gz")] for s in samples])
    stats["accel_std_m_s2"] = float(np.std(np.linalg.norm(acceleration, axis=1)))
    stats["gyro_rms_rad_s"] = float(np.sqrt(np.mean(np.sum(gyro ** 2, axis=1))))
    if not payload["mounting_calibrated"]:
        return stats
    angles = [s.get("smoothed_slope_deg") for s in samples]
    if all(a is not None and np.isfinite(a) and abs(a) < 90 for a in angles):
        stats["fused_percent"] = float(100 * np.tan(np.radians(np.mean(angles))))
    if payload["level_gravity"] is not None:
        acceleration = acceleration @ imu.level_basis(payload["level_gravity"]).T
    mean = acceleration.mean(axis=0)
    denominator = np.hypot(mean[1], mean[2])
    if denominator > 0:
        stats["mean_percent"] = float(100 * mean[0] / denominator)
    denominator = np.hypot(acceleration[:, 1], acceleration[:, 2])
    valid = denominator > 0
    if np.count_nonzero(valid) > len(samples) / 2:
        stats["median_percent"] = float(np.median(100 * acceleration[valid, 0] / denominator[valid]))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="observations.sqlite3")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    connection = None
    try:
        connection = sqlite3.connect(Path(args.database).expanduser().resolve().as_uri() + "?mode=ro",
                                     uri=True)
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='imu_windows'").fetchone():
            print("No raw IMU windows yet. Restart main.py to record them; historical summaries cannot be replayed.")
            return 0
        rows = connection.execute("""
            SELECT o.id, o.record_time, o.slope_percent, w.payload
            FROM observations o JOIN imu_windows w ON w.observation_id = o.id
            ORDER BY o.id DESC LIMIT ?
        """, (args.limit,)).fetchall()
        print("Candidate values do not apply window timing/coverage checks; saved slope may be NULL. Fused values require new angle recordings.")
        print("id\ttime_utc\tsamples\tsaved_%\tfused_%\taccel_mean_%\taccel_median_%\taccel_std_m/s2\tgyro_rms_rad/s\tmethod")
        for identifier, timestamp, saved, payload in reversed(rows):
            payload = json.loads(payload)
            stats = summarize_window(payload)
            values = [saved, stats["fused_percent"], stats["mean_percent"], stats["median_percent"],
                      stats["accel_std_m_s2"], stats["gyro_rms_rad_s"]]
            numbers = ["NULL" if v is None else f"{v:.3f}" for v in values]
            print("\t".join([str(identifier), timestamp, str(stats["samples"]), *numbers, payload.get("slope_method", "unknown")]))
        if not rows:
            print("No raw IMU windows yet. Run main.py to record them.")
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Cannot inspect IMU windows: {error}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
