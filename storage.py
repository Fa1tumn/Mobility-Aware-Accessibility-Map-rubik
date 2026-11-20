import json
import sqlite3

# 창 진단값. 로컬 SQLite에만 저장하고 Supabase에는 올리지 않는다(upload_supabase 참고).
# rejected_count/rejected_reasons: 품질 검사(과다 노출·노출 부족·단색)로 추론하지 않은 프레임 수와 이유.
WINDOW_METADATA_FIELDS = ("window_start_time", "window_end_time", "frame_count", "inference_count",
                          "rejected_count", "rejected_reasons")

CONFIG = {
    "db_path": "observations.sqlite3",
}


def open_database():
    db_path = CONFIG["db_path"]
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_time TEXT,
                latitude REAL,
                longitude REAL,
                surface_label TEXT,
                surface_confidence REAL,
                area_accept_ratio REAL,
                material_accept_ratio REAL,
                model_version TEXT,
                policy_version TEXT,
                slope_percent REAL,
                vibration_frequency_avg_hz REAL,
                objects_supported INTEGER,
                objects TEXT
            )
        """)
        columns = []
        for row in cursor.execute("PRAGMA table_info(observations)"):
            columns.append(row[1])
        for name in ("slope_percent", "vibration_frequency_avg_hz",
                     "area_accept_ratio", "material_accept_ratio"):
            if name not in columns:
                cursor.execute(f"ALTER TABLE observations ADD COLUMN {name} REAL")
        for name in ("model_version", "policy_version"):
            if name not in columns:
                cursor.execute(f"ALTER TABLE observations ADD COLUMN {name} TEXT")
        for name in WINDOW_METADATA_FIELDS:
            if name not in columns:
                kind = "INTEGER" if name.endswith("count") else "TEXT"
                cursor.execute(f"ALTER TABLE observations ADD COLUMN {name} {kind}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imu_windows (
                observation_id INTEGER PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
                payload TEXT NOT NULL
            )
        """)
        connection.commit()
        cursor.close()
    except Exception:
        connection.close()
        raise
    return connection


def save_record(connection, record: dict, imu_window=None) -> int:
    """Commit the observation and optional local IMU inputs together."""

    cursor = connection.cursor()
    try:
        objects_json = None
        if record["objects"] is not None:
            objects_json = json.dumps(record["objects"], ensure_ascii=False)
        cursor.execute("""
            INSERT INTO observations (
                record_time,
                latitude,
                longitude,
                surface_label,
                surface_confidence,
                area_accept_ratio,
                material_accept_ratio,
                model_version,
                policy_version,
                slope_percent,
                vibration_frequency_avg_hz,
                objects_supported,
                objects, """ + ", ".join(WINDOW_METADATA_FIELDS) + """
            ) VALUES (""" + ", ".join("?" * (13 + len(WINDOW_METADATA_FIELDS))) + """)
        """, (
            record["record_time"],
            record["latitude"],
            record["longitude"],
            record["surface_label"],
            record["surface_confidence"],
            record.get("area_accept_ratio"),
            record.get("material_accept_ratio"),
            record.get("model_version"),
            record.get("policy_version"),
            record.get("slope_percent"),
            record.get("vibration_frequency_avg_hz"),
            record["objects_supported"],
            objects_json,
            *(record.get(field) for field in WINDOW_METADATA_FIELDS),
        ))
        record_id = cursor.lastrowid
        if imu_window is not None:
            cursor.execute("INSERT INTO imu_windows (observation_id, payload) VALUES (?, ?)",
                           (record_id, json.dumps(imu_window, allow_nan=False, separators=(",", ":"))))
        connection.commit()
        return record_id
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def close_database(connection) -> None:
    connection.close()
