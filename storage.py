import json
import sqlite3

CONFIG = {
    "db_path": "observations.sqlite3",
}


def open_database():
    db_path = CONFIG["db_path"]
    connection = sqlite3.connect(db_path)
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
                slope_percent REAL,
                vibration_frequency_avg_hz REAL,
                objects_supported INTEGER,
                objects TEXT
            )
        """)
        columns = []
        for row in cursor.execute("PRAGMA table_info(observations)"):
            columns.append(row[1])
        for name in ("slope_percent", "vibration_frequency_avg_hz"):
            if name not in columns:
                cursor.execute(f"ALTER TABLE observations ADD COLUMN {name} REAL")
        connection.commit()
        cursor.close()
    except Exception:
        connection.close()
        raise
    return connection


def save_record(connection, record: dict) -> int:
    """Return the record ID after commit; roll back on failure."""

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
                slope_percent,
                vibration_frequency_avg_hz,
                objects_supported,
                objects
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["record_time"],
            record["latitude"],
            record["longitude"],
            record["surface_label"],
            record["surface_confidence"],
            record.get("slope_percent"),
            record.get("vibration_frequency_avg_hz"),
            record["objects_supported"],
            objects_json
        ))
        connection.commit()
        return cursor.lastrowid
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def close_database(connection) -> None:
    connection.close()
