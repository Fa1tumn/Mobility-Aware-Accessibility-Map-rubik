import argparse
import json
import os
from pathlib import Path
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

CONFIG = {
    "db_path": "observations.sqlite3",
    "table": "observations",
    "batch_size": 100,
    "timeout_s": 20,
}


def read_records(db_path):
    database_path = Path(db_path).expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM observations ORDER BY id").fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["objects_supported"] = bool(record["objects_supported"])
            if record["objects"] is not None:
                record["objects"] = json.loads(record["objects"])
            records.append(record)
        return records
    finally:
        connection.close()


def upload_records(records, supabase_url, api_key):
    parts = urlsplit(supabase_url.strip())
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("SUPABASE_URL must be the HTTPS project URL.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("SUPABASE_URL must contain only the project address or REST API base path.")
    if parts.path.rstrip("/") not in ("", "/rest/v1"):
        raise ValueError("Use the project URL or the REST API base URL ending in /rest/v1.")
    if not api_key:
        raise ValueError("Set SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY.")
    query = urlencode({"on_conflict": "id,record_time"})
    url = f"https://{parts.netloc}/rest/v1/{CONFIG['table']}?{query}"
    print(f"Upload endpoint: {url}")
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    uploaded = 0
    for start in range(0, len(records), CONFIG["batch_size"]):
        batch = records[start:start + CONFIG["batch_size"]]
        body = json.dumps(batch, allow_nan=False).encode("utf-8")
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=CONFIG["timeout_s"]) as response:
                response.read()
        except HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            message = message.replace(api_key, "[redacted]")[:500]
            raise RuntimeError(f"Supabase upload failed (HTTP {error.code}): {message}") from None
        except URLError as error:
            raise RuntimeError(f"Cannot reach Supabase: {error.reason}") from None
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(records)} records.")
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload SQLite observations to Supabase.")
    parser.add_argument("database", nargs="?", default=CONFIG["db_path"])
    args = parser.parse_args()
    try:
        supabase_url = os.environ.get("SUPABASE_URL", "https://vptvvtxqqocpaepziqmz.supabase.co/rest/v1/")
        api_key = os.environ.get("SUPABASE_SECRET_KEY", "")
        if not api_key:
            api_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not supabase_url or not api_key:
            raise ValueError("Set SUPABASE_URL and SUPABASE_SECRET_KEY before uploading.")
        records = read_records(args.database)
        if not records:
            print("No records to upload.")
            return
        uploaded = upload_records(records, supabase_url, api_key)
        print(f"Upload complete: {uploaded} records sent to {CONFIG['table']}.")
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as error:
        message = str(error)
        if api_key:
            message = message.replace(api_key, "[redacted]")
        print(f"Upload error: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
