import json
import math
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

CONFIG = {
    "host": "0.0.0.0",       # Bind address; phones must send to the Pi's LAN IP.
    "port": 8000,
    "max_request_bytes": 4096,
}


def start_phone() -> dict:
    phone_state = {
        "latest_gps": None,
        "lock": threading.Lock(),
        "server": None,
        "thread": None,
    }

    class GPSHandler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def do_POST(self):
            if self.path != "/gps":
                self.send_error(404, "Use POST /gps")
                return

            try:
                size = int(self.headers.get("Content-Length"))
            except (TypeError, ValueError):
                self.send_error(400, "Invalid Content-Length")
                return
            if size <= 0:
                self.send_error(400, "Empty request")
                return
            if size > CONFIG["max_request_bytes"]:
                self.send_error(413, "Request too large")
                return

            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    self.send_error(400, "Incomplete request")
                    return
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                self.send_error(400, "Invalid JSON or read timeout")
                return

            if not update_gps(phone_state, payload):
                self.send_error(400, "Invalid GPS fields")
                return
            response = b'{"ok":true}'
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except OSError:
                return

        def log_message(self, format, *args):
            return

    server = HTTPServer((CONFIG["host"], CONFIG["port"]), GPSHandler)
    phone_state["server"] = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    phone_state["thread"] = thread
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    print(f"GPS receiver started on port {server.server_address[1]} (POST /gps).")
    return phone_state


def update_gps(phone_state: dict, payload: dict) -> bool:
    """Return False for invalid GPS payloads; keep the previous location."""
    # bool is a subclass of int, but is not valid GPS data.
    if not isinstance(payload, dict):
        return False
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    accuracy_m = payload.get("accuracy_m")
    for value in (latitude, longitude, accuracy_m):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            if not math.isfinite(value):
                return False
        except OverflowError:
            return False
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or accuracy_m < 0:
        return False

    # Pi receive time, not the phone sensor sampling time.
    gps = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "accuracy_m": float(accuracy_m),
        "gps_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gps_monotonic": time.monotonic(),
    }

    with phone_state["lock"]:
        phone_state["latest_gps"] = gps
    return True


def get_latest_gps(phone_state: dict):
    with phone_state["lock"]:
        gps = phone_state["latest_gps"]
        if gps is None:
            return None
        gps_snapshot = gps.copy()
    return gps_snapshot


def stop_phone(phone_state: dict) -> None:
    if phone_state is None:
        return
    server = phone_state["server"]
    thread = phone_state["thread"]
    if server is not None:
        try:
            if thread is not None and thread.is_alive():
                server.shutdown()
        finally:
            server.server_close()
    if thread is not None and thread.ident is not None:
        thread.join(timeout=3)
        if thread.is_alive():
            raise RuntimeError("GPS receiver thread did not stop.")


if __name__ == "__main__":
    phone_state = start_phone()
    try:
        while True:
            gps = get_latest_gps(phone_state)
            if gps is not None:
                print("Latest GPS:", gps)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping GPS receiver.")
    finally:
        stop_phone(phone_state)
