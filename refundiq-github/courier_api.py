"""Mock courier tracking API - a real HTTP service the agent calls over the network.
GET /track/<tracking_id>  ->  JSON shipment status
Run standalone: python courier_api.py   (tools.py also auto-starts it in the background)"""
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import COURIER_HOST, COURIER_PORT, DB_PATH


class CourierHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/track/"):
            return self._send(404, {"error": "unknown endpoint"})
        tracking_id = self.path.split("/track/", 1)[1].strip()
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT tracking_id, status, expected_date, delivered_date, simulate "
            "FROM shipments WHERE tracking_id = ?", (tracking_id,)).fetchone()
        con.close()
        if not row:
            return self._send(404, {"error": f"tracking id {tracking_id} not found"})
        if row[4] == "timeout":
            time.sleep(5)  # simulate a courier outage: client will time out
        self._send(200, {"tracking_id": row[0], "status": row[1],
                         "expected_delivery": row[2], "delivered_on": row[3],
                         "courier": "SwiftShip Logistics"})

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client already gave up (timeout case)

    def log_message(self, *args):
        pass  # keep the console quiet


def start_in_background():
    """Start the server on a daemon thread if nothing is listening yet."""
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex((COURIER_HOST, COURIER_PORT)) == 0:
            return  # already running
    server = ThreadingHTTPServer((COURIER_HOST, COURIER_PORT), CourierHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


if __name__ == "__main__":
    print(f"Courier API on http://{COURIER_HOST}:{COURIER_PORT}/track/<id>")
    ThreadingHTTPServer((COURIER_HOST, COURIER_PORT), CourierHandler).serve_forever()
