#!/usr/bin/env python3
"""vsandbox client daemon — long-lived HTTP proxy with connection reuse.

Listens on a unix socket (or TCP port) and proxies RPC requests to the
remote vsandbox_service over a persistent HTTP/1.1 keep-alive connection.
This avoids the ~150ms overhead of establishing a new TCP connection through
the SSH tunnel for every tool call.

Usage:
    python3 vsandbox_client_daemon.py \
        --endpoint http://host.docker.internal:1804 \
        --socket /tmp/vsandbox_daemon.sock

    # Or with TCP:
    python3 vsandbox_client_daemon.py \
        --endpoint http://host.docker.internal:1804 \
        --port 18040

The daemon accepts POST requests on any path and forwards them to the
remote endpoint with the same path and body, using a persistent connection.
It writes a PID file next to the socket for lifecycle management.

Timing:
    When VSANDBOX_TIMING_CSV is set, each proxied RPC appends a row
    in the same format as vsandbox_exec_client.py.

Shutdown:
    Send SIGTERM or SIGINT, or DELETE /shutdown.
"""

import argparse
import csv
import http.client
import http.server
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from urllib.parse import urlparse


class _PersistentUpstream:
    """Thread-safe persistent HTTP connection to the upstream server."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None

    def _ensure_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=610)
        return self._conn

    def _reset_conn(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def request(
        self, method: str, path: str, body: bytes, headers: dict
    ) -> tuple[int, bytes, dict]:
        """Send request, return (status, body, headers). Auto-reconnects once on failure."""
        with self._lock:
            for attempt in range(2):
                try:
                    conn = self._ensure_conn()
                    conn.request(method, path, body=body, headers=headers)
                    resp = conn.getresponse()
                    resp_body = resp.read()
                    resp_headers = dict(resp.getheaders())
                    return resp.status, resp_body, resp_headers
                except (ConnectionError, OSError, http.client.HTTPException):
                    self._reset_conn()
                    if attempt == 1:
                        raise

    def close(self):
        with self._lock:
            self._reset_conn()


class _DaemonHandler(http.server.BaseHTTPRequestHandler):
    """Proxies requests to the upstream vsandbox_service."""

    # Set by the server before starting
    upstream: _PersistentUpstream = None  # type: ignore
    timing_csv: str = ""
    _shutdown_event: threading.Event = None  # type: ignore

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        t0 = time.time()
        try:
            status, resp_body, resp_headers = self.upstream.request(
                "POST",
                self.path,
                body,
                {"Content-Type": "application/json"},
            )
            delay = time.time() - t0
        except Exception as e:
            delay = time.time() - t0
            self._write_timing(self.path, delay, body)
            self.send_error(502, f"Upstream error: {e}")
            return

        self._write_timing(self.path, delay, body)

        self.send_response(status)
        self.send_header(
            "Content-Type", resp_headers.get("content-type", "application/json")
        )
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_DELETE(self):
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")
            if self._shutdown_event:
                self._shutdown_event.set()
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # suppress default stderr logging

    def _write_timing(self, path: str, delay: float, body: bytes):
        if not self.timing_csv:
            return
        # Extract sandbox_id from the request body
        sandbox_id = ""
        try:
            payload = json.loads(body)
            req = payload.get("req", payload)
            sandbox_id = req.get("sandbox_id", "")
        except Exception:
            pass
        service_name = path.lstrip("/")
        try:
            os.makedirs(os.path.dirname(self.timing_csv), exist_ok=True)
            with open(self.timing_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        sandbox_id,
                        "vsandbox_daemon",
                        service_name,
                        delay,
                        str(uuid.uuid4()),
                        "None",
                    ]
                )
        except OSError:
            pass


class _UnixHTTPServer(http.server.HTTPServer):
    """HTTPServer that listens on a unix domain socket."""

    address_family = socket.AF_UNIX

    def server_bind(self):
        # Remove stale socket file
        if os.path.exists(self.server_address):
            os.unlink(self.server_address)
        http.server.HTTPServer.server_bind(self)
        # Make socket world-accessible (docker containers may run as root)
        os.chmod(self.server_address, 0o777)


def main():
    parser = argparse.ArgumentParser(description="vsandbox client daemon")
    parser.add_argument(
        "--endpoint", required=True, help="Upstream vsandbox_service URL"
    )
    parser.add_argument("--socket", help="Unix socket path to listen on")
    parser.add_argument(
        "--port", type=int, help="TCP port to listen on (alternative to --socket)"
    )
    parser.add_argument("--pidfile", help="Write PID to this file")
    args = parser.parse_args()

    if not args.socket and not args.port:
        parser.error("Either --socket or --port is required")

    # Parse upstream endpoint
    parsed = urlparse(args.endpoint)
    upstream_host = parsed.hostname or "127.0.0.1"
    upstream_port = parsed.port or 80

    upstream = _PersistentUpstream(upstream_host, upstream_port)
    shutdown_event = threading.Event()

    # Configure handler class
    _DaemonHandler.upstream = upstream
    _DaemonHandler.timing_csv = os.environ.get("VSANDBOX_TIMING_CSV", "")
    _DaemonHandler._shutdown_event = shutdown_event

    # Create server
    if args.socket:
        server = _UnixHTTPServer(args.socket, _DaemonHandler)
        listen_addr = args.socket
    else:
        server = http.server.HTTPServer(("127.0.0.1", args.port), _DaemonHandler)
        listen_addr = f"127.0.0.1:{args.port}"

    # Write PID file
    pidfile = args.pidfile
    if not pidfile and args.socket:
        pidfile = args.socket + ".pid"
    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

    # Signal handling
    def _handle_signal(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(
        f"vsandbox daemon listening on {listen_addr}, upstream={args.endpoint}",
        flush=True,
    )

    # Serve in a thread so we can wait on shutdown_event
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    shutdown_event.wait()

    print("vsandbox daemon shutting down...", flush=True)
    server.shutdown()
    upstream.close()

    # Cleanup
    if args.socket and os.path.exists(args.socket):
        os.unlink(args.socket)
    if pidfile and os.path.exists(pidfile):
        os.unlink(pidfile)


if __name__ == "__main__":
    main()
