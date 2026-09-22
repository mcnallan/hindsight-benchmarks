#!/usr/bin/env python3
"""Transparent OpenAI HTTP proxy that records request/response NDJSON traces."""

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import requests


class CaptureProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: str
    output_path: Path
    timeout: float
    write_lock = threading.Lock()

    def _record(self, record: dict) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record, ensure_ascii=False)
        with self.write_lock:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _forward(self) -> None:
        request_id = str(uuid.uuid4())
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""
        parsed_body = None
        if body:
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = body.decode("utf-8", errors="replace")

        self._record(
            {
                "event": "request",
                "request_id": request_id,
                "method": self.command,
                "path": self.path,
                "body": parsed_body,
            }
        )

        target = self.upstream.rstrip("/") + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "authorization"}
        }
        started = time.monotonic()
        try:
            response = requests.request(
                self.command,
                target,
                headers=headers,
                data=body or None,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - started
            response_body = response.content
            try:
                parsed_response = response.json()
            except requests.JSONDecodeError:
                parsed_response = response_body.decode("utf-8", errors="replace")
            self._record(
                {
                    "event": "response",
                    "request_id": request_id,
                    "status": response.status_code,
                    "elapsed_seconds": elapsed,
                    "body": parsed_response,
                }
            )
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in {
                    "connection",
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._record(
                {
                    "event": "error",
                    "request_id": request_id,
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            payload = json.dumps(
                {"error": {"message": str(exc), "type": type(exc).__name__}}
            ).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    do_GET = _forward
    do_POST = _forward

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=610)
    args = parser.parse_args()

    parsed = urlsplit(args.upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit("--upstream must be an http(s) base URL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    CaptureProxy.upstream = args.upstream.rstrip("/")
    CaptureProxy.output_path = args.output
    CaptureProxy.timeout = args.timeout
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), CaptureProxy)
    print(f"Capturing {args.listen_host}:{args.listen_port} -> {CaptureProxy.upstream}")
    print(f"Trace: {CaptureProxy.output_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
