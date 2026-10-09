#!/usr/bin/env python3
"""Eight simultaneous strict retain calls with sampled server admission evidence."""

import argparse
import copy
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from validate_retain_trace import validate_trace


def select_payload(trace=None, fixture=None, request_line=None, request_id=None):
    if fixture:
        record = json.loads(fixture.read_text())
        body = record.get("body", record)
        coordinate = {"fixture": str(fixture)}
    else:
        body = None
        with trace.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                record = json.loads(line)
                if record.get("event") != "request":
                    continue
                candidate = record.get("body") or {}
                schema = candidate.get("response_format", {}).get("json_schema", {}).get("schema", {})
                if "facts" not in schema.get("properties", {}):
                    continue
                if request_line is not None and line_no != request_line:
                    continue
                if request_id is not None and record.get("request_id") != request_id:
                    continue
                body = candidate
                coordinate = {"trace": str(trace), "request_line": line_no, "request_id": record.get("request_id")}
                break
        if body is None:
            raise ValueError("No matching strict facts request found")
    wrapper = body.get("response_format", {}).get("json_schema", {})
    if wrapper.get("strict") is not True or "facts" not in wrapper.get("schema", {}).get("properties", {}):
        raise ValueError("Source must contain a strict facts json_schema request")
    if not body.get("messages"):
        raise ValueError("Source request must contain representative messages")
    return copy.deepcopy(body), coordinate


def scheduler_gauges(metrics_text, model):
    values = {"running": None, "waiting": None}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r'(\S+?)(?:\{(.*?)\})?\s+([-+\d.eE]+)(?:\s+\d+)?$', line)
        if not match:
            continue
        name, labels, value = match.groups()
        if labels:
            advertised = re.search(r'(?:^|,)\s*model_name="((?:\\.|[^"\\])*)"', labels)
            if advertised and json.loads('"' + advertised.group(1) + '"') != model:
                continue
        key = "running" if name.endswith("num_requests_running") else "waiting" if name.endswith("num_requests_waiting") else None
        if key:
            values[key] = (values[key] or 0) + float(value)
    return values


def run_probe(args, get=requests.get, post=requests.post):
    body, coordinate = select_payload(args.trace, args.fixture, args.request_line, args.request_id)
    thinking_mode = getattr(args, "thinking", "off")
    expected_thinking = None if thinking_mode == "native" else thinking_mode == "on"
    body.update(model=args.model, temperature=0.1, stream=False)
    if thinking_mode == "native":
        body.pop("chat_template_kwargs", None)
        body.pop("reasoning_effort", None)
    else:
        body["chat_template_kwargs"] = {"enable_thinking": expected_thinking}
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = 16384
    key = os.environ.get(args.api_key_env, "EMPTY")
    headers = {"Authorization": f"Bearer {key}"}
    base = args.base_url.rstrip("/")
    metrics_url = args.metrics_url or base.removesuffix("/v1") + "/metrics"
    for url in (base, metrics_url):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Use an HTTP(S) endpoint without embedded credentials")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty or new")
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o700)
    trace_path = args.output / "wire.ndjson"
    metrics_path = args.output / "metrics.ndjson"
    models = get(base + "/models", headers=headers, timeout=30)
    models.raise_for_status()
    if args.model not in {row["id"] for row in models.json().get("data", [])}:
        raise ValueError("Endpoint does not advertise the selected model")

    def snapshot():
        response = get(metrics_url, headers=headers, timeout=5)
        response.raise_for_status()
        return scheduler_gauges(response.text, args.model)

    baseline = snapshot()
    if baseline["running"] is None or baseline["waiting"] is None:
        raise ValueError("Metrics lack selected model running/waiting gauges")
    if baseline["running"] or baseline["waiting"]:
        raise ValueError("Endpoint is not idle; admission cannot be attributed to this probe")
    lock = threading.Lock()
    release = threading.Barrier(9, timeout=30)
    done = threading.Event()
    metrics_state = {"samples": 0, "errors": 0, "max_running": 0, "max_waiting": 0}
    started = time.monotonic()

    def write(handle, record):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with lock:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def monitor(handle):
        while not done.is_set():
            try:
                gauges = snapshot()
                metrics_state["samples"] += 1
                for metric in ("running", "waiting"):
                    if gauges[metric] is not None:
                        metrics_state["max_" + metric] = max(metrics_state["max_" + metric], gauges[metric])
                write(handle, {"elapsed_seconds": time.monotonic() - started, **gauges})
            except Exception as exc:
                metrics_state["errors"] += 1
                write(handle, {"elapsed_seconds": time.monotonic() - started, "error_type": type(exc).__name__})
            done.wait(args.poll_ms / 1000)

    def call(index, handle):
        request_id = f"capacity-{index}"
        write(handle, {"event": "request", "request_id": request_id, "body": body})
        release.wait()
        begin = time.monotonic()
        try:
            response = post(base + "/chat/completions", headers=headers, json=body, timeout=args.timeout)
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.text
            write(handle, {"event": "response", "request_id": request_id, "status": response.status_code,
                           "elapsed_seconds": time.monotonic() - begin, "body": response_body})
        except Exception as exc:
            # Exception text may include request details; retain only its type.
            write(handle, {"event": "error", "request_id": request_id, "error_type": type(exc).__name__,
                           "elapsed_seconds": time.monotonic() - begin})

    with trace_path.open("w") as wire, metrics_path.open("w") as metrics:
        trace_path.chmod(0o600)
        metrics_path.chmod(0o600)
        write(metrics, {"elapsed_seconds": 0, "baseline": True, **baseline})
        monitor_thread = threading.Thread(target=monitor, args=(metrics,), daemon=True)
        monitor_thread.start()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(call, i, wire) for i in range(8)]
                release.wait()
                for future in as_completed(futures):
                    future.result()
        finally:
            done.set()
            monitor_thread.join(timeout=6)
    validation = validate_trace(trace_path, args.model, expected_thinking, 16384,
                                getattr(args, "require_separated_reasoning", False))
    (args.output / "trace_summary.private.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.output / "trace_summary.private.json").chmod(0o600)
    require_nonempty = getattr(args, "require_nonempty", False)
    nonempty_passed = not require_nonempty or validation["empty_facts_responses"] == 0
    calls_valid = validation["error_count"] == 0 and validation["http_responses"] == 8 and nonempty_passed
    eight_observed = metrics_state["max_running"] == 8
    summary = {
        "model": args.model, "endpoint": base, "source": coordinate,
        "requested_concurrency": 8, "retain_max_tokens": 16384, "thinking_enabled": expected_thinking,
        "thinking_control": thinking_mode,
        "reasoning_responses": validation["reasoning_responses"],
        "reasoning_field_responses": validation["reasoning_field_responses"],
        "poll_interval_ms": args.poll_ms, "baseline": baseline, **metrics_state,
        "all_eight_calls_valid": calls_valid,
        "require_each_response_nonempty": require_nonempty,
        "nonempty_requirement_passed": nonempty_passed,
        "empty_facts_responses": validation["empty_facts_responses"],
        "eight_running_observed": eight_observed,
        "validation_error_count": validation["error_count"],
        "capacity_proven": calls_valid and eight_observed,
        "duration_seconds": round(time.monotonic() - started, 3),
        "caveat": "Running/waiting gauges are sampled process-wide. No cache reset is performed. "
                  "Fewer than eight observed running requests leaves eight-slot admission unproven; "
                  "pair this artifact with effective scheduler configuration and report the observed counts.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.json").chmod(0o600)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--fixture", type=Path, help="Explicit OpenAI request JSON, preserving representative prompts/schema")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--request-line", type=int)
    selection.add_argument("--request-id")
    parser.add_argument("--base-url", required=True, help="OpenAI base URL ending in /v1")
    parser.add_argument("--metrics-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="QUALITY_RETAIN_API_KEY")
    parser.add_argument("--poll-ms", type=int, choices=range(5, 21), default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--require-nonempty", action="store_true", help="Require facts from each response when the selected source is substantive")
    parser.add_argument("--thinking", choices=("on", "off", "native"), default="off")
    parser.add_argument("--require-separated-reasoning", action="store_true")
    args = parser.parse_args()
    if args.fixture and (args.request_line is not None or args.request_id is not None):
        parser.error("Request selection applies only to --trace")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    summary = run_probe(args)
    print(json.dumps(summary, indent=2), flush=True)
    # 2 distinguishes valid outputs with insufficient sampled admission evidence.
    raise SystemExit(0 if summary["capacity_proven"] else 2 if summary["all_eight_calls_valid"] else 1)


if __name__ == "__main__":
    main()
