#!/usr/bin/env python3
"""One-shot smoke-to-full handoff requiring a recorded source review."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


RUNNER = Path(__file__).resolve().parent
PROFILES = ("qwen35-9b", "mellum21-12b", "mellum2-instruct")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def trace_path(run):
    return Path((run / "trace-directory.txt").read_text().strip()).resolve() / "smoke.ndjson"


def write_review(run, report, request_ids):
    """Bind the evaluator's manual report to actual completed source pairs."""
    selected = set(request_ids)
    if not selected or not report.is_file() or not report.read_text().strip():
        raise ValueError("A nonempty manual report and reviewed request IDs are required")
    pending, pairs = {}, {}
    prefix = hashlib.sha256()
    prefix_bytes = 0
    trace = trace_path(run)
    with trace.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                break  # A capture writer may still be appending its final record.
            prefix.update(raw)
            prefix_bytes += len(raw)
            record = json.loads(raw)
            request_id = record["request_id"]
            if request_id not in selected:
                continue
            if record["event"] == "request":
                body = record["body"]
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                if "facts" not in schema.get("properties", {}):
                    raise ValueError("Reviewed ID must identify a strict facts request")
                pending[request_id] = (number, digest_messages(body["messages"]))
            elif record["event"] == "response" and request_id in pending:
                first_line, source_hash = pending.pop(request_id)
                if record.get("status") != 200:
                    raise ValueError("Reviewed pair has an unsuccessful response")
                facts = json.loads(record["body"]["choices"][0]["message"]["content"])["facts"]
                pairs[request_id] = {
                    "request_line": first_line, "response_line": number,
                    "source_messages_sha256": source_hash, "facts": len(facts),
                }
    if set(pairs) != selected:
        raise ValueError("Every reviewed ID must have a captured source and completed facts response")
    checkpoint = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "hindsight_eval_setup",
        "bounded_source_review_recorded": True,
        "semantic_accuracy_pass_claimed": False,
        "deployment_sha256": digest(run / "deployment.json"),
        "report_path": str(report.resolve()), "report_sha256": digest(report),
        "trace_path": str(trace), "trace_prefix_bytes": prefix_bytes,
        "trace_prefix_sha256": prefix.hexdigest(), "reviewed_pairs": pairs,
    }
    destination = run / "source-review-checkpoint.json"
    destination.write_text(json.dumps(checkpoint, indent=2) + "\n")
    destination.chmod(0o600)
    return checkpoint


def digest_messages(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def check_review(run):
    value = json.loads((run / "source-review-checkpoint.json").read_text())
    if not value.get("bounded_source_review_recorded") or not value.get("reviewed_pairs"):
        raise ValueError("Source review checkpoint is missing reviewed pairs")
    if value["deployment_sha256"] != digest(run / "deployment.json"):
        raise ValueError("Source review belongs to another deployment")
    if value["report_sha256"] != digest(Path(value["report_path"])):
        raise ValueError("Manual source report changed after its checkpoint")
    trace = trace_path(run)
    if value["trace_path"] != str(trace):
        raise ValueError("Source review belongs to another trace")
    remaining = value["trace_prefix_bytes"]
    prefix = hashlib.sha256()
    with trace.open("rb") as handle:
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Reviewed trace prefix was truncated")
            prefix.update(block)
            remaining -= len(block)
    if prefix.hexdigest() != value["trace_prefix_sha256"]:
        raise ValueError("Reviewed source/response capture changed")
    return value


def launcher_hashes():
    return {name: digest(RUNNER / name) for name in
            ("run_gpu1_quality.sh", "run_local_quality.py", "validate_retain_trace.py",
             "src/hindsight_benchmark/quality.py")}


def status_rows(run, profile):
    rows = [line.split("\t") for line in (run / "status.tsv").read_text().splitlines()]
    return [row for row in rows if len(row) >= 3 and row[0] == profile]


def handoff(args):
    run = args.run_dir.resolve()
    state_file = run / "handoff-state.json"
    if state_file.exists():
        raise ValueError("Handoff already exists; automatic restart is forbidden")
    if not os.environ.get("CODEX_LB_TOKEN"):
        raise ValueError("CODEX_LB_TOKEN must be explicitly supplied in the handoff environment")
    pins = launcher_hashes()
    deployment_hash = digest(run / "deployment.json")
    state = {"profile": args.profile, "smoke_session": args.smoke_session,
             "full_session": args.full_session, "deployment_sha256": deployment_hash,
             "launcher_sha256": pins, "state": "waiting_for_smoke_and_source_review"}

    def save(label, **extra):
        state.update(state=label, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        state_file.write_text(json.dumps(state, indent=2) + "\n")
        state_file.chmod(0o600)
        print(label, flush=True)

    save(state["state"])
    deadline = time.monotonic() + args.max_wait_seconds
    try:
        while time.monotonic() < deadline:
            rows = status_rows(run, args.profile)
            if any(row[1] == "full" for row in rows):
                raise ValueError("Full phase already exists; refusing a duplicate launch")
            terminal = [row[2] for row in rows if row[1] == "smoke" and row[2] != "running"]
            if terminal and terminal[-1] != "passed":
                raise ValueError("Smoke failed or was interrupted; full remains blocked")
            if terminal and (run / "source-review-checkpoint.json").exists():
                summary = json.loads((run / "smoke_trace_summary.json").read_text())
                if summary.get("error_count", len(summary.get("errors", []))) or not summary.get("extracted_facts"):
                    raise ValueError("Strict smoke trace gate failed")
                review = check_review(run)
                if digest(run / "deployment.json") != deployment_hash or launcher_hashes() != pins:
                    raise ValueError("Frozen deployment or launcher changed while waiting")
                command = shlex.join(["bash", str(RUNNER / "run_gpu1_quality.sh"),
                                      args.profile, str(run), "full"])
                # Credentials are structured process arguments, never rendered in logs/errors.
                argv = ["tmux", "new-session", "-d", "-s", args.full_session]
                env = dict(os.environ, QUALITY_MODEL_PROVENANCE_FILE=str(run / "deployment.json"))
                for name in ("CODEX_LB_TOKEN", "QUALITY_MODEL_PROVENANCE_FILE", "QUALITY_RETAIN_API_KEY",
                             "QUALITY_RETRIEVAL_API_KEY", "QUALITY_JUDGE_MODEL"):
                    if name in env:
                        argv.extend(["-e", name + "=" + env[name]])
                argv.append(command)
                launched = subprocess.run(argv, env=env, capture_output=True)
                if launched.returncode:
                    raise ValueError("Detached full session launch failed; no automatic retry")
                save("full_session_launched", source_review_checkpoint_sha256=digest(run / "source-review-checkpoint.json"),
                     reviewed_pair_count=len(review["reviewed_pairs"]))
                startup_deadline = time.monotonic() + 300
                full_trace = trace_path(run).with_name("full.ndjson")
                while time.monotonic() < startup_deadline:
                    full_rows = [row for row in status_rows(run, args.profile) if row[1] == "full"]
                    if any(row[2] != "running" and row[2] != "passed" for row in full_rows):
                        raise ValueError("Full launched but reported failure; preserve without restarting")
                    if full_trace.exists():
                        with full_trace.open() as handle:
                            for line in handle:
                                if not line.endswith("\n"):
                                    break
                                event = json.loads(line)
                                schema = event.get("body", {}).get("response_format", {}).get("json_schema", {}).get("schema", {})
                                if event.get("event") == "request" and "facts" in schema.get("properties", {}):
                                    save("full_running", actual_backend_strict_retain_started=True)
                                    return
                    time.sleep(args.poll_seconds)
                save("full_session_launched_startup_unverified", actual_backend_strict_retain_started=False)
                return
            exists = subprocess.run(["tmux", "has-session", "-t", args.smoke_session], capture_output=True)
            if not terminal and exists.returncode:
                raise ValueError("Smoke session ended without a passing terminal state")
            time.sleep(args.poll_seconds)
        raise ValueError("Handoff deadline reached; full remains blocked")
    except Exception as exc:
        # Never stringify a subprocess exception that might contain credential arguments.
        safe_message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        save("blocked", reason=safe_message)
        raise SystemExit(safe_message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("record-review")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--report", type=Path, required=True)
    review.add_argument("--request-id", action="append", required=True)
    launch = sub.add_parser("wait")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--profile", choices=PROFILES, required=True)
    launch.add_argument("--smoke-session", required=True)
    launch.add_argument("--full-session", required=True)
    launch.add_argument("--max-wait-seconds", type=float, default=14400)
    launch.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.command == "record-review":
        checkpoint = write_review(args.run_dir.resolve(), args.report.resolve(), args.request_id)
        print(json.dumps({"reviewed_pair_count": len(checkpoint["reviewed_pairs"]),
                          "deployment_sha256": checkpoint["deployment_sha256"]}), flush=True)
    else:
        if args.poll_seconds <= 0 or args.max_wait_seconds <= 0:
            parser.error("Wait durations must be positive")
        handoff(args)


if __name__ == "__main__":
    main()
