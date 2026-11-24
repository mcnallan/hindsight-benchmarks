#!/usr/bin/env python3
"""Report token usage from the retain capture without copying source or reply text."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_attempts(trace):
    attempts = {}
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # A live capture may still be writing its last record.
            record = json.loads(line)
            request_id = record["request_id"]
            body = record.get("body")
            body = body if isinstance(body, dict) else {}
            if record["event"] == "request" and record.get("path") == "/v1/chat/completions":
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                attempts[request_id] = {
                    "request_id": request_id,
                    "model": body.get("model"),
                    "reasoning_effort": body.get("reasoning_effort"),
                    "strict_facts": "facts" in schema.get("properties", {}),
                    "status": "pending",
                    "finish_reason": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                    "non_reasoning_completion_tokens": None,
                }
            elif request_id in attempts and record["event"] in ("response", "error"):
                attempt = attempts[request_id]
                usage = body.get("usage") or {}
                total = usage.get("completion_tokens")
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                final = None
                if isinstance(total, int) and isinstance(reasoning, int) and 0 <= reasoning <= total:
                    final = total - reasoning
                choices = body.get("choices") or [{}]
                attempt.update(
                    status=record.get("status", record.get("error_type", "error")),
                    finish_reason=choices[0].get("finish_reason"),
                    completion_tokens=total,
                    reasoning_tokens=reasoning,
                    non_reasoning_completion_tokens=final,
                )
    return list(attempts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    attempts = read_attempts(args.trace)
    if not attempts:
        print("No captured chat requests yet")
        return
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    completed = [
        row for row in attempts
        if row["strict_facts"] and row["status"] == 200 and row["finish_reason"] == "stop"
    ]
    measured = [row for row in completed if row["non_reasoning_completion_tokens"] is not None]
    print(json.dumps({
        "chat_attempts": len(attempts),
        "strict_facts_attempts": sum(row["strict_facts"] for row in attempts),
        "strict_facts_stopped": len(completed),
        "strict_facts_stopped_with_token_split": len(measured),
        "mean_reasoning_tokens": statistics.mean(row["reasoning_tokens"] for row in measured) if measured else None,
        "mean_non_reasoning_completion_tokens": statistics.mean(
            row["non_reasoning_completion_tokens"] for row in measured
        ) if measured else None,
        "max_completion_tokens": max((row["completion_tokens"] for row in measured), default=None),
    }, indent=2))


if __name__ == "__main__":
    main()
