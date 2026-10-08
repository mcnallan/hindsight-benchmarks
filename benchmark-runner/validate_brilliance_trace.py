#!/usr/bin/env python3
"""Gate Brilliance's full eval on actual Hindsight extraction wire responses."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    requests = {}
    responses = {}
    with args.trace.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["event"] == "request":
                body = record.get("body") or {}
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                if "facts" in schema.get("properties", {}):
                    requests[record["request_id"]] = body
            else:
                responses[record["request_id"]] = record
    errors = []
    facts = 0
    reasoning_tokens = 0
    max_completion_tokens = 0
    for request_id, request in requests.items():
        kwargs = request.get("chat_template_kwargs", {})
        if kwargs != {"reasoning_effort": "logic", "enable_thinking": True}:
            errors.append(f"{request_id}: wrong thinking controls")
        response = responses.get(request_id)
        if not response or response.get("status") != 200:
            errors.append(f"{request_id}: missing or unsuccessful response")
            continue
        body = response["body"]
        try:
            choice = body["choices"][0]
            if choice["finish_reason"] != "stop":
                errors.append(f"{request_id}: finish_reason={choice['finish_reason']}")
            data = json.loads(choice["message"]["content"])
            if not isinstance(data.get("facts"), list):
                errors.append(f"{request_id}: missing facts array")
            else:
                facts += len(data["facts"])
            usage = body.get("usage", {})
            reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            max_completion_tokens = max(max_completion_tokens, usage.get("completion_tokens", 0))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{request_id}: invalid facts response: {exc}")
    if not requests or not facts or not reasoning_tokens:
        errors.append("Expected strict facts requests, stored facts, and reported reasoning tokens")
    summary = {
        "strict_facts_requests": len(requests),
        "extracted_facts": facts,
        "reasoning_tokens": reasoning_tokens,
        "max_completion_tokens": max_completion_tokens,
        "errors": errors,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit("Brilliance extraction trace validation failed")


if __name__ == "__main__":
    main()
