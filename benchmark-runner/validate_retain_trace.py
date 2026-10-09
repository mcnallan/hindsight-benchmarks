#!/usr/bin/env python3
"""Validate captured strict facts against their actual schema and thinking mode."""

import argparse
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_trace(trace, model, thinking, completion_cap, require_separated_reasoning=False):
    pending, completed, ignored = {}, set(), set()
    errors, samples = [], []
    facts = reasoning_tokens = max_completion = error_count = request_count = 0
    response_count = transport_errors = unmatched_events = 0
    empty_facts_count = 0
    empty_facts_request_ids = []
    reasoning_responses = reasoning_characters = max_reasoning_characters = 0
    reasoning_fields = {"reasoning": 0, "reasoning_content": 0}

    def error(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 20:
            errors.append(message)

    with trace.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                request_id = record["request_id"]
                event = record["event"]
            except (json.JSONDecodeError, KeyError, TypeError):
                error(f"line {line_number}: invalid trace record")
                continue
            if event == "request":
                body = record.get("body") or {}
                response_format = body.get("response_format", {})
                wrapper = response_format.get("json_schema", {})
                schema = wrapper.get("schema", {})
                if "facts" not in schema.get("properties", {}):
                    ignored.add(request_id)
                    continue
                if request_id in pending or request_id in completed or request_id in ignored:
                    error(f"{request_id}: duplicate request ID")
                    continue
                request_count += 1
                # Keep validation metadata only, never messages/source chunks.
                pending[request_id] = schema
                if body.get("model") != model:
                    error(f"{request_id}: unexpected model")
                kwargs = body.get("chat_template_kwargs") or {}
                if ((thinking is None and "enable_thinking" in kwargs)
                        or kwargs.get("enable_thinking") is not thinking):
                    error(f"{request_id}: wrong thinking control")
                if thinking is None and body.get("reasoning_effort"):
                    error(f"{request_id}: reasoning effort on native nonreasoning request")
                if response_format.get("type") != "json_schema" or wrapper.get("strict") is not True:
                    error(f"{request_id}: missing strict json_schema request")
                if body.get("max_completion_tokens", body.get("max_tokens")) != completion_cap:
                    error(f"{request_id}: wrong retain completion cap")
                continue
            if request_id in ignored:
                continue
            if event not in ("response", "error"):
                error(f"{request_id}: unexpected trace event")
                continue
            schema = pending.pop(request_id, None)
            if schema is None:
                unmatched_events += 1
                label = "duplicate terminal event" if request_id in completed else "unmatched terminal event"
                error(f"{request_id}: {label}")
                continue
            completed.add(request_id)
            if event == "error":
                transport_errors += 1
                error(f"{request_id}: transport error ({record.get('error_type', 'unknown')})")
                continue
            response_count += 1
            if record.get("status") != 200:
                error(f"{request_id}: unsuccessful HTTP response ({record.get('status')})")
                continue
            try:
                body = record["body"]
                choice = body["choices"][0]
                if choice["finish_reason"] != "stop":
                    error(f"{request_id}: finish_reason={choice['finish_reason']}")
                message = choice["message"]
                content = message["content"]
                if re.search(r"\s{256,}", content):
                    error(f"{request_id}: repeated whitespace")
                usage = body.get("usage", {})
                tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                reasoning_tokens += tokens
                separated = set()
                for field in reasoning_fields:
                    value = message.get(field)
                    if value:
                        reasoning_fields[field] += 1
                        if not isinstance(value, str):
                            error(f"{request_id}: non-string {field}")
                            continue
                        separated.add(value)
                        if re.search(r"\s{256,}", value):
                            error(f"{request_id}: repeated whitespace in {field}")
                if separated:
                    reasoning_responses += 1
                    characters = sum(len(value) for value in separated)
                    reasoning_characters += characters
                    max_reasoning_characters = max(max_reasoning_characters, characters)
                if not thinking and (tokens or separated):
                    error(f"{request_id}: unexpected reasoning output")
                max_completion = max(max_completion, usage.get("completion_tokens", 0) or 0)
                data = json.loads(content)
                Draft202012Validator(schema).validate(data)
                if not data["facts"]:
                    empty_facts_count += 1
                    if len(empty_facts_request_ids) < 20:
                        empty_facts_request_ids.append(request_id)
                facts += len(data["facts"])
                for fact in data["facts"]:
                    if not str(fact.get("what", "")).strip():
                        error(f"{request_id}: empty fact text")
                if len(samples) < 3:
                    samples.extend(str(f.get("what", ""))[:500] for f in data["facts"][:3 - len(samples)])
            except Exception as exc:
                error(f"{request_id}: invalid facts response: {type(exc).__name__}")
    for request_id in pending:
        error(f"{request_id}: missing terminal response")
    if not request_count or not facts:
        error("Expected strict facts requests and nonempty extracted facts")
    if require_separated_reasoning and not reasoning_responses:
        error("Expected separated reasoning in reasoning or reasoning_content")
    return {
        "strict_facts_requests": request_count,
        "http_responses": response_count,
        "transport_errors": transport_errors,
        "missing_terminal_responses": len(pending),
        "unmatched_terminal_events": unmatched_events,
        "extracted_facts": facts,
        "empty_facts_responses": empty_facts_count,
        "empty_facts_request_ids_for_source_review": empty_facts_request_ids,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_responses": reasoning_responses,
        "reasoning_field_responses": reasoning_fields,
        "reasoning_characters": reasoning_characters,
        "max_reasoning_characters": max_reasoning_characters,
        "require_separated_reasoning": require_separated_reasoning,
        "max_completion_tokens": max_completion,
        "fact_samples_for_source_review": samples,
        "error_count": error_count,
        "error_sample_limit": 20,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", choices=("on", "off", "native"), default="off",
                        help="native: no thinking override and no reasoning output")
    parser.add_argument("--require-separated-reasoning", action="store_true")
    parser.add_argument("--completion-cap", type=int, default=16384)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    expected_thinking = None if args.thinking == "native" else args.thinking == "on"
    summary = validate_trace(args.trace, args.model, expected_thinking, args.completion_cap,
                             args.require_separated_reasoning)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["error_count"]:
        raise SystemExit("Retain extraction trace validation failed")


if __name__ == "__main__":
    main()
