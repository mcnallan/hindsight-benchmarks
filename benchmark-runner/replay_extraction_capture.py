#!/usr/bin/env python3
"""Replay captured retain extraction calls without changing their prompts."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


def classify(response: dict) -> dict:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    result = {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "content_chars": len(content),
        "trimmed_chars": len(content.rstrip()),
    }
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        result["classification"] = "invalid_json"
        return result
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        result["classification"] = "wrong_shape"
        result["top_level_keys"] = list(parsed)[:10] if isinstance(parsed, dict) else []
    elif not all(isinstance(fact, dict) and isinstance(fact.get("what"), str) for fact in parsed["facts"]):
        result["classification"] = "malformed_facts"
        result["fact_count"] = len(parsed["facts"])
    else:
        result["classification"] = "valid_facts"
        result["fact_count"] = len(parsed["facts"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("json_object", "json_schema"), required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, help="Override the captured completion limit")
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()

    requests = [
        row
        for row in (json.loads(line) for line in args.capture.open())
        if row["event"] == "request"
        and row.get("body", {}).get("response_format", {}).get("type") == args.mode
    ]
    if args.limit is not None:
        requests = requests[: args.limit]
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["request_id"] for line in args.output.open()}
    pending = [row for row in requests if row["request_id"] not in completed]
    print(f"mode={args.mode} captured={len(requests)} pending={len(pending)}", flush=True)

    def run(row: dict) -> dict:
        body = row["body"]
        client = OpenAI(base_url=args.base_url, api_key="local", timeout=120, max_retries=0)
        kwargs = {
            "model": args.model,
            "messages": body["messages"],
            "temperature": body["temperature"],
            "max_tokens": args.max_tokens or body["max_tokens"],
            "response_format": body["response_format"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if args.reasoning_effort != "omit":
            kwargs["reasoning_effort"] = args.reasoning_effort
        result = {"request_id": row["request_id"], "mode": args.mode}
        try:
            response = client.chat.completions.create(**kwargs).model_dump()
            result.update(classify(response))
            result["response"] = response
        except Exception as exc:
            result["classification"] = "api_error"
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)[:500]
        return result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, args.output.open("a") as output:
        futures = [pool.submit(run, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"{index}/{len(pending)} {result['classification']} "
                f"finish={result.get('finish_reason')} tokens={result.get('completion_tokens')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
