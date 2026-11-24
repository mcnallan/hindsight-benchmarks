#!/usr/bin/env python3
"""Replay captured retain extraction calls without changing their prompts."""

import argparse
import copy
import json
import os
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
        "reasoning_content_present": bool(message.get("reasoning_content") or message.get("reasoning")),
        "reasoning_tokens": ((response.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
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
    os.umask(0o077)
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
    parser.add_argument("--thinking", choices=("off", "on", "captured"), default="off",
                        help="Legacy default is off; captured preserves the original request's template flags")
    parser.add_argument("--temperature", type=float, help="Override only sampling, preserving captured prompts/schema")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.temperature is not None and args.temperature < 0:
        parser.error("temperature must be nonnegative")
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        parser.error("repetition penalty must be positive")

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
        body = copy.deepcopy(row["body"])
        client = OpenAI(base_url=args.base_url, api_key="local", timeout=600, max_retries=0)
        standard = {"model", "messages", "temperature", "max_tokens", "response_format", "reasoning_effort", "seed", "stream", "stream_options"}
        extra_body = {key: value for key, value in body.items() if key not in standard}
        if args.thinking != "captured":
            extra_body["chat_template_kwargs"] = {**extra_body.get("chat_template_kwargs", {}),
                                                 "enable_thinking": args.thinking == "on"}
        if args.repetition_penalty is not None:
            extra_body["repetition_penalty"] = args.repetition_penalty
        kwargs = {
            "model": args.model,
            "messages": body["messages"],
            "temperature": args.temperature if args.temperature is not None else body["temperature"],
            "max_tokens": args.max_tokens or body["max_tokens"],
            "response_format": body["response_format"],
            "extra_body": extra_body,
        }
        if args.seed is not None or body.get("seed") is not None:
            kwargs["seed"] = args.seed if args.seed is not None else body["seed"]
        if args.reasoning_effort != "omit":
            kwargs["reasoning_effort"] = args.reasoning_effort
        result = {"request_id": row["request_id"], "mode": args.mode,
                  "effective_request": {**kwargs, **extra_body}}
        result["effective_request"].pop("extra_body")
        try:
            response = client.chat.completions.create(**kwargs).model_dump()
            result.update(classify(response))
            result["response"] = response
            if args.mode == "json_schema":
                from jsonschema import Draft202012Validator
                try:
                    data = json.loads(response["choices"][0]["message"]["content"])
                    Draft202012Validator(body["response_format"]["json_schema"]["schema"]).validate(data)
                    result["strict_schema_valid"] = True
                except Exception as exc:
                    result["strict_schema_valid"] = False
                    result["schema_error_type"] = type(exc).__name__
        except Exception as exc:
            result["classification"] = "api_error"
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)[:500]
        return result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, args.output.open("a") as output:
        os.chmod(args.output, 0o600)
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
