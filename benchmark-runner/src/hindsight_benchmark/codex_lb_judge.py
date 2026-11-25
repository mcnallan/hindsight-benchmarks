"""Structured grading through the configured codex-lb endpoint."""

import json
import os
import time
import urllib.error
import urllib.request


def generate_json(model: str, prompt: str, schema: dict) -> dict:
    url = os.environ["HS_BENCH_CODEX_LB_URL"].rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": os.getenv("HS_BENCH_JUDGE_REASONING_EFFORT", "high"),
        "max_completion_tokens": 16384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": schema},
        },
    }
    request = urllib.request.Request(url, json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {os.environ['CODEX_LB_TOKEN']}",
        "Content-Type": "application/json",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                choice = json.load(response)["choices"][0]
            break
        except urllib.error.HTTPError as exc:
            if attempt == 2 or exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"codex-lb grading failed with HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    if choice["finish_reason"] != "stop":
        raise RuntimeError(f"codex-lb grading ended with {choice['finish_reason']}")
    verdict = json.loads(choice["message"]["content"])
    if not isinstance(verdict, dict) or not set(schema["required"]).issubset(verdict):
        raise RuntimeError("codex-lb returned an incomplete grading object")
    return verdict
