#!/usr/bin/env python3
"""Run the current BEAM retain-quality benchmark against a local LLM route."""

import argparse
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from hindsight_benchmark.quality import QualityBenchmark
from run_all_quality import BASE_CONFIG

RUNNER_DIR = Path(__file__).resolve().parent
COMPOSE_FILE = RUNNER_DIR / "compose.local-quality.yml"
TRACE_LOG_DIR = RUNNER_DIR.parent / "results" / "traces" / "hindsight"
JUDGE_BASE_URL = "http://127.0.0.1:2455/v1"
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "medium"
HINDSIGHT_VERSION = "0.9.2"

PRODUCTION_LIKE_CONFIG = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": "http://host.docker.internal:4000/v1",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "embedding",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS": "768",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE": "8",
    "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
    "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": "http://host.docker.internal:4000/v1/rerank",
    "HINDSIGHT_API_RERANKER_COHERE_MODEL": "reranker",
    "HINDSIGHT_API_RERANKER_COHERE_TIMEOUT": "60",
    "HINDSIGHT_API_LLM_REASONING_EFFORT": "low",
    "HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT": "medium",
    "HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT": "low",
}

NEMOTRON_EXTRA_BODY_CONFIG = {
    "HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY": (
        '{"extra_body":{"nvext":{"max_thinking_tokens":512},'
        '"chat_template_kwargs":{"enable_thinking":true}}}'
    ),
    "HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY": (
        '{"extra_body":{"nvext":{"max_thinking_tokens":1024},'
        '"chat_template_kwargs":{"enable_thinking":true}}}'
    ),
    "HINDSIGHT_API_CONSOLIDATION_LLM_EXTRA_BODY": (
        '{"extra_body":{"nvext":{"max_thinking_tokens":512},'
        '"chat_template_kwargs":{"enable_thinking":true}}}'
    ),
}

# Ling 3.0 Tiny otherwise emits its private reasoning as ordinary completion
# text, which exhausts the retain request timeout before its structured answer.
LING_NO_THINKING_EXTRA_BODY_CONFIG = {
    "HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY": (
        '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
    ),
    "HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY": (
        '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
    ),
    "HINDSIGHT_API_CONSOLIDATION_LLM_EXTRA_BODY": (
        '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
    ),
}

SECRET_KEYS = {
    "HINDSIGHT_API_LLM_API_KEY",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
    "HINDSIGHT_API_RERANKER_COHERE_API_KEY",
}


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _model_ids(base_url: str, api_key: str) -> set[str]:
    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return {item["id"] for item in response.json().get("data", [])}


def _effective_config(
    args, litellm_api_key: str, retain_api_key: str
) -> dict[str, str]:
    config = {
        **BASE_CONFIG,
        **PRODUCTION_LIKE_CONFIG,
        **(
            NEMOTRON_EXTRA_BODY_CONFIG
            if args.extra_body_profile == "nemotron"
            else LING_NO_THINKING_EXTRA_BODY_CONFIG
            if args.extra_body_profile == "ling-no-thinking"
            else {}
        ),
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_BASE_URL": args.retain_base_url,
        "HINDSIGHT_API_LLM_MODEL": args.retain_model,
        "HINDSIGHT_API_LLM_API_KEY": retain_api_key,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": litellm_api_key,
        "HINDSIGHT_API_RERANKER_COHERE_API_KEY": litellm_api_key,
    }
    if args.retain_concurrency is not None:
        config["HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT"] = str(
            args.retain_concurrency
        )
    config.pop("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", None)
    return config


def _print_config(config: dict[str, str]) -> None:
    print("Effective Hindsight environment:")
    for key in sorted(config):
        value = "REDACTED" if key in SECRET_KEYS else config[key]
        print(f"  {key}={value}")
    print(f"  QUALITY_JUDGE_BASE_URL={JUDGE_BASE_URL}")
    print(f"  QUALITY_JUDGE_MODEL={JUDGE_MODEL}")
    print(f"  QUALITY_JUDGE_REASONING_EFFORT={JUDGE_REASONING_EFFORT}")
    print("  CODEX_LB_TOKEN=REDACTED")


def _write_env_file(config: dict[str, str]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="hindsight-quality-", suffix=".env", delete=False
    ) as handle:
        for key, value in sorted(config.items()):
            handle.write(f"{key}={value}\n")
    path = Path(handle.name)
    path.chmod(0o600)
    return path


def _compose(project: str, env: dict[str, str], *args: str, check: bool = True):
    return subprocess.run(
        ["docker", "compose", "-p", project, "-f", str(COMPOSE_FILE), *args],
        cwd=RUNNER_DIR,
        env=env,
        check=check,
    )


def _save_hindsight_logs(
    project: str,
    env: dict[str, str],
    run_id: str,
    model_id: str,
    secrets: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(COMPOSE_FILE),
            "logs",
            "--no-color",
            "hindsight",
        ],
        cwd=RUNNER_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    content = completed.stdout + completed.stderr
    for secret in secrets:
        if secret:
            content = content.replace(secret, "REDACTED")
    TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_LOG_DIR / f"{_slug(model_id)}-{run_id}.log"
    path.write_text(content)
    print(f"Hindsight logs saved to {path}")


def _validate_smoke_result(result: dict) -> None:
    checks = {
        "two questions evaluated": result.get("total") == 2,
        "stored_fact_tokens > 0": (result.get("stored_fact_tokens") or 0) > 0,
        "recall returned facts": (result.get("recalled_fact_count") or 0) > 0,
        "no generation errors": result.get("generation_errors") == 0,
        "no judge errors": result.get("judge_errors") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Smoke validation failed: {', '.join(failed)}")
    print("Smoke validation passed: " + "; ".join(checks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--retain-model", required=True)
    parser.add_argument(
        "--retain-base-url", default="http://host.docker.internal:4000/v1"
    )
    parser.add_argument(
        "--retain-preflight-url", default="http://127.0.0.1:4000/v1"
    )
    parser.add_argument("--provider-id", default="local")
    parser.add_argument(
        "--extra-body-profile",
        choices=("nemotron", "ling-no-thinking", "none"),
        default="nemotron",
    )
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--retain-concurrency", type=int, choices=range(1, 9))
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    litellm_api_key = os.environ.get("QUALITY_LITELLM_API_KEY")
    retain_api_key = os.environ.get("QUALITY_RETAIN_API_KEY", litellm_api_key)
    codex_lb_token = os.environ.get("CODEX_LB_TOKEN")
    if not litellm_api_key:
        raise SystemExit("QUALITY_LITELLM_API_KEY is required")
    if not codex_lb_token:
        raise SystemExit("CODEX_LB_TOKEN is required")

    retain_models = _model_ids(args.retain_preflight_url, retain_api_key)
    if args.retain_model not in retain_models:
        raise RuntimeError(
            f"Retain model {args.retain_model!r} not found at {args.retain_preflight_url}"
        )
    judge_models = _model_ids(JUDGE_BASE_URL, codex_lb_token)
    if JUDGE_MODEL not in judge_models:
        raise RuntimeError(f"Judge model {JUDGE_MODEL!r} not found at {JUDGE_BASE_URL}")
    print(f"Verified retain model: {args.retain_model}")
    print(f"Verified judge model: {JUDGE_MODEL}")

    benchmark = QualityBenchmark(
        openai_base_url=JUDGE_BASE_URL,
        openai_api_key=codex_lb_token,
        openai_model=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING_EFFORT,
    )
    benchmark.verify_benchmark_calls()

    config = _effective_config(args, litellm_api_key, retain_api_key)
    _print_config(config)
    if args.preflight_only:
        return

    run_id = f"{int(time.time())}-{os.getpid()}"
    project = f"quality-{_slug(args.retain_model)[:24]}-{run_id}"
    port = _available_port()
    env_file = _write_env_file(config)
    compose_env = {
        **os.environ,
        "QUALITY_CONFIG_ENV_FILE": str(env_file),
        "QUALITY_HINDSIGHT_PORT": str(port),
        "QUALITY_HINDSIGHT_VERSION": HINDSIGHT_VERSION,
    }
    api_url = f"http://127.0.0.1:{port}"
    try:
        _compose(project, compose_env, "up", "-d", "--wait")
        result = benchmark.run(
            model_id=args.retain_model,
            provider_id=args.provider_id,
            api_url=api_url,
            max_questions_per_conversation=args.max_questions,
            max_conversations=args.max_conversations,
            save=not args.no_save,
            run_metadata={
                "model_label": args.label,
                "retain_llm_base_url": args.retain_base_url,
                "retain_llm_concurrency": int(
                    config["HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT"]
                ),
                "retain_batch_tokens": int(
                    config["HINDSIGHT_API_RETAIN_BATCH_TOKENS"]
                ),
                "embedding_provider": config["HINDSIGHT_API_EMBEDDINGS_PROVIDER"],
                "embedding_model": config["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"],
                "embedding_dimensions": int(
                    config["HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS"]
                ),
                "reranker_provider": config["HINDSIGHT_API_RERANKER_PROVIDER"],
                "reranker_model": config["HINDSIGHT_API_RERANKER_COHERE_MODEL"],
            },
        )
        print("Result JSON:")
        print(json.dumps(result, indent=2))
        if args.max_conversations == 1 and args.max_questions == 2:
            _validate_smoke_result(result)
    finally:
        _save_hindsight_logs(
            project,
            compose_env,
            run_id,
            args.retain_model,
            (litellm_api_key, retain_api_key, codex_lb_token),
        )
        _compose(project, compose_env, "down", "--volumes", check=False)
        env_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
