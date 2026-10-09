#!/usr/bin/env bash
# Each endpoint runs in an independent temporary Hindsight/database stack.
set -uo pipefail
umask 077
RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:?usage: run_gpu1_quality.sh qwen35-9b|mellum21-12b|mellum2-instruct RUN_DIR}
RUN_DIR=${2:?usage: run_gpu1_quality.sh PROFILE RUN_DIR}
MODE=${3:-auto}
case "$MODE" in auto|smoke|full) ;; *) echo 'Mode must be auto, smoke, or full' >&2; exit 2 ;; esac
PYTHON="$RUNNER_DIR/.venv/bin/python"
EXTRA_BODY_PROFILE=no-thinking
THINKING_MODE=off
TRACE_REASONING_ARGS=()
case "$PROFILE" in
  qwen35-9b)
    MODEL=qwen3.5-9b-test
    RESULT_ID=qwen3.5-9b-no-thinking-int4-bge-minilm
    LABEL='Qwen3.5 9B (local INT4, no thinking, BGE/MiniLM)'
    UPSTREAM=http://192.168.39.244:8129 ;;
  mellum21-12b)
    MODEL=mellum2.1-12b-test
    RESULT_ID=mellum2.1-12b-thinking-int4-bge-minilm
    LABEL='Mellum 2.1 12B (local INT4, thinking, BGE/MiniLM)'
    EXTRA_BODY_PROFILE=thinking
    THINKING_MODE=on
    TRACE_REASONING_ARGS=(--require-separated-reasoning)
    UPSTREAM=http://192.168.39.244:8121 ;;
  mellum2-instruct)
    MODEL=mellum2-instruct-test
    RESULT_ID=mellum2-12b-instruct-int4-bge-minilm
    LABEL='Mellum 2 12B Instruct (local INT4, native nonreasoning, BGE/MiniLM)'
    EXTRA_BODY_PROFILE=native-instruct
    THINKING_MODE=native
    UPSTREAM=http://192.168.39.244:8122 ;;
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac
if [[ $PROFILE == mellum* && $MODE == auto ]]; then
  echo 'Mellum requires smoke plus the manual source-review handoff before full' >&2
  exit 2
fi
: "${CODEX_LB_TOKEN:?CODEX_LB_TOKEN is required}"
: "${QUALITY_MODEL_PROVENANCE_FILE:?Set a sanitized selected-deployment JSON after performance/sanity validation}"
export PYTHONUNBUFFERED=1
export QUALITY_RETAIN_API_KEY=${QUALITY_RETAIN_API_KEY:-EMPTY}
export QUALITY_JUDGE_MODEL=${QUALITY_JUDGE_MODEL:-gpt-5.6-luna}
export QUALITY_RETRIEVAL_API_KEY
if [[ -z ${QUALITY_RETRIEVAL_API_KEY:-} ]]; then
  QUALITY_RETRIEVAL_API_KEY=$("$PYTHON" - <<'PY'
from pathlib import Path
from dotenv import dotenv_values
path = Path.home() / "Projects/agents-mono/platform/hindsight/.env"
print(dotenv_values(path).get("HINDSIGHT_API_LLM_API_KEY", ""))
PY
)
fi
: "${QUALITY_RETRIEVAL_API_KEY:?QUALITY_RETRIEVAL_API_KEY is required}"
if [[ -e "$RUN_DIR/status.tsv" && $MODE != full ]]; then
  echo "Run directory already contains status; use a fresh directory" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
if [[ $MODE == full ]]; then
  [[ $(cat "$RUN_DIR/judge-model.txt") == "$QUALITY_JUDGE_MODEL" ]] || { echo 'Judge model differs from smoke' >&2; exit 2; }
  cmp -s -- "$QUALITY_MODEL_PROVENANCE_FILE" "$RUN_DIR/deployment.json" || { echo 'Deployment recipe differs from smoke' >&2; exit 2; }
  if [[ $PROFILE == mellum* ]]; then
    "$PYTHON" - "$RUNNER_DIR" "$RUN_DIR" <<'PY' || exit 2
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from handoff_gpu1_quality import check_review
check_review(Path(sys.argv[2]))
PY
  fi
  "$PYTHON" - "$RUN_DIR/smoke_trace_summary.json" "$RUN_DIR/status.tsv" "$PROFILE" <<'PY' || exit 2
import json, sys
summary = json.load(open(sys.argv[1]))
rows = [line.strip().split('\t') for line in open(sys.argv[2])]
assert summary.get('error_count', len(summary['errors'])) == 0 and summary['extracted_facts'] > 0, 'Smoke trace gate did not pass'
assert [sys.argv[3], 'smoke', 'passed'] in [row[:3] for row in rows], 'Smoke did not pass'
assert not any(row[1] == 'full' for row in rows[1:]), 'Full phase already started; use a new run'
PY
else
  cp -- "$QUALITY_MODEL_PROVENANCE_FILE" "$RUN_DIR/deployment.json" || exit 2
  printf '%s\n' "$QUALITY_JUDGE_MODEL" >"$RUN_DIR/judge-model.txt"
fi
"$PYTHON" -m json.tool "$RUN_DIR/deployment.json" >/dev/null || exit 2
printf '%s\n' "$MODEL" >"$RUN_DIR/retain-model.txt"
STATUS_FILE="$RUN_DIR/status.tsv"
LOG_FILE="$RUN_DIR/quality.log"
if [[ $MODE != full ]]; then printf 'model\tphase\tstate\ttimestamp\n' >"$STATUS_FILE"; fi
record() { printf '%s\t%s\t%s\t%s\n' "$PROFILE" "$1" "$2" "$(date --iso-8601=seconds)" >>"$STATUS_FILE"; }
TRACE_DIR="$RUNNER_DIR/../results/traces/$PROFILE-$(basename -- "$RUN_DIR")"
mkdir -p "$TRACE_DIR"
printf '%s\n' "$TRACE_DIR" >"$RUN_DIR/trace-directory.txt"
proxy_pid=
cleanup() {
  if [[ -n $proxy_pid ]]; then
    kill "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'record batch interrupted; exit 130' INT
trap 'record batch terminated; exit 143' TERM
COMMON_ARGS=(
  --label "$LABEL" --result-model-id "$RESULT_ID" --retain-model "$MODEL"
  --retain-preflight-url "$UPSTREAM/v1" --retain-upstream-base-url "$UPSTREAM/v1"
  --retain-deployment-metadata "$RUN_DIR/deployment.json"
  --extra-body-profile "$EXTRA_BODY_PROFILE" --retrieval-profile bge-minilm
  --retain-concurrency 8 --retain-max-completion-tokens 16384 --strict-retain-schema
)
run_phase() {
  local phase=$1
  shift
  local trace="$TRACE_DIR/$phase.ndjson"
  local capture_port
  capture_port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0",0)); print(s.getsockname()[1]); s.close()')
  "$PYTHON" "$RUNNER_DIR/capture_openai_proxy.py" \
    --listen-port "$capture_port" --upstream "$UPSTREAM" --output "$trace" >"$RUN_DIR/proxy-$phase.log" 2>&1 &
  proxy_pid=$!
  sleep 1
  if ! kill -0 "$proxy_pid" 2>/dev/null; then record "$phase" capture_proxy_failed; return 1; fi
  record "$phase" running
  "$PYTHON" "$RUNNER_DIR/run_local_quality.py" "${COMMON_ARGS[@]}" \
    --retain-base-url "http://host.docker.internal:$capture_port/v1" --retain-wire-trace "$trace" "$@" >>"$LOG_FILE" 2>&1
  local rc=$?
  cleanup
  proxy_pid=
  if (( rc != 0 )); then record "$phase" "failed:$rc"; return "$rc"; fi
  "$PYTHON" "$RUNNER_DIR/validate_retain_trace.py" "$trace" --model "$MODEL" --thinking "$THINKING_MODE" "${TRACE_REASONING_ARGS[@]}" \
    --summary "$RUN_DIR/${phase}_trace_summary.json" >>"$LOG_FILE" 2>&1
  rc=$?
  if (( rc != 0 )); then record "$phase" "trace_failed:$rc"; return "$rc"; fi
  record "$phase" passed
}
if [[ $MODE != full ]]; then
  run_phase smoke --max-conversations 1 --max-questions 2 --no-save || exit $?
fi
if [[ $MODE == smoke ]]; then exit 0; fi
run_phase full || exit $?
"$PYTHON" - "$RUNNER_DIR/../results/leaderboard/llm/local-$RESULT_ID.json" <<'PY' >>"$LOG_FILE" 2>&1
import json, sys
result = json.load(open(sys.argv[1]))['quality']
checks = {
    "80 questions": result.get("total") == 80,
    "4 conversations": len(result.get("sample_ids", [])) == 4,
    "complete result": result.get("partial") is False,
    "no generation errors": result.get("generation_errors") == 0,
    "no judge errors": result.get("judge_errors") == 0,
    "stored facts": (result.get("stored_fact_tokens") or 0) > 0,
    "recall returned facts": (result.get("recalled_fact_count") or 0) > 0,
}
failed = [label for label, passed in checks.items() if not passed]
if failed: raise SystemExit("Full result validation failed: " + ", ".join(failed))
print("Full result validation passed: " + "; ".join(checks))
PY
rc=$?
if (( rc != 0 )); then record result "failed:$rc"; exit "$rc"; fi
record result passed
