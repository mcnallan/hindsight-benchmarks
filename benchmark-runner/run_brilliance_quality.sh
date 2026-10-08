#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_brilliance_quality.sh RUN_DIR}
PYTHON="$RUNNER_DIR/.venv/bin/python"
export PYTHONUNBUFFERED=1
export QUALITY_RETAIN_API_KEY=EMPTY
export QUALITY_JUDGE_MODEL=${QUALITY_JUDGE_MODEL:-gpt-5.6-luna}
export QUALITY_RETRIEVAL_API_KEY
QUALITY_RETRIEVAL_API_KEY=${QUALITY_RETRIEVAL_API_KEY:-$(sed -n 's/^HINDSIGHT_API_LLM_API_KEY=//p' /home/blake/Projects/agents-mono/platform/hindsight/.env | head -1)}
: "${CODEX_LB_TOKEN:?CODEX_LB_TOKEN is required}"
: "${QUALITY_RETRIEVAL_API_KEY:?QUALITY_RETRIEVAL_API_KEY is required}"
mkdir -p "$RUN_DIR"
STATUS_FILE="$RUN_DIR/status.tsv"
LOG_FILE="$RUN_DIR/brilliance.log"
TRACE="$RUNNER_DIR/../results/traces/lfm25-brilliance-$(basename -- "$RUN_DIR").ndjson"
printf 'model\tphase\tstate\ttimestamp\n' >"$STATUS_FILE"
record() {
  printf '%s\t%s\t%s\t%s\n' brilliance "$1" "$2" "$(date --iso-8601=seconds)" >>"$STATUS_FILE"
}
CAPTURE_PORT=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1]); s.close()')
"$PYTHON" "$RUNNER_DIR/capture_openai_proxy.py" \
  --listen-port "$CAPTURE_PORT" --upstream http://192.168.39.245:8125 \
  --output "$TRACE" >"$RUN_DIR/proxy.log" 2>&1 &
proxy_pid=$!
trap 'kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$proxy_pid" 2>/dev/null; then
  record setup capture_proxy_failed
  exit 1
fi

COMMON_ARGS=(
  --label 'LFM2.5 Brilliance (local, logic thinking, BGE/MiniLM retrieval)'
  --result-model-id lfm25-brilliance-logic-bge-minilm
  --retain-model lfm25-brilliance
  --retain-base-url "http://host.docker.internal:$CAPTURE_PORT/v1"
  --retain-preflight-url http://192.168.39.245:8125/v1
  --retain-upstream-base-url http://192.168.39.245:8125/v1
  --retain-wire-trace "$TRACE"
  --extra-body-profile brilliance-logic
  --retrieval-profile bge-minilm
  --retain-concurrency 8
  --retain-max-completion-tokens 16384
  --strict-retain-schema
)
record smoke running
"$PYTHON" "$RUNNER_DIR/run_local_quality.py" "${COMMON_ARGS[@]}" \
  --max-conversations 1 --max-questions 2 --no-save >"$LOG_FILE" 2>&1
smoke_rc=$?
if (( smoke_rc != 0 )); then
  record smoke "failed:$smoke_rc"
  exit "$smoke_rc"
fi
"$PYTHON" "$RUNNER_DIR/validate_brilliance_trace.py" "$TRACE" \
  --summary "$RUN_DIR/smoke_trace_summary.json" >>"$LOG_FILE" 2>&1
trace_rc=$?
if (( trace_rc != 0 )); then
  record smoke "trace_failed:$trace_rc"
  exit "$trace_rc"
fi
record smoke passed
record full running
"$PYTHON" "$RUNNER_DIR/run_local_quality.py" "${COMMON_ARGS[@]}" >>"$LOG_FILE" 2>&1
full_rc=$?
if (( full_rc == 0 )); then
  record full passed
else
  record full "failed:$full_rc"
fi
exit "$full_rc"
