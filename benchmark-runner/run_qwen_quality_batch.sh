#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_qwen_quality_batch.sh RUN_DIR}
PYTHON="$RUNNER_DIR/.venv/bin/python"
RUNNER="$RUNNER_DIR/run_local_quality.py"
STATUS_FILE="$RUN_DIR/status.tsv"
LOG_FILE="$RUN_DIR/qwen.log"
export PYTHONUNBUFFERED=1
export QUALITY_RETAIN_API_KEY=local

mkdir -p "$RUN_DIR"
printf 'model\tphase\tstate\ttimestamp\n' >"$STATUS_FILE"

record() {
  printf '%s\t%s\t%s\t%s\n' qwen "$1" "$2" "$(date --iso-8601=seconds)" >>"$STATUS_FILE"
}

COMMON_ARGS=(
  --label 'Qwen3.5 4B (local, no thinking, strict extraction)'
  --result-model-id qwen3.5-4b-no-thinking-strict
  --retain-model qwen3.5-4b
  --retain-base-url http://192.168.39.245:8120/v1
  --retain-preflight-url http://192.168.39.245:8120/v1
  --extra-body-profile qwen-no-thinking
  --retain-concurrency 8
  --retain-max-completion-tokens 16384
  --strict-retain-schema
)

record smoke running
{
  echo "[$(date --iso-8601=seconds)] Starting Qwen 1-conversation/2-question smoke test"
  "$PYTHON" "$RUNNER" "${COMMON_ARGS[@]}" --max-conversations 1 --max-questions 2 --no-save
} >"$LOG_FILE" 2>&1
smoke_rc=$?
if (( smoke_rc != 0 )); then
  record smoke "failed:$smoke_rc"
  exit "$smoke_rc"
fi
record smoke passed

record full running
{
  echo "[$(date --iso-8601=seconds)] Smoke passed; starting all 4 conversations and 80 questions"
  "$PYTHON" "$RUNNER" "${COMMON_ARGS[@]}"
} >>"$LOG_FILE" 2>&1
full_rc=$?
if (( full_rc == 0 )); then
  record full passed
else
  record full "failed:$full_rc"
fi
exit "$full_rc"
