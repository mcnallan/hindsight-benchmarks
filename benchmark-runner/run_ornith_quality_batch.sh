#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_ornith_quality_batch.sh RUN_DIR}
PYTHON="$RUNNER_DIR/.venv/bin/python"
RUNNER="$RUNNER_DIR/run_local_quality.py"
STATUS_FILE="$RUN_DIR/status.tsv"
export PYTHONUNBUFFERED=1
export QUALITY_RETAIN_API_KEY=local

mkdir -p "$RUN_DIR"
printf 'model\tphase\tstate\ttimestamp\n' >"$STATUS_FILE"

record() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date --iso-8601=seconds)" >>"$STATUS_FILE"
}

COMMON_ARGS=(
  --retain-model ornith-1.5-9b
  --retain-base-url http://192.168.39.245:8120/v1
  --retain-preflight-url http://192.168.39.245:8120/v1
  --retain-concurrency 8
  --retain-max-completion-tokens 16384
  --strict-retain-schema
)

run_phase() {
  local name=$1 phase=$2
  local profile label result_id
  if [[ $name == no-thinking ]]; then
    profile=ornith-no-thinking
    label='Ornith 1.5 9B (local, no thinking, strict extraction)'
    result_id=ornith-1.5-9b-no-thinking-strict
  else
    profile=ornith-thinking
    label='Ornith 1.5 9B (local, thinking, strict extraction)'
    result_id=ornith-1.5-9b-thinking-strict
  fi
  local log="$RUN_DIR/$name.log"
  local args=(
    "${COMMON_ARGS[@]}"
    --label "$label"
    --result-model-id "$result_id"
    --extra-body-profile "$profile"
  )
  if [[ $phase == smoke ]]; then
    args+=(--max-conversations 1 --max-questions 2 --no-save)
  fi

  record "$name" "$phase" running
  {
    echo "[$(date --iso-8601=seconds)] Starting $name $phase"
    "$PYTHON" "$RUNNER" "${args[@]}"
  } >>"$log" 2>&1
  local run_rc=$?
  if (( run_rc == 0 )); then
    record "$name" "$phase" passed
  else
    record "$name" "$phase" "failed:$run_rc"
  fi
  return "$run_rc"
}

# Finish both integration smokes before any long full run. Keep full runs
# serial so the same GPU, retrieval sidecars, and judge have no cross-run load.
run_phase no-thinking smoke || exit 1
run_phase thinking smoke || exit 1
run_phase no-thinking full || exit 1
run_phase thinking full
