#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_thinking_variants_batch.sh RUN_DIR}
PYTHON="$RUNNER_DIR/.venv/bin/python"
RUNNER="$RUNNER_DIR/run_local_quality.py"
HINDSIGHT_ENV=/home/blake/Projects/agents-mono/platform/hindsight/.env
STATUS_FILE="$RUN_DIR/status.tsv"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR"
printf 'model\tphase\tstate\ttimestamp\n' >"$STATUS_FILE"

record() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date --iso-8601=seconds)" >>"$STATUS_FILE"
}

run_job() {
  local name=$1
  shift
  local log="$RUN_DIR/$name.log"

  record "$name" smoke running
  {
    echo "[$(date --iso-8601=seconds)] Starting $name smoke test"
    "$PYTHON" "$RUNNER" "$@" --max-conversations 1 --max-questions 2 --no-save
  } >"$log" 2>&1
  local smoke_rc=$?
  if (( smoke_rc != 0 )); then
    record "$name" smoke "failed:$smoke_rc"
    return "$smoke_rc"
  fi
  record "$name" smoke passed

  record "$name" full running
  {
    echo "[$(date --iso-8601=seconds)] Smoke passed; starting all 4 conversations and 80 questions"
    "$PYTHON" "$RUNNER" "$@"
  } >>"$log" 2>&1
  local full_rc=$?
  if (( full_rc == 0 )); then
    record "$name" full passed
  else
    record "$name" full "failed:$full_rc"
  fi
  return "$full_rc"
}

run_ling_thinking() {
  export QUALITY_RETAIN_API_KEY=local
  run_job ling-thinking \
    --label 'Ling 3.0 Tiny (local, thinking)' \
    --result-model-id ling-3.0-tiny-thinking \
    --retain-model ling-3.0-tiny \
    --retain-base-url http://192.168.39.245:8000/v1 \
    --retain-preflight-url http://192.168.39.245:8000/v1 \
    --extra-body-profile ling-thinking \
    --retain-concurrency 8 \
    --retain-max-completion-tokens 16384
}

run_nemotron_no_thinking() {
  QUALITY_RETAIN_API_KEY=$(sed -n 's/^HINDSIGHT_API_LLM_API_KEY=//p' "$HINDSIGHT_ENV" | head -1)
  export QUALITY_RETAIN_API_KEY
  if [[ -z $QUALITY_RETAIN_API_KEY ]]; then
    record nemotron-no-thinking setup missing_api_key
    return 1
  fi
  run_job nemotron-no-thinking \
    --label 'Nemotron 3.5 Lightning (no thinking)' \
    --result-model-id nemotron-lightning-no-thinking \
    --retain-model nemotron-lightning \
    --retain-base-url http://host.docker.internal:4000/v1 \
    --retain-preflight-url http://127.0.0.1:4000/v1 \
    --extra-body-profile nemotron-no-thinking \
    --retain-concurrency 8 \
    --strict-retain-schema \
    --retain-max-completion-tokens 4096
}

record supervisor batch running
overall=0
run_ling_thinking &
ling_pid=$!
run_nemotron_no_thinking &
nemotron_pid=$!
record supervisor batch "running:ling=$ling_pid,nemotron=$nemotron_pid"
wait "$ling_pid" || overall=1
wait "$nemotron_pid" || overall=1
record supervisor batch "complete:$overall"
exit "$overall"
