#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_jina_quality_batch.sh RUN_DIR}
PYTHON="$RUNNER_DIR/.venv/bin/python"
RUNNER="$RUNNER_DIR/run_local_quality.py"
STATUS_FILE="$RUN_DIR/status.tsv"
HINDSIGHT_ENV=/home/blake/Projects/agents-mono/platform/hindsight/.env
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
    "$PYTHON" "$RUNNER" "$@" \
      --max-conversations 1 \
      --max-questions 2 \
      --no-save
  } >"$log" 2>&1
  local smoke_rc=$?
  if (( smoke_rc != 0 )); then
    record "$name" smoke "failed:$smoke_rc"
    return "$smoke_rc"
  fi
  record "$name" smoke passed

  record "$name" full running
  {
    echo "[$(date --iso-8601=seconds)] Smoke passed; starting full 4-conversation/80-question run"
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

run_luna() {
  export QUALITY_RETAIN_API_KEY=${CODEX_LB_TOKEN:?CODEX_LB_TOKEN is required}
  run_job luna \
    --label 'GPT-5.6 Luna (local sanity check)' \
    --retain-model gpt-5.6-luna \
    --retain-base-url http://host.docker.internal:2455/v1 \
    --retain-preflight-url http://127.0.0.1:2455/v1 \
    --extra-body-profile none \
    --retain-concurrency 8
}

run_nemotron() {
  export QUALITY_RETAIN_API_KEY
  QUALITY_RETAIN_API_KEY=$(sed -n 's/^HINDSIGHT_API_LLM_API_KEY=//p' "$HINDSIGHT_ENV" | head -1)
  if [[ -z $QUALITY_RETAIN_API_KEY ]]; then
    record nemotron setup missing_api_key
    return 1
  fi
  run_job nemotron \
    --label 'Nemotron 3.5 Lightning' \
    --retain-model nemotron-lightning \
    --retain-base-url http://host.docker.internal:4000/v1 \
    --retain-preflight-url http://127.0.0.1:4000/v1 \
    --extra-body-profile nemotron \
    --retain-concurrency 8
}

run_ling() {
  export QUALITY_RETAIN_API_KEY=local
  run_job ling \
    --label 'Ling 3.0 Tiny' \
    --retain-model ling-3.0-tiny \
    --retain-base-url http://192.168.39.245:8000/v1 \
    --retain-preflight-url http://192.168.39.245:8000/v1 \
    --extra-body-profile ling-no-thinking \
    --retain-concurrency 8 \
    --retain-max-completion-tokens 4096
}

run_luna &
luna_pid=$!
run_nemotron &
nemotron_pid=$!
run_ling &
ling_pid=$!

record supervisor batch "running:luna=$luna_pid,nemotron=$nemotron_pid,ling=$ling_pid"

overall=0
wait "$luna_pid" || overall=1
wait "$nemotron_pid" || overall=1
wait "$ling_pid" || overall=1
record supervisor batch "complete:$overall"
exit "$overall"
