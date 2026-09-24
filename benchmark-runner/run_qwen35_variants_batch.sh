#!/usr/bin/env bash
set -uo pipefail

RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?usage: run_qwen35_variants_batch.sh RUN_DIR}
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

run_phase() {
  local name=$1 phase=$2
  local model url profile label result_id
  case "$name" in
    base-thinking)
      model=qwen3.5-4b
      url=http://192.168.39.244:8120/v1
      profile=qwen-thinking
      label='Qwen3.5 4B (local, thinking, strict extraction)'
      result_id=qwen3.5-4b-thinking-strict
      ;;
    distilled-thinking)
      model=qwen3.5-4b-distilled
      url=http://192.168.39.245:8121/v1
      profile=qwen-thinking
      label='Qwen3.5 4B Opus Reasoning Distilled v2 (local, thinking, strict extraction)'
      result_id=qwen3.5-4b-distilled-thinking-strict
      ;;
    distilled-no-thinking)
      model=qwen3.5-4b-distilled
      url=http://192.168.39.245:8121/v1
      profile=qwen-distilled-no-thinking
      label='Qwen3.5 4B Opus Reasoning Distilled v2 (local, no thinking, strict extraction)'
      result_id=qwen3.5-4b-distilled-no-thinking-strict
      ;;
    *) echo "Unknown model profile: $name" >&2; return 2 ;;
  esac

  local args=(
    --label "$label"
    --result-model-id "$result_id"
    --retain-model "$model"
    --retain-base-url "$url"
    --retain-preflight-url "$url"
    --extra-body-profile "$profile"
    --retain-concurrency 8
    --retain-max-completion-tokens 16384
    --strict-retain-schema
  )
  if [[ $phase == smoke ]]; then
    args+=(--max-conversations 1 --max-questions 2 --no-save)
  fi

  record "$name" "$phase" running
  {
    echo "[$(date --iso-8601=seconds)] Starting $name $phase"
    "$PYTHON" "$RUNNER" "${args[@]}"
  } >>"$RUN_DIR/$name.log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    record "$name" "$phase" passed
  else
    record "$name" "$phase" "failed:$rc"
  fi
  return "$rc"
}

# GPU1 and GPU2 are independent. Keep the two GPU2 variants serial, and do not
# start any full run until all three integration smokes have passed.
run_phase base-thinking smoke &
base_smoke_pid=$!
run_phase distilled-thinking smoke &
distilled_smoke_pid=$!
wait "$base_smoke_pid"
base_smoke_rc=$?
wait "$distilled_smoke_pid"
distilled_smoke_rc=$?
if (( base_smoke_rc != 0 || distilled_smoke_rc != 0 )); then
  record batch smoke failed
  exit 1
fi
distilled_no_thinking_smoke_rc=0
run_phase distilled-no-thinking smoke || distilled_no_thinking_smoke_rc=$?
if (( distilled_no_thinking_smoke_rc == 0 )); then
  record batch smoke passed
else
  # A failed no-thinking smoke must not suppress the two independent,
  # already-passed thinking runs. It only disqualifies its own full run.
  record batch smoke partial
fi

run_phase base-thinking full &
base_full_pid=$!
run_phase distilled-thinking full &
distilled_full_pid=$!
wait "$distilled_full_pid"
distilled_full_rc=$?
if (( distilled_no_thinking_smoke_rc == 0 )); then
  run_phase distilled-no-thinking full
  distilled_no_thinking_rc=$?
else
  record distilled-no-thinking full skipped_after_failed_smoke
  distilled_no_thinking_rc=$distilled_no_thinking_smoke_rc
fi
wait "$base_full_pid"
base_full_rc=$?
if (( base_full_rc == 0 && distilled_full_rc == 0 && distilled_no_thinking_rc == 0 )); then
  record batch full passed
else
  record batch full failed
  exit 1
fi
