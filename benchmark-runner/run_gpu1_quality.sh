#!/usr/bin/env bash
# Each endpoint runs in an independent temporary Hindsight/database stack.
set -uo pipefail
umask 077
RUNNER_DIR=${QUALITY_RUNNER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PROFILE=${1:?usage: run_gpu1_quality.sh PROFILE RUN_DIR [smoke|full|auto]}
RUN_DIR=${2:?usage: run_gpu1_quality.sh PROFILE RUN_DIR}
MODE=${3:-auto}
export QUALITY_JUDGE_MODEL=${QUALITY_JUDGE_MODEL:-gpt-6-sol}
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
  qwen36-35b-twu-low-latency)
    MODEL=qwen3.6-35b-test
    RESULT_ID=qwen3.6-35b-twu-low-latency-no-thinking-int4-bge-minilm
    LABEL='Qwen3.6 35B Twu LowLatency (local INT4, no thinking, BGE/MiniLM)'
    UPSTREAM=http://192.168.39.244:8136 ;;
  qwen36-35b-gsq-bf16kv)
    MODEL=qwen3.6-35b-gsq-test
    RESULT_ID=qwen3.6-35b-gsq-no-thinking-mixed-w2a16-w8a16-bf16kv-bge-minilm
    LABEL='Qwen3.6 35B GSQ (mixed W2A16/W8A16, BF16 KV, no thinking, BGE/MiniLM)'
    UPSTREAM=http://127.0.0.1:18137 ;;
  qwen36-35b-gsq-fp8kv)
    MODEL=qwen3.6-35b-gsq-test
    RESULT_ID=qwen3.6-35b-gsq-no-thinking-mixed-w2a16-w8a16-fp8kv-bge-minilm
    LABEL='Qwen3.6 35B GSQ (mixed W2A16/W8A16, FP8 KV, no thinking, BGE/MiniLM)'
    UPSTREAM=http://127.0.0.1:18137 ;;
  mellum21-12b)
    MODEL=mellum2.1-12b-test
    RESULT_ID=mellum2.1-12b-thinking-int4-bge-minilm
    LABEL='Mellum 2.1 12B (local INT4, thinking, BGE/MiniLM)'
    EXTRA_BODY_PROFILE=thinking
    THINKING_MODE=on
    TRACE_REASONING_ARGS=(--require-separated-reasoning)
    UPSTREAM=http://192.168.39.244:8121 ;;
  mellum21-12b-no-thinking)
    MODEL=mellum2.1-12b-test
    RESULT_ID=mellum2.1-12b-no-thinking-int4-bge-minilm
    LABEL='Mellum 2.1 12B (local INT4, no thinking, BGE/MiniLM)'
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
if [[ -n ${QUALITY_RETAIN_UPSTREAM_URL:-} ]]; then
  "$RUNNER_DIR/.venv/bin/python" - "$QUALITY_RETAIN_UPSTREAM_URL" <<'PY' || exit 2
import sys
from urllib.parse import urlsplit
url = urlsplit(sys.argv[1])
assert url.scheme in ('http', 'https') and url.netloc, 'Use an HTTP(S) upstream URL'
assert not url.username and not url.password, 'Do not embed credentials in the upstream URL'
assert url.path in ('', '/') and not url.query and not url.fragment, 'Supply the upstream root without /v1 or query parameters'
PY
  UPSTREAM=${QUALITY_RETAIN_UPSTREAM_URL%/}
fi
LINEAR_ACTIVATIONS=${QUALITY_LINEAR_ACTIVATIONS:-}
KV_DTYPE=${QUALITY_KV_DTYPE:-}
if [[ $PROFILE == qwen36-35b-gsq-* && ( -n $LINEAR_ACTIVATIONS || -n $KV_DTYPE ) ]]; then
  echo 'GSQ has native BF16 activations; choose its explicit KV profile instead of Mellum matrix variables' >&2
  exit 2
fi
if [[ -n $LINEAR_ACTIVATIONS || -n $KV_DTYPE ]]; then
  case "$LINEAR_ACTIVATIONS" in bfloat16) LINEAR_ID=bf16linear; LINEAR_LABEL='BF16 linear activations' ;; int8) LINEAR_ID=int8linear; LINEAR_LABEL='eligible INT8 linear activations' ;; *) echo 'Set QUALITY_LINEAR_ACTIVATIONS=bfloat16|int8 for a matrix arm' >&2; exit 2 ;; esac
  case "$KV_DTYPE" in bfloat16) KV_ID=bf16kv; KV_LABEL='BF16 KV' ;; fp8_e4m3) KV_ID=fp8kv; KV_LABEL='FP8 KV' ;; *) echo 'Set QUALITY_KV_DTYPE=bfloat16|fp8_e4m3 for a matrix arm' >&2; exit 2 ;; esac
  RESULT_ID=${RESULT_ID/-bge-minilm/-$LINEAR_ID-$KV_ID-bge-minilm}
  LABEL=${LABEL/BGE\/MiniLM/$LINEAR_LABEL, $KV_LABEL, BGE\/MiniLM}
fi
if [[ -n ${QUALITY_RESULT_ID_SUFFIX:-} ]]; then
  [[ $QUALITY_RESULT_ID_SUFFIX =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'Result suffix must contain only letters, numbers, dots, underscores or hyphens' >&2; exit 2; }
  RESULT_ID=$RESULT_ID-$QUALITY_RESULT_ID_SUFFIX
fi
if [[ $QUALITY_JUDGE_MODEL != gpt-5.6-luna ]]; then
  [[ $QUALITY_JUDGE_MODEL =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'QA model alias must contain only letters, numbers, dots, underscores or hyphens' >&2; exit 2; }
  RESULT_ID=$RESULT_ID-qa-$QUALITY_JUDGE_MODEL
  LABEL="$LABEL [answers and judge: $QUALITY_JUDGE_MODEL]"
fi
if [[ ( $PROFILE == mellum* || $PROFILE == qwen36-35b-* ) && $MODE == auto ]]; then
  echo 'This profile requires smoke plus the manual source-review handoff before full' >&2
  exit 2
fi
: "${CODEX_LB_TOKEN:?CODEX_LB_TOKEN is required}"
: "${QUALITY_MODEL_PROVENANCE_FILE:?Set a sanitized selected-deployment JSON after performance/sanity validation}"
export PYTHONUNBUFFERED=1
export QUALITY_RETAIN_API_KEY=${QUALITY_RETAIN_API_KEY:-EMPTY}
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
  if [[ $PROFILE == mellum* || $PROFILE == qwen36-35b-* ]]; then
    "$PYTHON" - "$RUNNER_DIR" "$RUN_DIR" "$PROFILE" <<'PY' || exit 2
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from handoff_gpu1_quality import check_review
review = check_review(Path(sys.argv[2]))
if sys.argv[3].startswith('qwen36-35b-'):
    report = json.loads(Path(review['report_path']).read_text())
    assert report.get('semantic_result') == 'pass', 'Qwen3.6 full requires a passed source-faithfulness smoke review'
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
if [[ $PROFILE == qwen36-35b-gsq-* ]]; then
  "$PYTHON" - "$RUN_DIR/deployment.json" "$PROFILE" <<'PY' || exit 2
import json, sys
deployment = json.load(open(sys.argv[1]))
expected_kv = 'bfloat16' if sys.argv[2].endswith('-bf16kv') else 'fp8_e4m3'
assert deployment.get('served_model_name') == 'qwen3.6-35b-gsq-test', 'GSQ served model differs from selected deployment'
assert deployment.get('kv_dtype') == expected_kv, 'GSQ KV profile differs from selected deployment'
assert deployment.get('linear_activation_dtype') == 'bfloat16', 'GSQ requires native BF16 activations'
assert deployment.get('runtime_kv_tokens', 0) > 0 and deployment.get('server_max_num_seqs') == 4, 'Record selected GSQ four-slot capacity before smoke'
PY
fi
if [[ -n $LINEAR_ACTIVATIONS ]]; then
  "$PYTHON" - "$RUN_DIR/deployment.json" "$LINEAR_ACTIVATIONS" "$KV_DTYPE" <<'PY' || exit 2
import json, sys
deployment = json.load(open(sys.argv[1]))
assert deployment.get('linear_activation_dtype') == sys.argv[2], 'Linear activation profile differs from observed deployment provenance'
assert deployment.get('kv_dtype') == sys.argv[3], 'KV profile differs from observed deployment provenance'
assert deployment.get('runtime_kv_tokens', 0) > 0, 'Record actual runtime KV capacity before smoke'
PY
fi
"$PYTHON" - "$RUN_DIR" "$MODE" "$RESULT_ID" "$LINEAR_ACTIVATIONS" "$KV_DTYPE" <<'PY' || exit 2
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1]) / 'quality-profile.json'
profile = dict(result_id=sys.argv[3], linear_activation_dtype=sys.argv[4] or None, kv_dtype=sys.argv[5] or None)
if os.environ.get('QUALITY_JUDGE_MODEL', 'gpt-6-sol') != 'gpt-5.6-luna':
    profile['answer_generator_model'] = os.environ.get('QUALITY_JUDGE_MODEL', 'gpt-6-sol')
    profile['judge_model'] = profile['answer_generator_model']
if os.environ.get('QUALITY_RETAIN_UPSTREAM_URL'):
    profile['upstream_root'] = os.environ['QUALITY_RETAIN_UPSTREAM_URL'].rstrip('/')
if sys.argv[2] == 'full':
    if path.exists():
        assert json.loads(path.read_text()) == profile, 'Quality matrix arm differs from smoke'
    else:
        assert not sys.argv[4] and not sys.argv[5], 'Matrix smoke profile is missing'
else:
    path.write_text(json.dumps(profile, indent=2) + '\n')
PY
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
    --retain-base-url "http://host.docker.internal:$capture_port/v1" --retain-wire-trace "$trace" \
    --question-trace-path "$RUN_DIR/$phase-question-trace.private.json" "$@" >>"$LOG_FILE" 2>&1
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
  SMOKE_ARGS=(--max-conversations 1 --max-questions "${QUALITY_SMOKE_MAX_QUESTIONS:-2}" --no-save)
  if [[ -n ${QUALITY_SMOKE_QUESTION_ABILITIES:-} ]]; then
    SMOKE_ARGS+=(--question-abilities "$QUALITY_SMOKE_QUESTION_ABILITIES")
  fi
  run_phase smoke "${SMOKE_ARGS[@]}" || exit $?
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
