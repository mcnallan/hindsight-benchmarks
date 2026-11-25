#!/usr/bin/env bash
set -euo pipefail
umask 077

run_dir=$(realpath "${1:?missing benchmark run directory}")
queue=${2:?missing queue}
cd "$(dirname "$0")"
runner=$PWD
python=$runner/.venv/bin/python
case "$queue" in
  qwen) profiles=(qwen35-4b-off) ;;
  inkling) profiles=(inkling-2-mini-off inkling-2-mini-minimal) ;;
  inkling-minimal) queue=inkling; profiles=(inkling-2-mini-minimal) ;;
  *) printf 'Unknown queue %s\n' "$queue" >&2; exit 2 ;;
esac
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/$queue.log") 2>&1

export PYTHONUNBUFFERED=1
export HS_BENCH_CODEX_LB_URL=${HS_BENCH_CODEX_LB_URL:-http://127.0.0.1:2455/v1}
export HS_BENCH_JUDGE_REASONING_EFFORT=high
: "${CODEX_LB_TOKEN:?missing codex-lb credential}"
export EVAL_RETRIEVAL_KEY=${EVAL_RETRIEVAL_KEY:-$("$python" -c 'from dotenv import dotenv_values; from pathlib import Path; print(dotenv_values(Path.home() / "Projects/agents-mono/platform/hindsight/.env")["HINDSIGHT_API_LLM_API_KEY"])')}
if [[ $queue == inkling && -z ${INKLING_API_KEY:-} ]]; then
  INKLING_API_KEY=$(ssh -o BatchMode=yes -o ConnectTimeout=10 mcnallan@192.168.39.245 'cat ~/inkling/.env.secret' |
    "$python" -c 'from dotenv import dotenv_values; import sys; print(dotenv_values(stream=sys.stdin)["VLLM_API_KEY"])')
fi
export TARGET_API_KEY=${INKLING_API_KEY:-unused}

for profile in "${profiles[@]}"; do
  dir=$run_dir/$profile
  mkdir -p "$dir"
  case "$profile" in
    qwen35-4b-off)
      model=qwen3.5-4b
      base=${QWEN_EVAL_BASE_URL:-http://10.10.5.1:8000/v1}
      control='{"chat_template_kwargs":{"enable_thinking":false}}'
      ;;
    inkling-2-mini-off)
      model=inkling-2-mini
      base=${INKLING_EVAL_BASE_URL:-http://192.168.39.245:18089/v1}
      # The deployed tokenizer maps <|content_thinking|> to 200008. `none`
      # alone still emits that token. Bias keeps the native tool channels open.
      control='{"reasoning_effort":"none","logit_bias":{"200008":-100}}'
      ;;
    inkling-2-mini-minimal)
      model=inkling-2-mini
      base=${INKLING_EVAL_BASE_URL:-http://192.168.39.245:18089/v1}
      control='{"reasoning_effort":"minimal"}'
      ;;
  esac
  config=$dir/evals.json
  "$python" - "$config" "$model" "$base" "$control" <<'PY'
import json
from pathlib import Path
import sys

out, model, base, control = sys.argv[1:]
config = json.loads(Path("evals.json").read_text())
config["judge_model"] = "gpt-6.2-sol"
config["providers"]["local-eval"] = {
    "hindsight_provider": "openai", "base_url": base, "api_key_env": "TARGET_API_KEY",
}
config["providers"]["retrieval"] = {
    "base_url": "http://127.0.0.1:4000/v1", "rerank_url": "http://127.0.0.1:4000/v1/rerank",
    "api_key_env": "EVAL_RETRIEVAL_KEY",
}
config["model_settings"][model] = {
    "LLM_EXTRA_BODY": control,
    "RETAIN_MAX_COMPLETION_TOKENS": "16384",
    "CONSOLIDATION_MAX_COMPLETION_TOKENS": "16384",
    "REFLECT_MAX_COMPLETION_TOKENS": "16384",
    "EMBEDDINGS_OPENAI_DIMENSIONS": "384",
    "EMBEDDINGS_OPENAI_BATCH_SIZE": "32",
    "EMBEDDINGS_QUERY_PREFIX": "Represent this sentence for searching relevant passages: ",
}
Path(out).write_text(json.dumps(config, indent=2) + "\n")
PY

  common=(--config "$config" --provider local-eval --model "$model" --result-id "local-${profile}-sol62-high"
    --embedder retrieval:BAAI/bge-small-en-v1.5 --reranker retrieval:cross-encoder/ms-marco-MiniLM-L6-v2)
  if [[ ! -f $dir/smoke.done ]]; then
    printf '\n%s %s system smoke\n' "$(date -Is)" "$profile"
    if ! "$python" -m hindsight_benchmark.evals "${common[@]}" --operations retain,reflect --repeats 1 \
      --cases 'coreference-roles or examples-do-not-leak or date-borrowed-from-nearby-subject or (test_reflect and hq-owner-latest)' \
      > "$dir/smoke.log" 2>&1; then
      printf '%s failed to execute the smoke. See %s/smoke.log\n' "$profile" "$dir" >&2
      exit 1
    fi
    "$python" - "$dir/smoke.log" "$profile" <<'PY'
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

log, profile = sys.argv[1:]
lines = Path(log).read_text().splitlines()
run = Path(next(line.removeprefix("run and traces in ") for line in reversed(lines)
                if line.startswith("run and traces in ")))
cases = list(ElementTree.parse(run / "junit-0.xml").iter("testcase"))
if len(cases) != 4 or any(c.find("error") is not None or c.find("skipped") is not None for c in cases):
    raise SystemExit(f"Smoke did not complete four cases cleanly. See {log}")
transport_errors = ("ServiceException:", "codex-lb grading failed", "APIConnectionError:")
if any(marker in case_text for case_text in (c.findtext("failure", "") for c in cases)
       for marker in transport_errors):
    raise SystemExit(f"Smoke hit an endpoint error. See {log}")
usage = [json.loads(line) for line in (run / "tokens-0.jsonl").read_text().splitlines()]
calls = sum(row["calls"] for row in usage)
thoughts = sum(row.get("thoughts_tokens", 0) for row in usage)
if calls == 0 or (thoughts > 0) != profile.endswith("minimal"):
    raise SystemExit(f"Smoke token telemetry disagrees with {profile}: {calls} calls, {thoughts} reasoning tokens")
print(f"Smoke telemetry: {calls} calls, {thoughts} reasoning tokens; verdicts in {log}")
PY
    touch "$dir/smoke.done"
  fi
  for op in retain reflect consolidation refresh_pages; do
    [[ ! -f $dir/$op.done ]] || continue
    printf '\n%s %s %s\n' "$(date -Is)" "$profile" "$op"
    "$python" -m hindsight_benchmark.evals "${common[@]}" --operations "$op" > "$dir/$op.log" 2>&1
    touch "$dir/$op.done"
  done
  printf '%s %s AMB memory benchmarks\n' "$(date -Is)" "$profile"
  "$python" "$runner/run_local_amb.py" "$dir" >> "$dir/amb.log" 2>&1
done
printf '%s %s queue complete\n' "$(date -Is)" "$queue"
