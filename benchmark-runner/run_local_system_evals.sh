#!/usr/bin/env bash
# Queue the local Hindsight system evals for one endpoint. See LOCAL_SYSTEM_EVALS.md.
#
#   run_local_system_evals.sh RUN_DIR QUEUE [--select stock,amb] [--operations OPS] [--cases IDS]
#                             [--repeats N] [--after-session NAME] [--judge MODEL:EFFORT]
#
# QUEUE is qwen, inkling (disabled then minimal) or inkling-minimal. --select picks the
# benchmarks, in this order per profile: stock (the pinned upstream system suite, written to
# the leaderboard), custom (wording and counterfactual fixtures), paired (stock, wording and
# counterfactual arms side by side) and amb. The default, stock,amb, is the original queue.
# --operations limits stock or the custom/paired suites to some operations; --cases and
# --repeats apply to custom/paired only. --after-session waits for that tmux session to
# exit first, so two queues on one endpoint never overlap. --judge picks the codex-lb grader
# (default gpt-6.2-sol:high); a non-default judge gets its own result IDs and job dirs.
set -euo pipefail
umask 077

usage() {
  sed -n '2,13p' "$0" >&2
  exit 2
}
run_dir=$(realpath "${1:?missing benchmark run directory}")
queue=${2:?missing queue}
shift 2
select=stock,amb operations="" cases="" repeats="" after_session="" judge=gpt-6.2-sol:high
while (($#)); do
  case $1 in
    --select) select=${2:?}; shift 2 ;;
    --operations) operations=${2:?}; shift 2 ;;
    --cases) cases=${2:?}; shift 2 ;;
    --repeats) repeats=${2:?}; shift 2 ;;
    --after-session) after_session=${2:?}; shift 2 ;;
    --judge) judge=${2:?}; shift 2 ;;
    *) usage ;;
  esac
done
IFS=, read -ra selected <<< "$select"
want() { [[ ,$select, == *,$1,* ]]; }
for item in "${selected[@]}"; do
  case $item in stock|custom|paired|amb) ;; *) printf 'Unknown selection %s\n' "$item" >&2; exit 2 ;; esac
done
if want stock && [[ -n $cases$repeats ]]; then
  printf -- '--cases and --repeats apply to custom and paired; queue stock separately\n' >&2
  exit 2
fi
judge_model=${judge%%:*} judge_effort=${judge#*:}
if [[ $judge != *:* || ! $judge_model =~ ^gpt-[0-9.]+-[a-z]+$ ||
      ! $judge_effort =~ ^(none|minimal|low|medium|high|xhigh)$ ]]; then
  printf -- '--judge takes MODEL:EFFORT, such as gpt-6.1-terra:medium\n' >&2
  exit 2
fi
# gpt-6.2-sol:high -> sol62-high, the tag every existing result ID carries.
judge_version=${judge_model#gpt-} judge_version=${judge_version%%-*}
judge_tag=${judge_model##*-}${judge_version//./}-$judge_effort
if want custom && want paired; then
  printf 'paired already includes the custom arms; select one of them\n' >&2
  exit 2
fi

cd "$(dirname "$0")"
runner=$PWD
python=${HS_QUEUE_PYTHON:-$runner/.venv/bin/python}
case "$queue" in
  qwen) profiles=(qwen35-4b-off) ;;
  inkling) profiles=(inkling-2-mini-off inkling-2-mini-minimal) ;;
  inkling-minimal) queue=inkling; profiles=(inkling-2-mini-minimal) ;;
  *) printf 'Unknown queue %s\n' "$queue" >&2; exit 2 ;;
esac
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/$queue.log") 2>&1

if [[ -n $after_session ]]; then
  printf '%s waiting for tmux session %s to exit\n' "$(date -Is)" "$after_session"
  while tmux has-session -t "=$after_session" 2>/dev/null; do sleep 60; done
fi

export PYTHONUNBUFFERED=1
export HS_BENCH_CODEX_LB_URL=${HS_BENCH_CODEX_LB_URL:-http://127.0.0.1:2455/v1}
export HS_BENCH_JUDGE_REASONING_EFFORT=$judge_effort HS_PAIRED_JUDGE_MODEL=$judge_model
: "${CODEX_LB_TOKEN:?missing codex-lb credential}"
export EVAL_RETRIEVAL_KEY=${EVAL_RETRIEVAL_KEY:-$("$python" -c 'from dotenv import dotenv_values; from pathlib import Path; print(dotenv_values(Path.home() / "Projects/agents-mono/platform/hindsight/.env")["HINDSIGHT_API_LLM_API_KEY"])')}
if [[ $queue == inkling && -z ${INKLING_API_KEY:-} ]]; then
  INKLING_API_KEY=$(ssh -o BatchMode=yes -o ConnectTimeout=10 mcnallan@192.168.39.245 'cat ~/inkling/.env.secret' |
    "$python" -c 'from dotenv import dotenv_values; import sys; print(dotenv_values(stream=sys.stdin)["VLLM_API_KEY"])')
fi
export TARGET_API_KEY=${INKLING_API_KEY:-unused}
if want custom || want paired; then
  fixture_set=$("$python" -c 'from hindsight_benchmark.paired import FIXTURE_SET; print(FIXTURE_SET)')
fi

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
  [[ $judge_tag == sol62-high ]] || config=$dir/evals-$judge_tag.json
  "$python" - "$config" "$model" "$base" "$control" "$judge_model" <<'PY'
import json
from pathlib import Path
import sys

out, model, base, control, judge = sys.argv[1:]
config = json.loads(Path("evals.json").read_text())
config["judge_model"] = judge
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

  base_id="local-${profile}-$judge_tag"
  retrieval=(--embedder retrieval:BAAI/bge-small-en-v1.5 --reranker retrieval:cross-encoder/ms-marco-MiniLM-L6-v2)
  common=(--config "$config" --provider local-eval --model "$model" --result-id "$base_id" "${retrieval[@]}")
  if { want stock || want custom || want paired; } && [[ ! -f $dir/smoke.done ]]; then
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
  if want stock; then
    stock_ops=${operations:-retain,reflect,consolidation,refresh_pages}
    for op in ${stock_ops//,/ }; do
      [[ ! -f $dir/$op.done ]] || continue
      printf '\n%s %s %s\n' "$(date -Is)" "$profile" "$op"
      "$python" -m hindsight_benchmark.evals "${common[@]}" --operations "$op" > "$dir/$op.log" 2>&1
      touch "$dir/$op.done"
    done
  fi
  for suite in custom paired; do
    want "$suite" || continue
    suite_ops=${operations:-retain,reflect}
    suite_repeats=${repeats:-2}
    # Each selection gets its own job dir and result ID, so a subset never overwrites the full set.
    selection=$(printf '%s|%s|%s' "$suite_ops" "$cases" "$suite_repeats" | sha1sum | cut -c1-8)
    job=$suite-$fixture_set-$selection
    jobdir=$dir/$job
    # The result ID already names the judge; its state dir must too, or a resume would refuse.
    [[ $judge_tag == sol62-high ]] || jobdir=$dir/$judge_tag-$job
    [[ ! -f $jobdir/done ]] || continue
    mkdir -p "$jobdir"
    printf '\n%s %s %s ops=%s cases=%s repeats=%s\n' "$(date -Is)" "$profile" "$job" "$suite_ops" "${cases:-all}" "$suite_repeats"
    "$python" -m hindsight_benchmark.evals --config "$config" --provider local-eval --model "$model" \
      --result-id "$base_id-$job" "${retrieval[@]}" --suite "$suite" --operations "$suite_ops" \
      ${cases:+--case-ids "$cases"} --repeats "$suite_repeats" --state "$jobdir" >> "$jobdir/run.log" 2>&1
    touch "$jobdir/done"
  done
  if want amb; then
    printf '%s %s AMB memory benchmarks\n' "$(date -Is)" "$profile"
    "$python" "$runner/run_local_amb.py" "$dir" >> "$dir/amb.log" 2>&1
  fi
done
printf '%s %s queue complete\n' "$(date -Is)" "$queue"
