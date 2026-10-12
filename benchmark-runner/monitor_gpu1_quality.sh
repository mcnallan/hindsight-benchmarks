#!/usr/bin/env bash
set -u
RUN_DIR=${1:?usage: monitor_gpu1_quality.sh RUN_DIR}
RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "Quality run at $(date --iso-8601=seconds)"
if [[ -f "$RUN_DIR/status.tsv" ]]; then column -t -s $'\t' "$RUN_DIR/status.tsv"; fi
if [[ -f "$RUN_DIR/full-question-trace.private.json" ]]; then
  "$RUNNER_DIR/.venv/bin/python" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
try:
    trace = json.loads((run / 'full-question-trace.private.json').read_text())
    result = trace.get('result', {})
    if trace.get('complete') and result:
        print(f"Benchmark scoring complete: {result.get('correct')}/{result.get('total')} ({result.get('accuracy')}%); "
              f"generation errors={result.get('generation_errors')}, judge errors={result.get('judge_errors')}")
    else:
        print(f"Benchmark scoring in progress: {len(trace.get('questions', []))} questions captured")
    summary_path = run / 'full_trace_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        errors = summary.get('error_count', len(summary.get('errors', [])))
        print(f"Extraction trace check: {'FAILED' if errors else 'passed'}; "
              f"{errors} violations across {summary.get('strict_facts_requests')} captured attempts. "
              "This checks every attempt, including retries.")
    else:
        print('Extraction trace check: pending')
except (OSError, ValueError) as error:
    print(f'Benchmark summary unavailable: {type(error).__name__}')
PY
fi
if [[ -f "$RUN_DIR/quality.log" ]]; then echo; tail -n 18 "$RUN_DIR/quality.log"; fi
if [[ -f "$RUN_DIR/retain-model.txt" ]]; then
  model=$(cat "$RUN_DIR/retain-model.txt")
  prefix=${model//./-}
  prefix=${prefix:0:24}
  while IFS= read -r container; do
    [[ $container == "quality-$prefix-"*"-hindsight-1" ]] || continue
    echo
    echo "Live Hindsight activity: $container"
    docker logs --tail 160 "$container" 2>&1 | rg 'WORKER_TASK|slow llm call|ERROR|Traceback' | tail -n 8
  done < <(docker ps --format '{{.Names}}')
fi
