#!/usr/bin/env bash
set -u
RUN_DIR=${1:?usage: monitor_gpu1_quality.sh RUN_DIR}
echo "Quality run at $(date --iso-8601=seconds)"
if [[ -f "$RUN_DIR/status.tsv" ]]; then column -t -s $'\t' "$RUN_DIR/status.tsv"; fi
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
