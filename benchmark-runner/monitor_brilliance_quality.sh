#!/usr/bin/env bash
set -u
RUN_DIR=${1:?usage: monitor_brilliance_quality.sh RUN_DIR}
echo "Brilliance quality run at $(date --iso-8601=seconds)"
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
fi
echo
if [[ -f "$RUN_DIR/brilliance.log" ]]; then
  tail -n 18 "$RUN_DIR/brilliance.log"
fi
container=$(docker ps --format '{{.Names}}' | grep -E '^quality-lfm25-brilliance-.*-hindsight-1$' | head -1)
if [[ -n $container ]]; then
  echo
  echo 'Live Hindsight activity:'
  docker logs --tail 160 "$container" 2>&1 \
    | grep -E 'WORKER_TASK|slow llm call|ERROR|Traceback' | tail -n 8
fi
