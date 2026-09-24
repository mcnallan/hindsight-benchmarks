#!/usr/bin/env bash
set -u

RUN_DIR=${1:?usage: monitor_qwen_quality.sh RUN_DIR}
echo "Qwen quality run at $(date --iso-8601=seconds)"
echo
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
else
  echo 'Waiting for status.tsv'
fi

echo
echo '===== runner (latest output) ====='
if [[ -f "$RUN_DIR/qwen.log" ]]; then
  tail -n 18 "$RUN_DIR/qwen.log"
else
  echo 'Waiting for qwen.log'
fi

container=$(docker ps --format '{{.Names}}' | grep -E '^quality-qwen3-5-4b-.*-hindsight-1$' | head -1)
if [[ -n $container ]]; then
  echo
  echo '===== live Hindsight activity ====='
  docker logs --tail 160 "$container" 2>&1 \
    | grep -E 'WORKER_TASK|slow llm call|ERROR|Traceback' \
    | tail -n 8
fi
