#!/usr/bin/env bash
set -u

RUN_DIR=${1:?usage: monitor_qwen35_variants.sh RUN_DIR}
echo "Qwen3.5 quality batch at $(date --iso-8601=seconds)"
echo
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
else
  echo 'Waiting for status.tsv'
fi

for name in base-thinking distilled-thinking distilled-no-thinking; do
  echo
  echo "===== $name (latest output) ====="
  if [[ -f "$RUN_DIR/$name.log" ]]; then
    tail -n 10 "$RUN_DIR/$name.log"
  else
    echo 'Waiting for log'
  fi
done

echo
echo '===== active temporary Hindsight stacks ====='
docker ps --format '{{.Names}}\t{{.Status}}' \
  | grep -E '^quality-qwen3-5-4b(-distilled)?-' || echo none

for container in $(docker ps --format '{{.Names}}' \
  | grep -E '^quality-qwen3-5-4b(-distilled)?-.*-hindsight-1$'); do
  echo
  echo "===== live Hindsight activity: $container ====="
  docker logs --tail 140 "$container" 2>&1 \
    | grep -E 'WORKER_TASK|slow llm call|JSON parse error|Task execution failed|ERROR' \
    | tail -n 5
done
