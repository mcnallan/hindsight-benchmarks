#!/usr/bin/env bash
set -u

RUN_DIR=${1:?usage: monitor_ornith_quality.sh RUN_DIR}
echo "Ornith quality run at $(date --iso-8601=seconds)"
echo
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
else
  echo 'Waiting for status.tsv'
fi

for name in no-thinking thinking; do
  echo
  echo "===== $name (latest output) ====="
  if [[ -f "$RUN_DIR/$name.log" ]]; then
    tail -n 12 "$RUN_DIR/$name.log"
  else
    echo 'Waiting for log'
  fi
done

container=$(docker ps --format '{{.Names}}' | grep -E '^quality-ornith-1-5-9b-.*-hindsight-1$' | head -1)
if [[ -n $container ]]; then
  echo
  echo '===== live Hindsight activity ====='
  docker logs --tail 160 "$container" 2>&1 \
    | grep -E 'WORKER_TASK|slow llm call|ERROR|Traceback' \
    | tail -n 8
fi
