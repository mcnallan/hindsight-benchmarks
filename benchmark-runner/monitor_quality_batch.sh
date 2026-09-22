#!/usr/bin/env bash
set -u

RUN_DIR=${1:?usage: monitor_quality_batch.sh RUN_DIR}

echo "Quality batch status at $(date --iso-8601=seconds)"
echo
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
else
  echo "Waiting for status.tsv"
fi

for name in luna nemotron ling; do
  echo
  echo "===== $name (latest output) ====="
  if [[ -f "$RUN_DIR/$name.log" ]]; then
    tail -n 12 "$RUN_DIR/$name.log"
  else
    echo "Waiting for log"
  fi

  case $name in
    luna) pattern='quality-gpt-5-6-luna-.*-hindsight-1' ;;
    nemotron) pattern='quality-nemotron-lightning-.*-hindsight-1' ;;
    ling) pattern='quality-ling-3-0-tiny-.*-hindsight-1' ;;
  esac
  container=$(docker ps --format '{{.Names}}' | grep -E "$pattern" | head -1)
  if [[ -n $container ]]; then
    echo "--- live Hindsight activity ---"
    docker logs --tail 120 "$container" 2>&1 \
      | grep -E 'WORKER_TASK|slow llm call|ERROR|Traceback' \
      | tail -n 4
  fi
done

echo
echo "===== active temporary Hindsight stacks ====="
docker ps --format '{{.Names}}\t{{.Status}}' | grep '^quality-' || echo none
