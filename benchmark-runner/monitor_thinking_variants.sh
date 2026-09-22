#!/usr/bin/env bash
set -u

RUN_DIR=${1:?usage: monitor_thinking_variants.sh RUN_DIR}
echo "Thinking-variant quality batch at $(date --iso-8601=seconds)"
echo
if [[ -f "$RUN_DIR/status.tsv" ]]; then
  column -t -s $'\t' "$RUN_DIR/status.tsv"
else
  echo 'Waiting for status.tsv'
fi

for name in ling-thinking nemotron-no-thinking; do
  echo
  echo "===== $name (latest output) ====="
  if [[ -f "$RUN_DIR/$name.log" ]]; then
    tail -n 14 "$RUN_DIR/$name.log"
  else
    echo 'Waiting for log'
  fi
done

echo
echo '===== active temporary Hindsight stacks ====='
docker ps --format '{{.Names}}\t{{.Status}}' | grep '^quality-' || echo none
