#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

repo=$(cd "$(dirname "$0")/.." && pwd)
run_dir=${1:-$(cat "$repo/results/runs/current-system-evals")}
printf '%s\n' "$run_dir"
for queue in qwen inkling; do
  if [[ -f $run_dir/$queue.log ]]; then
    printf '\n%s queue\n' "$queue"
    tail -n 3 "$run_dir/$queue.log"
  fi
done
for profile in "$run_dir"/*/evals.json; do
  dir=$(dirname "$profile")
  name=$(basename "$dir")
  result=$repo/results/leaderboard/evals/local-$name-sol62-high.json
  printf '\n%s\n' "$name"
  if [[ -f $result ]]; then
    jq -r '.operations | to_entries[] | "  \(.key) \(.value.passed)/\(.value.total), \(.value.errors) errors, \(.value.tokens.thoughts_tokens // 0) reasoning tokens"' "$result"
  fi
  for stage in smoke retain reflect consolidation refresh_pages; do
    if [[ -f $dir/$stage.log && ! -f $dir/$stage.done ]]; then
      printf '  %s ' "$stage"
      tail -n 1 "$dir/$stage.log"
    fi
  done
  for summary in "$dir"/amb/*/summary.json; do
    [[ $(basename "$(dirname "$summary")") != smoke ]] || continue
    jq -r '"  AMB \(.dataset)/\(.split) \(.correct)/\(.total_queries), score \((.accuracy * 1000 | round) / 10)%"' "$summary"
  done
  for failure in "$dir"/amb/*/failed.txt; do
    printf '  %s\n' "$(cat "$failure")"
  done
  if [[ -f $dir/amb.log ]]; then
    tail -n 2 "$dir/amb.log"
  fi
  for job in "$dir"/*/job.json; do
    jobdir=$(dirname "$job")
    rundir=$repo/benchmark-runner/$(jq -r .run_dir "$job")
    expected=$(jq -r '.settings | "\(.repeats) passes of \(.operations | join(",")) cases=\(if (.case_ids | length) > 0 then (.case_ids | join(",")) else "all" end)"' "$job")
    rows=$(cat "$rundir"/records-*.jsonl 2>/dev/null | jq -s 'length' 2>/dev/null || echo 0)
    graded=$(cat "$rundir"/records-*.jsonl 2>/dev/null | jq -s '[.[] | select(.passed != null)] | "\(map(select(.passed)) | length)/\(length) passed, " ' -r 2>/dev/null || true)
    infra=$(cat "$rundir"/records-*.jsonl 2>/dev/null | jq -s '[.[] | select(.passed == null)] | length' 2>/dev/null || echo 0)
    state=running; [[ -f $jobdir/done ]] && state=done
    printf '  %s %s: %s executions, %s%s infrastructure (%s)\n' "$(basename "$jobdir")" "$state" "$rows" "$graded" "$infra" "$expected"
    [[ $state == done ]] || tail -n 1 "$jobdir/run.log" 2>/dev/null | cut -c1-200
  done
done
