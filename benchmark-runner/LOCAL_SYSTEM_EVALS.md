# Local Hindsight system evals

The local queue uses the Hindsight 1.2.0 system suite pinned in
[`evals.json`](evals.json) at `7b46a07c82f997903230f0fb33deafe9dfd2e22b`.
It starts isolated Hindsight servers with fresh embedded PostgreSQL instances
and sends model calls to the existing inference endpoints. The
[pinned system-eval guide](https://github.com/vectorize-io/hindsight/blob/7b46a07c82f997903230f0fb33deafe9dfd2e22b/hindsight-system-evals/README.md)
describes the fixtures and cases. The historical four-conversation BEAM
comparison has its own [runner and protocol](LOCAL_QUALITY.md).

## Model and retrieval profiles

| Profile | Default endpoint and API model | Request controls |
| --- | --- | --- |
| `qwen35-4b-off` | Production Dynamo at `http://10.10.5.1:8000/v1` with `qwen3.5-4b` | `chat_template_kwargs.enable_thinking=false` |
| `inkling-2-mini-off` | GPU2 HyperInkling at `http://192.168.39.245:18089/v1` with `inkling-2-mini` | `reasoning_effort=none` and `logit_bias={"200008":-100}` |
| `inkling-2-mini-minimal` | Same GPU2 endpoint and model | `reasoning_effort=minimal` |

Qwen uses the production Ampere vLLM recipe with 4-bit weights, INT8 eligible
activations, FP8 E4M3 KV and FlashInfer. The Inkling comparison used the
HyperInkling 0.1.4 Ampere deployment with 4-bit experts, Humming W4A8 and
FP8 KV. Check live deployment settings before comparing runs across engine
restarts or model changes. The serving recipes live in the deployment repo
under `platform/inference/dynamo/prod-qwen35-4b/` and
`platform/inference/vllm/inkling-3090/`.

The Inkling server can still reason at `reasoning_effort=none`. Token `200008`
is its thinking-channel marker. Suppressing it produced zero reported reasoning
tokens in the disabled smoke and full system run, while keeping native tool
calls available. Revalidate that token ID and both strict JSON and tool calls
when changing the tokenizer or serving stack. Final-text prefill is specific
to the older retain-only BEAM runner. The system comparisons request up to
16,384 completion tokens for retain, reflect and consolidation.

All profiles use BAAI BGE-small-en-v1.5 embeddings and cross-encoder
MS MARCO MiniLM-L6-v2 reranking through LiteLLM on `127.0.0.1:4000`. Embeddings
use 384 dimensions, the BGE query instruction, and batches of at most 32 to
match the serving limit. GPT-6.2 Sol at high effort grades through codex-lb at
`http://127.0.0.1:2455/v1`. The configured decision model is disabled. Two Reflect
cases and two Refresh pages cases per pass use upstream's scripted decision stub
to test follow-up search orchestration. The public reference boards use a Gemini
judge. The visualizer labels the judge on each run's expanded row.

## Prepare and launch the queues

From the root of this checkout, install the runner and the pinned AMB checkout.
The launcher fetches and prepares pinned Hindsight on first use.

```bash
cd benchmark-runner
uv sync --locked
mkdir -p .hindsight
git init .hindsight/amb
git -C .hindsight/amb fetch --depth=1 https://github.com/vectorize-io/agent-memory-benchmark.git 01bd83f3fa5245d6be3986cfab81fdd828f26acd
git -C .hindsight/amb checkout --detach FETCH_HEAD
uv sync --locked --project .hindsight/amb
cd ..
```

Export `CODEX_LB_TOKEN` in the shell launching the queue. If tmux already runs,
ensure the session inherits the credential. `EVAL_RETRIEVAL_KEY` defaults to
the existing workstation Hindsight credential. `INKLING_API_KEY` defaults to
the GPU2 Inkling credential read over SSH. Use the environment overrides on
other workstations, with values supplied by their configured secret store.
`QWEN_EVAL_BASE_URL`, `INKLING_EVAL_BASE_URL`, and `HS_BENCH_CODEX_LB_URL`
override the model and judge URLs. Keep credentials out of command arguments,
benchmark configs, logs and result files.

Check the model list, actual structured completion and tool-call shapes,
LiteLLM embeddings and reranking, and an actual structured judge completion.
An HTTP 200 model listing alone leaves inference availability unverified.

Use one run directory for the comparison and a detached session per model
endpoint. Check `tmux list-sessions` for an active run on the same profile.
Each profile has a stable leaderboard result ID, so concurrent writers for
the same profile can overwrite each other's results.

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$PWD/results/runs/system-evals-$stamp"
mkdir -p "$run_dir"
printf '%s\n' "$run_dir" > results/runs/current-system-evals
tmux new-session -d -s "hindsight-qwen35-$stamp" -c "$PWD" \
  bash "$PWD/benchmark-runner/run_local_system_evals.sh" "$run_dir" qwen
tmux new-session -d -s "hindsight-inkling-$stamp" -c "$PWD" \
  bash "$PWD/benchmark-runner/run_local_system_evals.sh" "$run_dir" inkling
```

Each profile runs a four-case technical smoke, then Retain (42 cases), Reflect
(17), Consolidation (25), and Refresh pages (20), with two passes per operation.
The smoke checks endpoint health, available telemetry, and the expected
reasoning mode. Its grades diagnose quality. It can proceed after an ordinary
graded failure. A failed technical smoke stops the queue before the full run.

The `inkling` queue runs the entire disabled profile, including AMB, before
starting `inkling-2-mini-minimal`. `inkling-minimal` runs only the minimal
profile. To serialize Qwen and Inkling behind one judge, run both launchers in
one tmux session and join them with `&&`. A failed AMB split records `failed.txt`
and the queue proceeds to the other splits. A system-run execution error stops
the queue. Queue exit and completed benchmark coverage are separate checks.

## Monitor, resume and run selected cases

```bash
watch -n 15 bash benchmark-runner/monitor_local_system_evals.sh
```

The monitor accepts a run directory argument. The default comes from
`results/runs/current-system-evals`. Queue logs are `qwen.log` and `inkling.log`.
Per-profile files include `evals.json`, `smoke.log`, operation logs, `.done`
markers and `amb.log`. JUnit XML, per-pass token totals and per-case traces live
under `benchmark-runner/.hindsight/runs/`. `operations[*].run_dir` in a result
points to the raw evidence for that operation.

Use the same run directory and the same profile settings to resume a stopped
queue. An existing `.done` marker skips the corresponding system stage. An AMB
`summary.json` skips the completed split. An interrupted stage without a marker
normally starts again. The lower-level `--resume` option skips passes that
already wrote JUnit XML. Inspect those passes before reusing them after an
endpoint failure. For codex-lb HTTP 429, verify a real judge completion before
restarting. For a 504 from native Reflect, the associated AMB log identifies
the query and server timeout. An aborted split or partial save is evidence for
diagnosis only.

The generated profile config can also run selected operations or stock cases
directly. The example uses Qwen and a config generated by a previous queue.
Set `CODEX_LB_TOKEN` and `EVAL_RETRIEVAL_KEY` in the process environment first.

```bash
cd benchmark-runner
run_dir=$(cat ../results/runs/current-system-evals)
export TARGET_API_KEY=unused
export HS_BENCH_CODEX_LB_URL=http://127.0.0.1:2455/v1
export HS_BENCH_JUDGE_REASONING_EFFORT=high
uv run --locked hindsight-evals \
  --config "$run_dir/qwen35-4b-off/evals.json" --provider local-eval --model qwen3.5-4b \
  --result-id qwen-retain-diagnostic \
  --embedder retrieval:BAAI/bge-small-en-v1.5 \
  --reranker retrieval:cross-encoder/ms-marco-MiniLM-L6-v2 \
  --operations retain --repeats 1 --cases 'coreference-roles or examples-do-not-leak'
```

`--cases` uses pytest's `-k` selection and keeps the diagnostic out of the
leaderboard JSON. Omit `--cases` for a complete operation and give it a distinct
`--result-id` when changing the procedure. Custom fixture overlays and named
benchmark selection in the queue are planned follow-up work.

## AMB and published results

After the system suites, each profile uses the same pinned Hindsight with
the pinned [Agent Memory Benchmark](https://github.com/vectorize-io/agent-memory-benchmark).
The order is PrecisionMemBench single-turn smoke and full retrieval, LoCoMo
Fixed with native Reflect, standard LoCoMo, BEAM 100K, PersonaMem 32K,
LifeBench EN, and LongMemEval S. The RAG splits use GPT-6.2 Sol high as a fixed
answerer and grader. PrecisionMemBench scores the returned belief set without a
generation judge. AMB starts a fresh server per split and uses four Hindsight
worker slots, four concurrent queries, and one concurrent ingestion unit.
The original 2,003-question LifeBench EN file comes from a pinned revision.
Logs and reports are under each profile's `amb/` directory.

Publish completed system results in `results/leaderboard/evals/` along with
their configuration changes. The operation pages read those files directly.
All operations requires four complete operations for the profile, uses the
mean of their pass rates, and shows cost per 1,000 cases of each operation.
The Inkling rows apply the supplied local rates to measured tokens, including
reasoning in output. Per million tokens, those rates are $0.075 for uncached
input, $0.015 for cache reads and $0.30 for output. The local comparison page
also reads AMB summaries from the active run directory. Credentials, raw
traces, bank data, and raw AMB responses stay in ignored locations.
