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
`--result-id` when changing the procedure.

## Select benchmarks and run paired robustness cases

The queue takes optional selectors after the queue name. Without them it runs
the original queue, so the `qwen`, `inkling` and `inkling-minimal` commands
above are unchanged.

```bash
run_local_system_evals.sh RUN_DIR QUEUE [--select stock,amb] [--operations OPS] \
  [--cases IDS] [--repeats N] [--after-session NAME] [--judge MODEL:EFFORT]
```

`--select` takes a comma list from `stock`, `custom`, `paired` and `amb`, run in
that order per profile. `stock` is the pinned upstream suite and writes the
leaderboard result. `custom` runs only the wording and counterfactual fixtures.
`paired` runs the stock case beside both variants, each in a fresh bank.
`amb` runs the AMB splits on their own. `--operations` limits stock or the
fixture suites. `--cases` (stock case IDs) and `--repeats` (default 2) apply to
`custom` and `paired`. `--after-session` waits for another tmux session to exit,
so two queues on one model endpoint never overlap. Any system selection runs
the technical smoke first unless the profile already has `smoke.done`.

`--judge` picks the codex-lb grader, `gpt-6.2-sol:high` by default. Any other
judge gets its own result IDs (`local-PROFILE-terra61-medium-…` for
`gpt-6.1-terra:medium`), job dirs and config file, so it never resumes or
overwrites a Sol job. The report and visualizer show each judge as its own row;
compare profiles only across rows graded by the same judge. Run the fixture
grader check below with the new judge before switching. On 25 November
`gpt-6.1-terra:medium` graded all 58 probes as expected, as Sol high did. Both
judges hit request timeouts when codex-lb carried several graders at once.

The fixtures live in this fork under
[`src/hindsight_benchmark/paired/`](src/hindsight_benchmark/paired/), keyed to
the pinned commit and the original case ID. The pinned checkout stays
unmodified. Fixture set `7b46a07c-v2` covers 12 cases. (`v2` split one `coreference-roles` counterfactual criterion that `v1` made stricter than stock; `v1` results are kept only as evidence.) Retain has
`coreference-roles`, `multi-speaker-attribution`, `context-names-the-speaker`,
`in-text-correction`, `date-borrowed-from-nearby-subject` and
`owner-borrowed-from-nearby-subject`. Reflect has `hq-owner-latest`,
`hq-absent-2024`, `hq-scoped-2fa`, `sp-rotation`, `sp-upgrade-owner` and
`sp-postmortem`. A wording variant renames entities or rephrases while keeping
the answer. A counterfactual changes one fact, and its traps reject the stock
answer. Each counterfactual also records a `stale_claim`, which is graded beside
the verdict and never scored. The paired modules call the stock runners and
criteria wording, loading the pinned fixtures as a pytest plugin.

Each selection gets its own job directory under the profile,
`<suite>-<fixture set>-<selection hash>`, and its own result ID,
`local-<profile>-sol62-high-<job>`. The first start writes `job.json` with the
run directory, fixture revision (a hash of the fixture sources) and effective
settings: model, endpoint, request controls, judge, retrieval, commit, repeats,
cases and arms. Restarting the same command resumes that run directory, skips
passes that wrote their JUnit file, and refuses to continue if the fixtures or
settings changed. Executions that hit a server, endpoint or judge error are
rerun up to twice in fresh banks after a real judge completion. Anything still
broken is reported as infrastructure, apart from graded outcomes.

Before running models on changed fixtures, grade every fixture probe with the
run's judge. Each probe is a canned correct, stock or wrong output, and the check
expects the variant's criteria to accept the revised answer and reject the old.

```bash
cd benchmark-runner/.hindsight/7b46a07c82f997903230f0fb33deafe9dfd2e22b/hindsight-system-evals
HINDSIGHT_EVAL_JUDGE_MODEL=gpt-6.2-sol HS_BENCH_CODEX_LB_URL=http://127.0.0.1:2455/v1 \
  HS_BENCH_JUDGE_REASONING_EFFORT=high \
  PYTHONPATH=$OLDPWD/../src:$PWD uv run --frozen python -m hindsight_benchmark.paired.check_graders OUT.json
```

A paired job writes no leaderboard row. It writes `results/paired/<result id>.json`
with the fixture revision, settings, per-arm pass rates and one compact row per
execution: verdict, trap, stale answer, failure stage, and gold facts retrieved.
Model output, judge reasoning and traces stay in the run directory's
`records-*.jsonl` and trace folders. Retain failures are extraction failures.
Reflect failures are retrieval when a gold fact never reached the model, and
generation otherwise. Read traces before attributing a case to grading. The
monitor shows each job's progress. To combine jobs:

The visualizer's Paired cases tab (`/paired`) shows every case's stock,
wording and counterfactual text side by side, marked against stock, with the
pilot results read from `results/paired/`. Regenerate its fixture data after
changing a fixture:

```bash
cd benchmark-runner/.hindsight/7b46a07c82f997903230f0fb33deafe9dfd2e22b/hindsight-system-evals
PYTHONPATH=$OLDPWD/../src:$PWD uv run --frozen python -m hindsight_benchmark.paired.export_visualizer \
  ../../../../visualizer/lib/paired-cases.json
```

```bash
benchmark-runner/.venv/bin/python benchmark-runner/report_paired.py \
  results/paired/local-*-paired-7b46a07c-v2-*.json --out results/paired/REPORT.md
```

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
