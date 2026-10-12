# Local retain-quality comparison

`run_local_quality.py` runs the repository's current frozen BEAM quality
benchmark with production-like OpenAI embeddings and Cohere-compatible
reranking. It launches an isolated Hindsight 0.9.2 container and an ephemeral
PostgreSQL database for every invocation. The database is removed when the run
ends; full benchmark result JSON remains under `../results/leaderboard/llm/`.

The answer generator and judge default to `gpt-6-sol` through codex-lb, at
medium reasoning effort. The benchmark's dataset, prompts, facts-only context,
recall parameters, timestamps, token counter, rubrics, and scoring are not
changed.

Set credentials only in the process environment; do not add them to this repo:

```bash
export QUALITY_RETRIEVAL_API_KEY=...
# CODEX_LB_TOKEN must already be set.
# Set QUALITY_RETAIN_API_KEY only when the retain endpoint uses a different key.
```

Run the API and benchmark call-shape preflights:

```bash
uv run python run_local_quality.py \
  --label "Nemotron 3.5 Lightning" \
  --retain-model nemotron-lightning \
  --preflight-only
```

Run the required smoke test:

```bash
uv run python run_local_quality.py \
  --label "Nemotron 3.5 Lightning" \
  --retain-model nemotron-lightning \
  --max-conversations 1 \
  --max-questions 2 \
  --no-save
```

Run all 4 conversations and 80 questions:

```bash
uv run python run_local_quality.py \
  --label "Nemotron 3.5 Lightning" \
  --retain-model nemotron-lightning
```

For a later model, change only `--label`, `--retain-model`, and, if needed,
`--retain-base-url` / `--retain-preflight-url`. Pass
`--extra-body-profile none` for a model that does not use Nemotron's `nvext`
options. Keep every non-retain-model setting unchanged for a fair comparison.

For LFM2.5 Brilliance with logic thinking and the newer BGE/MiniLM retrieval
services, run the smoke-gated launcher from the repository root:

```bash
bash benchmark-runner/run_brilliance_quality.sh "$PWD/results/runs/brilliance-$(date +%Y%m%d-%H%M%S)"
```

The launcher uses `chat_template_kwargs={"reasoning_effort":"logic","enable_thinking":true}`,
temperature 0.1, strict facts JSON schema, eight retain requests, and a 16,384-token
total retain completion cap. It selects 384-dimensional BGE-small embeddings
with the BGE query instruction and MiniLM-L6 reranking through local LiteLLM.
`QUALITY_RETRIEVAL_API_KEY` can be supplied explicitly; otherwise it reads the
existing credential from the local production Hindsight `.env` without logging it.
`CODEX_LB_TOKEN` stays in the process environment. `QUALITY_JUDGE_MODEL` defaults
to `gpt-6-sol` for both roles. Preserve explicit historical model overrides;
verify the actual selected route before launching a new run.

Every extraction request and response is captured under ignored `results/traces/`
without authorization headers. After the two-question smoke, the launcher checks
the actual logic-thinking controls, reported reasoning tokens, valid facts JSON,
and completion finish reasons before starting the full four-conversation run.
It saves a separate `local-lfm25-brilliance-logic-bge-minilm.json` leaderboard result.
Scores using this retrieval profile have different test conditions from older Jina runs.

Monitor a launched run using its printed run directory:

```bash
watch -n 5 'bash benchmark-runner/monitor_brilliance_quality.sh /absolute/path/to/run-directory'
```

For the standalone GPU1-host endpoints, use
`run_gpu1_quality.sh PROFILE RUN_DIR [smoke|full|auto]`.

`qwen35-4b-redhat-w8a8` and `qwen35-4b-kiln-w8a8` target the independent
loopback SSH-forwarded ports 18142/18143. Both select native INT8 linear kernels,
FP8 E4M3 KV with FlashInfer, FP32 recurrent state, thinking OFF, text-only loading
and eight scheduler slots. Supply selected-deployment provenance with matching
`served_model_name`, `linear_activation_dtype: int8`, `kv_dtype: fp8_e4m3`,
positive `runtime_kv_tokens` and `server_max_num_seqs: 8`. Their dashboard IDs
name the quant, FP8 KV, retrieval and QA model separately from production W4.
Use explicit `smoke`, inspect the real source/fact and question traces, and
record the bounded review using `handoff_gpu1_quality.py record-review` before
explicit `full`; `auto` is refused. Each model uses separate capture ports,
temporary Hindsight/database stacks, run directories and result IDs. Keep the
SSH forwards alive through both phases. A prepared dashboard row or running
smoke does not imply a completed quality score.

`qwen35-4b-production-w4a8` reruns the current production Qwen through Dynamo
using the same strict schema, retrieval and GPT-6 Sol QA pipeline. Set
`QUALITY_DYNAMO_WORKER_ID` to the current decimal GPU2 generation instance ID
from Dynamo `/health`, and record that integer as `dynamo_worker_id` in selected
deployment provenance. The capture proxy adds both supported Dynamo placement
controls and records the effective body. Smoke/full bind the same worker pin
in `quality-profile.json`; obtain a new pin and fresh run if the worker restarts.
Normal production routing is unchanged. This keeps a GPU2 baseline pinned even
when another production worker rejoins. A production run may overlap ordinary
traffic and retains its CPU KV cache, so its retain time is not an isolated
performance measurement.

For these comparisons, prefer the existing six-question ability selection
below to supplement the default two abstention questions. Keep the complete
full evaluation unfiltered. A technical smoke pass establishes serving and
schema reliability for that run; manual source/recall findings remain quality
limitations to report. A repeated token-cap whitespace failure remains visible
even when an earlier smoke passed.

For an explicitly requested diagnostic rerun of this production baseline,
`QUALITY_DIAGNOSTIC_FULL=1` permits `full` after a completed smoke recovered an
extraction failure but failed its raw trace gate. This option is limited to
`qwen35-4b-production-w4a8`; candidate profiles still require a clean technical
smoke. It requires completed question evidence, stored/recalled facts, no QA
errors and the same manual source review. The original failed status stays
unchanged; `diagnostic-admission.json` and a separate admission status explain
the exception. Full request/response validation remains strict, so recovered
full-run extraction failures can still produce a failed trace status alongside
a saved scored result. Do not describe that result as a clean reliability pass.
Dataset, prompts, retrieval and scoring are unchanged; there is no automatic
restart or retry of the run.

`monitor_gpu1_quality.sh` reports completed question scoring separately from
the extraction trace gate. A full invocation can save all 80 scored questions
and then finish with `trace_failed:1` because the gate inspects every captured
LLM attempt, including failed attempts recovered by Hindsight retries. It checks
model/thinking/cap controls, transport and request/response completeness, clean
finish reasons, schema-valid facts JSON, reasoning and whitespace loops. It
does not evaluate whether the extracted facts are faithful to their sources;
that still requires manual source review and the unchanged benchmark scoring.
Keep the failed trace status alongside the completed score instead of treating
a retry as a clean reliability pass or restarting a completed benchmark.

When the same reviewed model recipe is staged on a different endpoint, set
`QUALITY_RETAIN_UPSTREAM_URL` to its HTTP(S) root URL without `/v1` or embedded
credentials. The override applies to preflight, capture forwarding and Hindsight
upstream metadata together. Keep the selected deployment provenance aligned with
that endpoint. `QUALITY_RESULT_ID_SUFFIX` can add a distinct run label to the
model result ID; supply the same suffix and endpoint for a later explicit `full`
phase. These overrides do not change dataset, prompt, retrieval or scoring settings.

`qwen36-35b-twu-low-latency` selects Twu LowLatency as `qwen3.6-35b-test`, with
explicit thinking OFF and a distinct dashboard result ID. Set the upstream
override when reaching its port 8136 endpoint through a local tunnel. Keep the
validated server at four slots and batch size 4,096; client concurrency remains
eight, so queued requests are expected. Run an explicit `smoke` first. The later
explicit `full` phase requires both the strict trace gate and a deployment/trace-bound
manual source review whose JSON report records `semantic_result: "pass"`.
A failed review leaves full unstarted. Dataset, prompts, recall and scoring stay
unchanged.

First complete deployment quantization, performance, and sanity validation.
Record the selected model revision/weights, physical GPU, scheduler capacity,
quantization, KV precision, attention backend, CPU cache/offload, and speculative
decoding settings in a sanitized JSON object; set `QUALITY_MODEL_PROVENANCE_FILE`
to that file. The runner copies it into the run directory and result provenance.
Install wrapper dependencies using `cd benchmark-runner && uv sync` when needed.

All profiles use retain temperature 0.1, strict facts schema, concurrency eight,
and a 16,384 retain completion cap, with BGE/MiniLM retrieval and verified GPT-6 Sol
answering/judging. `qwen35-9b` targets `qwen3.5-9b-test` on port 8129 with
`enable_thinking=false`. `mellum21-12b` targets `mellum2.1-12b-test` on port 8121
on physical GPU0 and explicitly sends `enable_thinking=true` on every retain
attempt. `mellum2-instruct` targets `mellum2-instruct-test` on port 8122 on physical
GPU1, using its native Instruct template without a thinking override. Do not
apply Qwen thinking sampling defaults to either Mellum profile.
The `native-instruct` runner profile also removes inherited reasoning-effort
settings; the native template has no thinking control or reasoning parser.

`mellum21-12b-no-thinking` uses the same Thinking checkpoint on port 8121 but
explicitly sends `enable_thinking=false`; its result ID is separate from thinking
ON. For a matched activation/cache comparison, explicitly set both
`QUALITY_LINEAR_ACTIVATIONS=bfloat16|int8` and
`QUALITY_KV_DTYPE=bfloat16|fp8_e4m3`. Use `mellum21-12b-no-thinking` for the
Thinking checkpoint with thinking OFF, `mellum2-instruct` for native Instruct,
and `mellum21-12b` only for a separate thinking ON control. The selected arm is
saved in `quality-profile.json`; a later full phase must select the same arm.

| Linear activations | KV dtype | Result ID suffix before `-bge-minilm` |
| --- | --- | --- |
| BF16 | BF16 | `-bf16linear-bf16kv` |
| BF16 | FP8 E4M3 | `-bf16linear-fp8kv` |
| Eligible INT8 | BF16 | `-int8linear-bf16kv` |
| Eligible INT8 | FP8 E4M3 | `-int8linear-fp8kv` |

Matrix arms require sanitized metadata with matching observed
`linear_activation_dtype`, `kv_dtype`, and positive actual `runtime_kv_tokens`.
The INT8 designation covers eligible linear activation kernels; record the
actual dense/MoE coverage rather than assuming every activation is quantized.
Keep weights, scheduler geometry, CPU cache/offload, speculation, prompt/schema,
concurrency, completion cap and retrieval/scoring settings fixed across arms.
The runtime owner must measure capacity and release each exact deployment first.
Historical runs retain their original IDs and deployment provenance.

Start with the BF16-linear/BF16-KV arm on both models. If it catastrophically
fails strict extraction, preserve its actual failing request IDs and replay a
bounded frozen selection across all four arms with `probe_retain_capacity.py`
using the same `--trace` and `--request-id` (or identical hashed fixtures).
Include the failing substantive source, a dense dated plan/ownership source, and
a clean control when available. Use `--thinking off` for Mellum 2.1 or
`--thinking native` for Instruct; retain its unchanged strict schema and 16,384
cap. Keep each arm's provenance beside its probe output. This compares actual
retain behavior and eight-way admission; it does not produce a BEAM score or
replace a clean full representative smoke. If baseline passes, continue the
matched one-conversation/two-question smokes for the other arms before deciding
whether to launch any full evaluation.

For a requested smoke-and-decide comparison, launch only the explicit `smoke`
phase in a fresh tmux session with `CODEX_LB_TOKEN` and
`QUALITY_MODEL_PROVENANCE_FILE` explicitly supplied as above, plus both matrix
environment variables when selecting an arm. Do not arm
`handoff_gpu1_quality.py wait` or start `full` until the user decides after reviewing
the smoke. This runs one conversation and two questions with `--no-save`.

For a smoke selected by existing BEAM abilities, set
`QUALITY_SMOKE_QUESTION_ABILITIES=information_extraction,knowledge_update,multi_session_reasoning`
and `QUALITY_SMOKE_MAX_QUESTIONS=6`. The runner accepts the equivalent
`--question-abilities` comma-separated list, filters in original dataset order,
then applies `--max-questions`. Unsupported abilities fail before model calls.
The wrapper forwards these settings only to smoke; full still evaluates all
four conversations and 80 original questions. Neither selection changes the
questions, gold answers, rubrics, prompts, recall or scoring.

Every completed question saves its full question, gold/rubric, recalled facts
context, generated answer and judge reasoning to a private mode-0600 JSON trace,
including with `--no-save`. That flag suppresses leaderboard saving. The wrapper
stores `smoke-question-trace.private.json` or `full-question-trace.private.json`
in the run directory; direct calls can choose `--question-trace-path`, otherwise
the trace goes under `results/traces/quality/`. Interrupted runs retain completed
question evidence with `complete: false`. These traces contain private source
material and should be reviewed locally.

GSQ Qwen3.6 uses the separate `qwen36-35b-gsq-bf16kv` and
`qwen36-35b-gsq-fp8kv` profiles. Their result IDs explicitly name mixed
W2A16/W8A16 weights and the selected KV type. They use native BF16 activations,
FP32 recurrent state and thinking off, rather than Mellum activation matrix
overrides. Match the owner-selected deployment's actual KV dtype and four-slot
capacity before running quality. Prepared dashboard rows do not imply completed
evaluation or admission. Both profiles reject automatic full dispatch; explicit
full requires the current smoke's passed technical trace and a deployment-bound
source review whose `semantic_result` is `pass`. Use the same six-ability-filtered
smoke settings above, then leave the full 80-question run unfiltered. Preserve
previous arithmetic/reference diagnostics separately from BEAM results.

`prepare_source_review.py` prepares a read-only review bundle from captured
smokes. It reuses the handoff helper's trace path and source-message hashing,
joins strict request/response records by request ID, and matches exact source
messages from an earlier reviewed run. Pass `--reference-run`, `--run-dir` and
`--output-dir`; optional `--extra-fixture LABEL=PATH` adds an unchanged frozen
request source such as a dense dated plan. The reference report's four source
cases and previously reviewed empty examples become review inputs, rather than
automatic semantic decisions for the new model.

The bundle preserves complete pairs privately, binds their hashes to the trace
prefix and deployment, and supplies a manual report template with
`semantic_result: pending`. Coarse flags for plan/completion wording, speaker
ownership, dates and versions are review prompts; mixed sources and legitimate
date conversions can trigger them. It never assigns a fidelity score or passes
a gate. Use `--verify-only --run-dir RUN --output-dir BUNDLE` to check the
evidence before reviewing. After writing the actual manual report, bind its
reviewed IDs with the existing `handoff_gpu1_quality.py record-review` command.
Do not use the `wait` subcommand as part of this read-only review workflow.

Future local runs default to `QUALITY_JUDGE_MODEL=gpt-6-sol`, medium effort.
The existing variable controls **both** answer generation and judging. An
explicit `QUALITY_JUDGE_MODEL=gpt-5.6-luna` preserves the historical combined
Luna route and result ID. Other QA aliases receive a `-qa-MODEL` result suffix;
the result records `answer_generator_model` and `judge_model`, and smoke/full
must use the same QA model. Keep historical runs' original model provenance.
Changing a future default does not change an already running process.

Before a detached launch, freeze `run_gpu1_quality.sh` with
`snapshot_quality_wrapper.snapshot_wrapper(RUNNER_DIR, RUN_DIR, PHASE)` and
invoke the returned snapshot path with Bash. Pass its returned
`QUALITY_RUNNER_DIR` environment so relative tools still resolve to the owning
component. Keep the snapshot and source hash in the launch receipt. Editing the
repository wrapper during an active Bash invocation can shift its read position
and cause a later parser failure; immutable snapshots prevent that corruption.
The prepared Twu and GSQ private launchers use this mechanism.

A replay of captured recall contexts changes the answer/judge pipeline without
re-running retain or recall. Label such a replay separately from a fresh
end-to-end run, preserve question dates and exact captured contexts, and do not
overwrite historical scores. Judges receive the question, gold/rubric and
generated answer; they do not receive raw source or recalled context. Upgrading
the judge therefore cannot independently establish whether a detail was invented
or whether a gold answer accurately reflects the source.

Obtain an explicit endpoint release from its performance owner before model
advertisement, capacity probes or smoke calls. Each model can start independently
once released; concurrent evaluations use separate run directories, result IDs,
banks, temporary stacks and capture ports. Client concurrency alone does not
prove eight server slots; verify effective configuration and actual admission.

Use a fresh run directory for each candidate. An explicit smoke phase allows
source faithfulness review before starting the background full phase:

```bash
export QUALITY_MODEL_PROVENANCE_FILE=/absolute/path/to/sanitized-deployment.json
bash benchmark-runner/run_gpu1_quality.sh qwen35-9b "$PWD/results/runs/qwen35-9b-RUN_ID" smoke
# Inspect smoke_trace_summary.json and the captured request/source facts.
# Confirm facts are faithful, recall returned facts, and generation/judging passed.
tmux new-session -d -s beam-qwen35-9b-RUN_ID \
  -e CODEX_LB_TOKEN="$CODEX_LB_TOKEN" \
  -e QUALITY_MODEL_PROVENANCE_FILE="$QUALITY_MODEL_PROVENANCE_FILE" \
  "bash benchmark-runner/run_gpu1_quality.sh qwen35-9b '$PWD/results/runs/qwen35-9b-RUN_ID' full"
watch -n 5 'bash benchmark-runner/monitor_gpu1_quality.sh /absolute/path/to/run-directory'
```

`full` requires the saved smoke gate and identical deployment metadata; it uses
the same benchmark settings without smoke limits. `auto` runs smoke and full
in sequence with the automated wire gate. Replace the profile and session/run
names for Mellum. Keep `CODEX_LB_TOKEN` explicitly set in the launch environment;
never log it. The explicit tmux environment avoids using a stale tmux-server token;
also pass any overridden retrieval key or verified judge fallback to that session.
Future runs use GPT-6 Sol for both roles; historical runs keep their recorded
Luna selection. The launcher uses `QUALITY_RETRIEVAL_API_KEY`, or reads the
local production Hindsight credential privately without logging it.

The Mellum 2.1 gate also requires observed separated reasoning somewhere in the
trace, inspecting both `reasoning` and `reasoning_content`. It records each field,
reasoning characters and reported tokens without printing reasoning text. Strict
JSON must remain in the final content. The native Instruct gate requires no
thinking override and rejects reasoning output. Absent reasoning-token usage
alone does not prove absence of reasoning: both response fields are inspected.

The trace gate streams the NDJSON file, retaining pending schema metadata without
source messages. It reports total errors, up to 20 error samples, and separate
transport/missing-response counts; a later successful retry does not erase an
earlier failed attempt. It validates every extraction response against the actual requested
JSON schema, requires nonempty facts across the trace and `finish_reason=stop`, verifies
the selected thinking control, and rejects reasoning in nonreasoning profiles or
long whitespace runs. It leaves
bounded fact samples for source review; schema validation alone does not prove
semantic faithfulness. The summary also records bounded request IDs and JSON
paths for possible inline thinking tags or reasoning prose inside JSON strings,
without printing those strings. These are manual source-review signals: quoted
source text may legitimately contain such words or tags, and unflagged strings
still require inspection for reasoning, unsupported claims, dates, ownership and
plans versus completed actions. A constrained JSON envelope does not prevent
reasoning from appearing inside an allowed fact string.
Individual schema-valid `facts=[]` responses are counted
and their request IDs sampled for source review. A closing invitation may
correctly yield no durable facts; an empty response to substantive source text
needs completeness review. A trace with no facts overall still fails, and the
runner still requires stored facts and nonempty recall. Separate smoke/full
traces live under `results/traces/`;
runner logs, deployment recipe, status, and trace summaries live in the run
directory. A successful full result must contain all four conversations and 80
questions with no generation/judge errors. Failed runs remain visible in status;
existing runs are never automatically restarted. The distinct leaderboard IDs
are `local-qwen3.5-9b-no-thinking-int4-bge-minilm`,
`local-mellum2.1-12b-thinking-int4-bge-minilm`,
`local-mellum2.1-12b-no-thinking-int4-bge-minilm`, and
`local-mellum2-12b-instruct-int4-bge-minilm`, with separate activation/KV suffixes
for the matrix arms above; these match dashboard manifest
entries and do not overwrite older Qwen or cancelled Brilliance evaluations.

Use separate smoke and full phases when coordinating these evaluations. Record a
bounded review of actual source/fact pairs before the full phase, including dates,
entity attribution, plans versus completed actions, dense-list omissions and any
empty facts. Known semantic errors remain quality findings for the full comparison;
they do not impose a perfect-model threshold. Truncation, invalid JSON, whitespace
loops or incorrect thinking controls fail the technical smoke gate and require
investigation before a full launch. Do not automatically restart a failed run.

For a long smoke, `handoff_gpu1_quality.py wait` can run in its own detached
session with the same explicit credential environment. It waits for the existing
smoke to pass and a manual source-review checkpoint, then launches one full
session using the unchanged wrapper and deployment. Failed or interrupted smokes
exit visibly without retry; an existing handoff or full phase is never restarted.
The source report documents findings rather than asserting semantic perfection.
After reviewing actual completed source/fact pairs, record their request IDs:

```bash
benchmark-runner/.venv/bin/python benchmark-runner/handoff_gpu1_quality.py record-review \
  --run-dir /absolute/path/to/run-directory \
  --report /absolute/path/to/manual-source-review.md \
  --request-id CAPTURED_REQUEST_ID --request-id ANOTHER_REVIEWED_REQUEST_ID
benchmark-runner/.venv/bin/python benchmark-runner/handoff_gpu1_quality.py wait \
  --run-dir /absolute/path/to/run-directory --profile mellum21-12b \
  --smoke-session beam-MODEL-smoke-RUN_ID --full-session beam-MODEL-full-RUN_ID
```

The checkpoint binds the actual reviewed source/response capture prefix, report
and deployment hashes. The coordinator pins launcher hashes when armed and
records its state in `handoff-state.json`. Mellum full phases require this
checkpoint even when launched directly. Use `smoke` plus this handoff for Mellum;
`auto` is refused for Mellum because it lacks the required manual checkpoint. A successful
handoff verifies an actual full-phase strict retain request, then exits without
monitoring the remainder of the full benchmark.

After the selected deployment passes its performance/sanity gate and before the
quality smoke, `probe_retain_capacity.py` can verify eight simultaneous strict
extraction calls. Reuse one representative captured request from the same BEAM
retain prompt/schema; request selection reads the capture incrementally. An
explicit OpenAI request fixture is also accepted with `--fixture`. Source prompts
and schema remain unchanged; the probe sets the target model, selected thinking control,
temperature 0.1, and the same 16,384 retain completion cap. Eight workers start
through a barrier. The endpoint must advertise the model and have idle scheduler
gauges before dispatch.

For Mellum 2.1, add `--thinking on --require-separated-reasoning`. For native
Mellum 2 Instruct, add `--thinking native`. The default remains thinking off for
Qwen. The probe does not reset caches.

For example, after the configuration gate, use the dense milestone source in the
reference capture (request line 32) for Qwen:

```bash
benchmark-runner/.venv/bin/python benchmark-runner/probe_retain_capacity.py \
  --trace results/traces/lfm25-brilliance-brilliance-logic-bge-minilm-20261008-135737.ndjson \
  --request-line 32 \
  --base-url http://192.168.39.244:8129/v1 --model qwen3.5-9b-test \
  --output results/runs/qwen35-9b-RUN_ID-capacity --poll-ms 10 --require-nonempty
```

For Mellum, change only the endpoint to port 8121, model to
`mellum2.1-12b-test`, and output directory. `--api-key-env` chooses the environment
variable holding a retain credential; it defaults to `QUALITY_RETAIN_API_KEY`
and uses `EMPTY` when unset. The probe never prints authentication headers or
credentials and never calls a cache-reset endpoint. It repeats the same source,
so existing prefix caching may affect timing; this is an admission/response check,
not a cold-performance benchmark or semantic-quality score. For a selected
substantive source, `--require-nonempty` explicitly requires facts from each of
the eight responses. This optional capacity-probe requirement does not apply
to filler chunks in the general trace validator.

Private artifacts are `wire.ndjson`, `metrics.ndjson`,
`trace_summary.private.json`, and `summary.json` in a new output directory. Metrics
are sampled at a configured 5–20 ms wait interval, plus HTTP request latency,
and filtered by the advertised model label when present. Exit 0 means all eight
responses passed the strict extraction gate and a sample observed exactly eight
running requests. Exit 1 means response validation failed. Exit 2 means all eight
responses were valid but eight running requests were not observed; short outputs
can finish between samples. Keep this distinction visible and combine observed
running/waiting counts with independently verified effective `MAX_NUM_SEQS`.
Metrics are process-wide, so unrelated endpoint traffic would prevent reliable
attribution; an idle baseline is required. A direct capacity pass does not replace
the representative quality smoke or source-faithfulness review.
