# Local retain-quality comparison

`run_local_quality.py` runs the repository's current frozen BEAM quality
benchmark with production-like OpenAI embeddings and Cohere-compatible
reranking. It launches an isolated Hindsight 0.9.2 container and an ephemeral
PostgreSQL database for every invocation. The database is removed when the run
ends; full benchmark result JSON remains under `../results/leaderboard/llm/`.

The answer generator and judge are both `gpt-5.6-luna` through codex-lb, at
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
to `gpt-5.6-luna`; set it to `gpt-6-luna` only after verifying the fallback endpoint.

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
`run_gpu1_quality.sh qwen35-9b|mellum21-12b|mellum2-instruct RUN_DIR [smoke|full|auto]`.
First complete deployment quantization, performance, and sanity validation.
Record the selected model revision/weights, physical GPU, scheduler capacity,
quantization, KV precision, attention backend, CPU cache/offload, and speculative
decoding settings in a sanitized JSON object; set `QUALITY_MODEL_PROVENANCE_FILE`
to that file. The runner copies it into the run directory and result provenance.
Install wrapper dependencies using `cd benchmark-runner && uv sync` when needed.

All profiles use retain temperature 0.1, strict facts schema, concurrency eight,
and a 16,384 retain completion cap, with BGE/MiniLM retrieval and verified Luna
answering/judging. `qwen35-9b` targets `qwen3.5-9b-test` on port 8129 with
`enable_thinking=false`. `mellum21-12b` targets `mellum2.1-12b-test` on port 8121
on physical GPU0 and explicitly sends `enable_thinking=true` on every retain
attempt. `mellum2-instruct` targets `mellum2-instruct-test` on port 8122 on physical
GPU1, using its native Instruct template without a thinking override. Do not
apply Qwen thinking sampling defaults to either Mellum profile.
The `native-instruct` runner profile also removes inherited reasoning-effort
settings; the native template has no thinking control or reasoning parser.

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
Keep GPT-5.6 Luna unless its verification fails and GPT-6 Luna is
separately verified. The launcher uses `QUALITY_RETRIEVAL_API_KEY`, or reads the
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
semantic faithfulness. Individual schema-valid `facts=[]` responses are counted
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
`local-mellum2.1-12b-thinking-int4-bge-minilm`, and
`local-mellum2-12b-instruct-int4-bge-minilm`; these match dashboard manifest
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
