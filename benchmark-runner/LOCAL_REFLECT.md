# Paired local reflect evaluation

`run_local_reflect.py` compares the retained RedHat W8A8 and current production
W4A8 recipes using the existing LoComo `conv-43` reflect suite: 29 source
sessions, 242 original questions, original gold/rubric and `budget="low"`.
This is separate from the 80-question BEAM retain-quality suite.

The memory bank is retained once by GPT-6 Sol at medium effort and shared by
both read-only reflect services using one isolated PostgreSQL database. Both
use BGE-small/MiniLM retrieval, GPT-6 Sol judging, explicit thinking OFF, a
4,096-token reflect completion cap, and Hindsight's native LiteLLM reflect
provider. The production capture explicitly pins the current GPU2 worker with
Dynamo's supported body/header controls. Normal production routing is unchanged.

The scoped reflect extra body also supplies `max_completion_tokens: 4096`:
the tested Hindsight caller did not send its configured completion cap on tool
rounds. LiteLLM extras fill missing per-call parameters; explicit caller values
still win. Verify the effective cap in every captured tool request instead of
inferring it from an environment variable. No upstream package is patched.

These conditions differ from the older reflect orchestrator's Gemini retain,
Gemini Flash Lite judge and RRF retrieval; record them rather than attributing
historical score differences to quantization. The existing LoComo dataset has
known evaluation limitations documented in the repository README. This runner
does not rewrite questions, answers or judging prompts.

Supply `CODEX_LB_TOKEN` explicitly through the process environment. Optional
`QUALITY_RETRIEVAL_API_KEY` overrides the local Hindsight retrieval credential;
credentials stay in temporary mode-0600 environment files, are excluded from
wire headers, and are redacted from saved Hindsight logs. Full weights stay on
the GPU serving host. Selected model deployment JSON must already be available.

```bash
benchmark-runner/.venv/bin/python benchmark-runner/run_local_reflect.py \
  --run-dir results/runs/NEW_REFLECT_RUN \
  --redhat-provenance results/runs/REDHAT_RUN/selected-deployment.json \
  --production-provenance results/runs/PRODUCTION_RUN/selected-deployment.json
```

Use a fresh run directory; existing runs are not restarted. The runner verifies
model advertisement and the actual GPT-6 Sol judge call shape, retains the bank,
checks all captured retain JSON against its schema, saves a private database
dump, then smoke-tests two original questions on each target concurrently.
It pauses at `awaiting_source_review`; inspect both `smoke-questions.private.json`
files, their generated answers/judge reasoning and native tool wire captures.

After an actual review, write `review-approved.json` with `smoke_sha256` mapping
`redhat` and `production` to the current smoke-file SHA256 hashes, and include
a bounded review report/findings. This is an evaluator checkpoint, not a new
human permission requirement. Only matching smoke hashes release full. Technical
smoke failures stop visibly; strict judging never silently falls back to string
matching. Full partial/error counts are preserved.

The two full loops run concurrently, serially within each model as in the
existing suite. Native tool-call finish reasons are valid. Wire checks also
inspect thinking controls, JSON tool arguments and complete responses. Separate
private traces save each completed question; smoke does not save leaderboard
scores. Full reflect results merge into each existing model file while retaining
its retain-quality result. A completed score is distinct from clean technical
traces. Server captures and database dump remain private; temporary containers
and credentials are cleaned up after the run. Keep the RedHat SSH forward alive.

Availability probes are excluded from extraction checks; actual transport,
schema and tool-call failures remain visible. Preparation uses two-second
operation polling without changing the existing BEAM polling default or retain
requests. Successful preparation saves `prepared-bank.json`; pass it with
`--prepared-bank` to reuse that exact validated dump in a fresh paired run.
Dataset, retain model, retrieval profile and dump hash must match. A database
dump is saved before final retain validation so a technical failure does not
discard the only retained-bank evidence. Preserve failed setup directories.

```bash
watch -n 10 'bash benchmark-runner/monitor_local_reflect.sh results/runs/NEW_REFLECT_RUN'
```

Read `status.json`, `reflect.log`, per-model question/trace summaries and the
combined private wire captures. Production's worker pin is stored in deployment
metadata and every captured LLM request. A changed GPU2 worker requires a fresh
pin/run. Do not use another inference endpoint while claiming this provenance.
