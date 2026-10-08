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
export QUALITY_LITELLM_API_KEY=...
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
