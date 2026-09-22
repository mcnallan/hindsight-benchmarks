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
