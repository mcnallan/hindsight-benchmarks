# Hindsight Benchmarks

The code and results behind [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io).

- **Benchmarks:** Hindsight as a whole on the [Agent Memory Benchmark](https://agentmemorybenchmark.ai): memory, and knowledge-bank search next to published numbers.
- **Model leaderboards:** which LLM to run behind each Hindsight operation (retain, consolidation, reflect and refresh pages), with and without a decision model.
- **Use cases:** Hindsight in a real task end to end: coding agents and long conversations, from the Agent Memory Benchmark.
- **Hindsight Models:** our decision, embedding and reranking models next to the ones people use today, from their Hugging Face cards.

## Layout

```
benchmark-runner/   hindsight-evals: runs Hindsight's system evals against a model
results/            one JSON per model run
visualizer/         the site (Next.js, static export)
```

## Run the evals for a model

For the workstation's Qwen and Inkling profiles, see
[Local Hindsight system evals](benchmark-runner/LOCAL_SYSTEM_EVALS.md) for
model controls, detached queues, selected cases, and resuming failed stages.

The runner pins a Hindsight commit in `benchmark-runner/evals.json`, checks it out, and runs its `hindsight-system-evals` one operation at a time.

```bash
cd benchmark-runner
export OPENROUTER_API_KEY=...
uv run hindsight-evals --model qwen/qwen3.8-flash                           # every operation
uv run hindsight-evals --model qwen/qwen3.8-flash --operations consolidation
uv run hindsight-evals --model qwen/qwen3.8-flash --decision internal:hs-decide-1-flash
```

- **Judge:** `gemini-3.1-flash-lite` (`judge_model` in `evals.json`), two passes per model.
- **Decision model:** `--decision <provider>:<model>`. `internal` reads its URL from `INTERNAL_BASE_URL`; serve hs-decide-1-flash with vLLM as its model card shows.
- **Cost:** the tokens each case really used, at the model's OpenRouter list price. hs-decide-1-flash is $0.02 per 1M input tokens, output free.
- **Results:** written to `results/leaderboard/evals/<provider>-<model>.json`. A run of some operations updates only those operations in the file.

## Update the site's data

```bash
cd benchmark-runner
amb_checkout=/absolute/path/to/agent-memory-benchmark
uv run hindsight-evals --export-site ../visualizer/lib/results.json            # model leaderboards
uv run hindsight-evals --export-structure ../visualizer/lib/eval-structure.json  # categories and cases
python pull_amb_coding.py "$amb_checkout"                                        # coding agents, conversations
python pull_amb_benchmarks.py "$amb_checkout"                                    # every card on the Benchmarks page
python pull_hf_cards.py                                                         # Hindsight Models, from the model cards
```

## Run the site

```bash
cd visualizer
npm install
npm run dev     # http://localhost:9998
```

The operation results appear under `/leaderboards`, with a local comparison
at `/leaderboards/local`. Historical facts-only BEAM results are at
`/leaderboard/retain`. See the [historical BEAM guide](benchmark-runner/LOCAL_QUALITY.md).
