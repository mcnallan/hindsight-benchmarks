# Local models for Hindsight retain and reflect

Research date: 2026-09-24. This is a test shortlist, not a claimed Hindsight ranking. Hugging Face figures below come from model authors' cards; the only Hindsight score cited is a saved result in this repository. No new checkpoint was downloaded or benchmarked for this report.

## Recommendation

**Best first non-reasoning test:** [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) with thinking disabled, using a verified 4-bit vLLM-compatible quantization such as [cyankiwi AWQ](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit). It retains the favorable hybrid KV architecture, offers more capacity than the 4B baseline, and has an official, well-documented source model. The quantization and engine must pass the actual Hindsight strict-schema retain workload.

The official Qwen card reports 9B above 4B on IFEval (91.5 vs 89.8) and AA-LCR long-context reasoning (63.0 vs 57.0). These are useful directional signals, not evidence of better Hindsight facts or latency.

**Best low-cost non-reasoning control:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) with `chat_template_kwargs.enable_thinking=false`. This already has a saved BEAM facts-only result of **31/80 (38.8%)** under strict extraction, eight retain workers, and 16,384 maximum completion tokens in [local-qwen3.5-4b-no-thinking-strict.json](../results/leaderboard/llm/local-qwen3.5-4b-no-thinking-strict.json). Use it as the same-run baseline, not as a universal model score.

**Best first efficient-reasoning test:** the current [Jackrong Qwen3.5-4B Opus distilled v2](https://huggingface.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2), followed by its [9B v2](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2) if a verified quantized serving path is available. The user's direct example (786 total completion tokens and about 300 reasoning tokens, versus 5,302/about 5,200 for base 4B) is more relevant to this decision than coding benchmarks, but it is one extraction example, not a quality score. The 9B v2 card claims a 22% shorter average reasoning trace than base on its own task; 32k extraction quality is unproved.

**Best reasoning quality challenger:** [Qwopus3.5-9B-v3](https://huggingface.co/Jackrong/Qwopus3.5-9B-v3), with [cpatonn AWQ](https://huggingface.co/cpatonn/Qwopus3.5-9B-v3-AWQ-4bit) as a possible serving artifact. Its card reports better coding and a small MMLU-Pro subset result than stock/v2, but explicitly says its chain of thought is somewhat longer than v2. Compare quality *per completion token* and wall time. [Qwopus3.5-4B-v3](https://huggingface.co/Jackrong/Qwopus3.5-4B-v3) serves the same purpose at 4B, and its card says reasoning is significantly longer than 4B v2.

## What this workload actually asks the model to do

The BEAM quality runner gives each conversation session to `retain`, then asks 20 questions per conversation from **facts-only recall**. The answer generator and judge are separate models. Its 80 questions cover ten abilities evenly: abstention, contradiction resolution, event ordering, information extraction, instruction following, knowledge update, multi-session reasoning, preference following, summarization, and temporal reasoning. The four sessions span coding, recommendation, lifestyle, and therapy. See [quality.py](../benchmark-runner/src/hindsight_benchmark/quality.py), [the frozen subset](../benchmark-runner/datasets/beam_128k_subset.json), and [the Hindsight extraction code](../../hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py).

The repo's faster leaderboard test also asks the candidate to extract facts into strict JSON from 20 diverse conversation scenarios ([benchmark.py](../benchmark-runner/src/hindsight_benchmark/benchmark.py), [simple.json](../benchmark-runner/datasets/simple.json)). It is a useful format/latency screen, while BEAM adds whether the facts support later answers.

Retain requires a potentially long `{"facts": [...]}` object: selective but complete atomic facts, named entities, coreference, event dates resolved from relative dates, and event/state and world/assistant distinctions. It uses an OpenAI-compatible structured response, validation and retries. The local Qwen strict run records a 50,000-token retain batch setting and a 16,384-token completion cap. A short JSON toy probe cannot establish that a model will finish the large facts array. Hindsight can split on output-too-long errors, but truncation/retries still harm throughput and may lose facts. The saved 4B no-thinking result had zero generator/judge errors but only 0/8 on contradiction and event ordering and 0/8 on summarization; these categories deserve case-level inspection, not just an aggregate score.

For production `reflect`, the model also has to call search tools and synthesize an answer from recalled facts, handle uncertainty and dates, and stay within a configurable context budget. The default reflect context target is about 100k tokens; a 32k serving window requires an explicit lower target. Retain and reflect can use separate per-operation model settings. This BEAM facts-only run measures the **retain model**, not a candidate reflect model; test reflect separately before using one model for both.

## GPU fit: eight simultaneous 32k sequences on one 24 GB RTX 3090

The cache estimates use the model `config.json` architecture and count *allocated full-attention KV entries*: `8 streams × 32,768 tokens × full-attention layers × 2 (K,V) × KV heads × head dimension × bytes/element`. They are a conservative simultaneous-occupancy target, not a measured vLLM allocation. The 24/32 linear-attention layers in Qwen3.5 also need recurrent state and runtime buffers. Vision weights, MTP heads, quantization metadata, CUDA graphs, fragmentation, and output tokens add memory. Actual fit requires an engine boot and eight running requests, not just `max_num_seqs=8`.

| Architecture | Full-attention config | BF16 KV, 8 × 32k | FP8 KV, 8 × 32k | Fit judgment |
| --- | --- | ---: | ---: | --- |
| Qwen3.5 dense 4B or 9B | 8 layers, 4 KV heads, 256 dim | **8 GiB** | **4 GiB** | 4B BF16 plausible; 9B needs weight quantization |
| Qwen3.5 35B-A3B MoE | 10 layers, 2 KV heads, 256 dim | 5 GiB | 2.5 GiB | 3B *active* is misleading: all 35B weights must reside somewhere |
| Qwen3-4B-Instruct-2507 | 36 layers, 8 KV heads, 128 dim | 36 GiB | 18 GiB | Cannot meet target with BF16 KV; FP8 plus 4-bit weights is tight |
| Ministral-3-3B | 26 layers, 8 KV heads, 128 dim | 26 GiB | 13 GiB | FP8 KV plus compact weights may fit; needs measurement |
| Ministral-3-8B | 34 layers, 8 KV heads, 128 dim | 34 GiB | 17 GiB | Poor eight-stream fit even with FP8 KV |

The native 9B BF16 checkpoint is roughly 18 GiB of text weights before vision/runtime allocations: **18 + 8 > 24 GiB**, so it cannot satisfy the BF16-KV target. A 9B 4-bit artifact around 6–9 GB on disk plus 8 GiB BF16 KV is plausible, with FP8 KV creating more headroom if the chosen vLLM version handles this hybrid architecture correctly. File size is not exact resident VRAM. The [Defiant Fable W4A16 repo](https://huggingface.co/TheUnderscore/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-W4A16-AWQ) reports about 9.09 GB of repository storage; its card reports LMDeploy testing, not vLLM validation. A 35B-A3B 4-bit weight set is roughly 18–22 GB *before* KV, so its appealing active-parameter count does not make it a primary 24 GB/eight-stream candidate. CPU expert offload or lower-bit weights would change the throughput/quality problem materially.

Because 32k is the **total** prompt plus generated tokens for a request, leave room for Hindsight's long JSON output. A 32k `max_model_len` with a 32k input cannot return the required facts. Capture actual extraction request lengths: the configured 50k retain batch ceiling does not imply every inference request fits a 32k served window.

## Candidate tiers

| Priority | Model and mode | Why test it | Main uncertainty |
| --- | --- | --- | --- |
| 1, direct | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), thinking off, 4-bit | Most promising quality/fit balance; official model; same favorable KV shape as 4B | Quantized strict JSON quality and real 8-way vLLM throughput |
| 1, direct control | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), thinking off | Existing 31/80 result and proven local eight-request serving path | Already weak on some BEAM abilities |
| 1, reasoning control | [Jackrong 4B Opus v2](https://huggingface.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2) | User-observed much shorter reasoning on one facts extraction | Need full BEAM quality and completion-token distribution |
| 2, reasoning | [Jackrong 9B Opus v2](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2) | More capacity; author claims shorter traces than stock | Quantized v2 vLLM artifact/provenance; long JSON and 32k training transfer |
| 2, reasoning quality | [Qwopus3.5-9B-v3](https://huggingface.co/Jackrong/Qwopus3.5-9B-v3) | Author reports better coding and small-set knowledge results | More thinking than v2; model-assisted coding adjudication; no BEAM proof |
| 2, direct/auto thinking | [Defiant Fable 9B](https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP) | User-suggested option; card claims compact thinking and improved ARC-C | Creative/uncensored tuning can drift from precise extraction; instruct and thinking need separate probes |
| 3, direct control | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Official pure non-thinking model, 262k native context | Full-attention KV makes eight 32k slots hard on 3090 |
| 3, direct alternate | [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Official FP8 checkpoint, card explicitly targets JSON/data extraction | Needs FP8 KV and actual 8-way memory test; different tokenizer/JSON mode |
| 3, reasoning alternate | [Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) | Small, 256k, official reasoning checkpoint | No evidence of short chains; vendor recommends reasoning-specific prompt/temp |
| 3, direct alternate | [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B, 128k, instruction/tool-call control | Full-attention KV and 32k eight-way fit need calculation/probe |
| exploratory | [Jackrong Qwen3.5-4B-Neo](https://huggingface.co/Jackrong/Qwen3.5-4B-Neo) | Card reports shorter traces than base and small-set MMLU-Pro improvement | Programming/math-heavy tuning, no extraction proof |

[Gemma-3-4B-it](https://huggingface.co/google/gemma-3-4b-it) is lower priority because its card lists an 8,192 output-token limit, which may truncate large retain arrays. [agentlans Qwen3.5-4B-Instruct-SingleTurn](https://huggingface.co/agentlans/Qwen3.5-4B-Instruct-SingleTurn) suppresses chain of thought but its reported 1,024-token SFT packing makes 32k extraction behavior uncertain. [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), including the [Opus distill](https://huggingface.co/Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled), is a useful larger-hardware ceiling, not a primary single-3090 candidate. MTP and speculative-draft repositories are acceleration artifacts; they do not by themselves establish better Hindsight facts or 3090 throughput.

## Test order and pass criteria

1. Reproduce the saved Qwen3.5-4B no-thinking strict result under the same dataset, retrieval, answer generator, judge, cap, and eight-way serve settings. Keep the stored facts and traces for case comparison.
2. Run the same representative **one-conversation/two-question smoke** on Qwen3.5-9B non-thinking 4-bit, the current 4B Opus v2, and 9B Opus v2. Use strict facts schema, not a short synthetic JSON object. Check `finish_reason`, schema validity after retries, fact count and stored-fact tokens, recall returning facts, generation/judge errors, and reasoning vs visible output tokens. If a model truncates repeatedly, diagnose that before a full run.
3. Run eight concurrent extraction requests of realistic lengths while reading the serving engine's **running and waiting** counts, actual GPU memory, prefix/KV cache metrics, effective `max_model_len`, and p50/p95 latency. The local launcher previously needed `MAX_NUM_SEQS=8` *and* container recreation; eight client workers alone did not prove eight active server sequences.
4. Run the same four-session/80-question BEAM facts-only evaluation for models that pass the smoke and fit gate. Compare per-ability correctness, stored facts and omitted details, schema/retry failure rate, retain wall time, and completion tokens per valid fact. Reserve a separate tool-calling/reflect test for the selected retain winners.

Keep `enable_thinking` in the top-level OpenAI request via `extra_body={"chat_template_kwargs":{"enable_thinking":false}}` for official Qwen3.5 when testing direct mode. The Jackrong v2/v3 published tokenizer templates appear to open `<think>` unconditionally and lack that switch; verify the *deployed* template and wire response before assuming these fine-tunes support a non-thinking mode. Validate model-card license/provenance and quant source before production use; a quant with a similar name may point to the prior v1 checkpoint.

### Evidence limits

Model-card coding, ARC-C, MMLU-Pro, and Arena scores are weak proxies for long-form Hindsight fact extraction. The Jackrong 9B v2 card reports 1,778 average thinking characters versus 2,284 for base on its own evaluation; Qwopus 9B v3 reports fewer thinking characters than base on a 280-question subset but slightly longer traces than v2. The 4B v3 card says substantially longer traces than 4B v2. None of those establish retained-fact coverage, strict-schema reliability, or eight-way 3090 serving. No new model was run for this report.
