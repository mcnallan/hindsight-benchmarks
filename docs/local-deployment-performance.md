# Local deployment performance: Qwen3.5 4B and Inkling 2 Mini

Snapshot of 25 November 2026. It covers our two self-hosted Hindsight model deployments on the
pinned Hindsight 1.2.0 system evals, the partial AMB memory benchmarks, and a paired
robustness check. The check tests whether the system-eval scores reflect real ability rather
than memorized test content.

## Summary

- **Inkling 2 Mini is the stronger local model.** It passes 94.2% of system-eval executions at
  minimal reasoning effort and 91.3% with thinking disabled. Qwen3.5 4B passes 62.0%. The gap is
  widest in Consolidation (100% / 96% vs 52%) and Refresh pages (82.5% vs 52.5%).
- **Minimal reasoning buys Inkling about 3 points**, mostly in Retain (97.6% vs 92.9%) and
  Consolidation. It costs about 184k reasoning tokens over the full suite, roughly 0.9k per
  execution. With thinking disabled, both models report zero reasoning tokens.
- **The paired check finds no sign of memorization.** On 12 cases, each beside a reworded copy
  and a copy with one fact changed, Inkling scores as well or better on the changed copies:
  24/24 on every counterfactual arm, with no stale answers. Its only failures are on one Retain
  case that tests for borrowed dates, and three of the four failed executions are grading false
  positives. Qwen's counterfactual arm is 4.2 points below stock, a net of one execution. Traces show
  ordinary reasoning errors, not a recalled stock answer.
- **The scores look trustworthy, within limits.** Each profile's ranking and gaps hold when
  questions are reworded or their answers changed. The check covers only 12 of the 104 stock cases
  and 24 executions per arm, so it would miss a contamination effect smaller than a few cases.
- **GPT-6.1 Terra at medium effort can replace GPT-6.2 Sol at high effort as the paired-suite
  judge.** Both graded all 58 fixture probes as expected. Thinking-disabled Inkling scored
  identically under both judges: 23/24 stock, 24/24 wording, 24/24 counterfactual.

## Deployments and test setup

| | Qwen3.5 4B | Inkling 2 Mini |
| --- | --- | --- |
| Profiles | `qwen35-4b-off` | `inkling-2-mini-off`, `inkling-2-mini-minimal` |
| Serving | Production Dynamo, Ampere vLLM recipe: 4-bit weights, INT8-eligible activations, FP8 E4M3 KV, FlashInfer | GPU2 HyperInkling 0.1.4 on Ampere: 4-bit experts, Humming W4A8, FP8 KV |
| Reasoning controls | `chat_template_kwargs.enable_thinking=false` | Off: `reasoning_effort=none` plus `logit_bias={"200008":-100}` to suppress the thinking-channel token. Minimal: `reasoning_effort=minimal` |
| Completion cap | 16,384 tokens for Retain, Reflect and Consolidation | same |

Shared by every run:

- **Hindsight:** pinned at `7b46a07c82f997903230f0fb33deafe9dfd2e22b` (1.2.0 system suite), each
  run against a fresh server and embedded PostgreSQL.
- **Retrieval:** BAAI BGE-small-en-v1.5 embeddings (384 dimensions) and the MS MARCO MiniLM-L6-v2
  cross-encoder reranker, served through LiteLLM.
- **Decision model:** disabled. Four cases per pass use upstream's scripted decision stub.
- **Judge:** GPT-6.2 Sol at high effort through codex-lb for the system evals, AMB and the first
  paired runs. GPT-6.1 Terra at medium effort for the later paired runs, marked as such below.
- **Repeats:** two of everything.

The public Hindsight reference boards use a Gemini judge, so these scores are not directly
comparable with them. The serving recipes live in the deployment repo
(`platform/inference/dynamo/prod-qwen35-4b/` and `platform/inference/vllm/inkling-3090/`). Check
live deployment settings before comparing against runs made after an engine or model change. The
full runbook is [LOCAL_SYSTEM_EVALS.md](../benchmark-runner/LOCAL_SYSTEM_EVALS.md).

## Main system evals

The pinned upstream suite has 104 cases (42 Retain, 17 Reflect, 25 Consolidation, 20 Refresh
pages), run twice, for 208 executions per profile. No profile had an errored execution.

| Operation | Qwen3.5 4B, thinking off | Inkling 2 Mini, thinking off | Inkling 2 Mini, minimal |
| --- | ---: | ---: | ---: |
| Retain | 69.0% (58/84) | 92.9% (78/84) | **97.6%** (82/84) |
| Reflect | 70.6% (24/34) | **91.2%** (31/34) | **91.2%** (31/34) |
| Consolidation | 52.0% (26/50) | 96.0% (48/50) | **100.0%** (50/50) |
| Refresh pages | 52.5% (21/40) | **82.5%** (33/40) | **82.5%** (33/40) |
| **All executions** | 62.0% (129/208) | 91.3% (190/208) | **94.2%** (196/208) |

### Where cases were lost

The table lists every category where at least one profile dropped an execution. All other
categories are at full marks for every profile.

| Operation | Category | Qwen off | Inkling off | Inkling minimal |
| --- | --- | ---: | ---: | ---: |
| Retain | selectivity | 4/6 | 6/6 | 6/6 |
| Retain | completeness | 2/4 | 4/4 | 4/4 |
| Retain | temporal | 4/8 | 8/8 | 8/8 |
| Retain | language | 11/14 | 12/14 | 13/14 |
| Retain | fidelity | 5/10 | 10/10 | 10/10 |
| Retain | mission | 2/4 | 4/4 | 4/4 |
| Retain | mode | 6/8 | 6/8 | 8/8 |
| Retain | entity labels | 4/6 | 6/6 | 6/6 |
| Retain | inputs | 2/6 | 6/6 | 6/6 |
| Retain | no invention | 6/6 | 4/6 | 5/6 |
| Reflect | update | 2/4 | 4/4 | 4/4 |
| Reflect | aggregation | 2/4 | 4/4 | 4/4 |
| Reflect | source authority | 5/8 | 6/8 | 6/8 |
| Reflect | pages | 1/4 | 3/4 | 3/4 |
| Consolidation | merge | 4/6 | 6/6 | 6/6 |
| Consolidation | update | 6/8 | 8/8 | 8/8 |
| Consolidation | counts | 0/4 | 4/4 | 4/4 |
| Consolidation | mission | 2/8 | 8/8 | 8/8 |
| Consolidation | scale | 2/6 | 6/6 | 6/6 |
| Consolidation | repetition | 0/6 | 4/6 | 6/6 |
| Refresh pages | precision | 5/6 | 6/6 | 6/6 |
| Refresh pages | update | 5/8 | 7/8 | 8/8 |
| Refresh pages | aggregation | 2/8 | 5/8 | 5/8 |
| Refresh pages | no overreach | 2/4 | 4/4 | 4/4 |
| Refresh pages | stability | 3/6 | 4/6 | 4/6 |
| Refresh pages | consistency | 0/4 | 3/4 | 2/4 |

Qwen's losses are broad, and four Consolidation and Refresh categories score zero or near it
(counts, repetition, consistency, mission). Inkling's remaining weak spots are shared by both of
its modes: Refresh pages aggregation, stability and consistency, Reflect source authority, and
Retain no invention. Retain no invention is the one category where Qwen beats Inkling. The paired
check below shows part of Inkling's loss there is grader noise.

### Token use

Totals over all 208 executions. Cached is the share of input served from the prefix cache. Output
is the visible final text and reasoning is counted apart, as Hindsight's OpenAI-compatible
provider reports them.

| Profile | Operation | Executions | Calls | Input | Cached | Output | Reasoning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B, thinking off | Retain | 84 | 128 | 431,692 | 88% | 62,473 | 0 |
| | Reflect | 34 | 156 | 2,157,406 | 61% | 71,715 | 0 |
| | Consolidation | 50 | 342 | 2,855,982 | 48% | 237,850 | 0 |
| | Refresh pages | 40 | 395 | 5,920,607 | 68% | 230,772 | 0 |
| | **All** | 208 | 1,021 | 11,365,687 | 62% | 602,810 | 0 |
| Inkling 2 Mini, thinking off | Retain | 84 | 128 | 478,802 | 92% | 44,809 | 0 |
| | Reflect | 34 | 149 | 1,910,674 | 39% | 21,352 | 0 |
| | Consolidation | 50 | 328 | 3,183,384 | 57% | 254,256 | 0 |
| | Refresh pages | 40 | 388 | 4,327,698 | 56% | 130,459 | 0 |
| | **All** | 208 | 993 | 9,900,558 | 55% | 450,876 | 0 |
| Inkling 2 Mini, minimal | Retain | 84 | 128 | 479,058 | 93% | 48,874 | 37,200 |
| | Reflect | 34 | 147 | 1,875,153 | 39% | 17,797 | 9,266 |
| | Consolidation | 50 | 328 | 3,181,182 | 57% | 251,432 | 89,826 |
| | Refresh pages | 40 | 359 | 3,719,144 | 55% | 109,330 | 47,593 |
| | **All** | 208 | 962 | 9,254,537 | 54% | 427,433 | 183,885 |

Qwen reads and writes the most tokens, mostly in longer Refresh pages loops, and still scores
lowest. Inkling's minimal mode adds 184k reasoning tokens (about 43% on top of its visible output)
but makes fewer Refresh calls than thinking-off mode and reads 6.5% less input.

## AMB memory benchmarks (partial)

The AMB queue ran AMB `01bd83f3`, graded by GPT-6.2 Sol at high effort, and is incomplete. Most
missing splits failed on infrastructure: codex-lb returned HTTP 429 rate limits during grading,
and one split hit a Hindsight 504. None failed on a model error. We cancelled Inkling minimal's
remaining splits during BEAM 100K to prioritize the paired check; finished splits are skipped
when it resumes.

| Split | Qwen3.5 4B, thinking off | Inkling 2 Mini, thinking off | Inkling 2 Mini, minimal |
| --- | ---: | ---: | ---: |
| PrecisionMemBench single-turn | 75.3% (58/77) | **92.2%** (71/77) | 90.9% (70/77) |
| LoCoMo-10 | failed (429) | **92.3%** (1,422/1,540) | 92.1% (1,418/1,540) |
| LoCoMo-10, fixed | 81.8% (1,229/1,502) | failed (504) | failed |
| PersonaMem 32K | failed (429) | 78.6% (463/589) | not run |
| BEAM 100K | failed (429) | 70.0% (315/400) | cancelled |
| LifeBench EN | failed (429) | failed (429) | not run |
| LongMemEval S | failed (429) | failed (429) | not run |

PrecisionMemBench is the only split all three profiles finished. It puts Inkling about 17 points
ahead of Qwen, consistent with the system evals. Its two Inkling modes are within one question.
Rerun the failed splits with `run_local_system_evals.sh <run dir> <queue> --select amb`, one queue
at a time, so codex-lb grades one benchmark at a time.

## Sanity check: paired robustness cases

### Method

Fixture set `7b46a07c-v2` takes 12 stock cases: 6 Retain and 6 Reflect, chosen for attribution,
invented facts and latest-state questions. Each gets two versions:

- **Wording:** new names, identifiers and phrasing, with the same relationships and the same
  answer.
- **Counterfactual:** one fact changed so the correct answer changes. Content, gold answers,
  traps and dates change together, and the traps explicitly reject the stock answer. We record
  when a counterfactual answer repeats the stock answer ("stale").

All three versions run in fresh banks under identical settings, twice each. That is 72 executions
per profile, with 24 per version. Before any model ran, the judge graded canned probe outputs to
confirm that each variant accepts its own answer and rejects the stock one. A model that
memorized the stock cases should do well on stock and wording but worse on counterfactuals, often
repeating the stock answer. The fixtures live in
[`src/hindsight_benchmark/paired/`](../benchmark-runner/src/hindsight_benchmark/paired/). The
visualizer's **Paired cases** tab shows every case side by side.

### Results

Δ is the variant's pass rate minus stock's over the same case and repeat. No run had an
infrastructure failure.

| Profile | Judge | Stock | Wording | Δ | Counterfactual | Δ | Stale |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B, thinking off | Sol high | 83.3% (20/24) | 87.5% (21/24) | +4.2 pp | 79.2% (19/24) | −4.2 pp | 2/24 |
| Inkling 2 Mini, thinking off | Sol high | 95.8% (23/24) | 100% (24/24) | +4.2 pp | 100% (24/24) | +4.2 pp | 0/24 |
| Inkling 2 Mini, thinking off | Terra medium | 95.8% (23/24) | 100% (24/24) | +4.2 pp | 100% (24/24) | +4.2 pp | 0/24 |
| Inkling 2 Mini, minimal | Terra medium | 95.8% (23/24) | 95.8% (23/24) | +0.0 pp | 100% (24/24) | +4.2 pp | 0/24 |

Per case, as stock · wording · counterfactual passes out of 2:

| Case | Qwen off (Sol) | Inkling off (Sol) | Inkling off (Terra) | Inkling minimal (Terra) |
| --- | --- | --- | --- | --- |
| Retain `coreference-roles` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Retain `multi-speaker-attribution` | 2 · **1** · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Retain `context-names-the-speaker` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Retain `in-text-correction` | **0 · 0 · 0** | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Retain `date-borrowed-from-nearby-subject` | 2 · 2 · 2 | **1** · 2 · 2 | **1** · 2 · 2 | **1 · 1** · 2 |
| Retain `owner-borrowed-from-nearby-subject` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `hq-owner-latest` | 2 · 2 · **0** | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `hq-absent-2024` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `hq-scoped-2fa` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `sp-rotation` | **0** · 2 · **1** | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `sp-upgrade-owner` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |
| Reflect `sp-postmortem` | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 | 2 · 2 · 2 |

### What the traces show

We read each failure's stored facts or answer, the retrieved rows and the judge's reasons, to
tell extraction, retrieval, generation and grading failures apart.

- **Inkling, `date-borrowed-from-nearby-subject` (4 failures, every run):**
  - Three are grading false positives. The model stored "notifications must be completed before
    fieldwork begins" without borrowing the project date. The trap judge read the
    "When: Before fieldwork begins" field as an invented deadline, and in one case only 2 of 3
    judges agreed.
  - One is a genuine borrowed date: thinking-off on Terra, stock, which stored "When: Before
    September 14".

  This case's trap was already unstable on known-correct stock output during fixture
  calibration. Without it, every Inkling arm is 24/24.
- **Inkling minimal effort sometimes adds a year** that the source never states ("September 14,
  2027"). No case penalizes this; it is worth watching in date-heavy workloads.
- **Qwen, `in-text-correction` (fails every version):** Qwen stores the corrected date one day
  early in stock, wording and counterfactual alike. Both counterfactual "stale" flags come from
  this same slip, not from recalling the stock answer.
- **Qwen, `hq-owner-latest` counterfactual (0/2):** generation errors. All four gold rows were
  retrieved, but Qwen misordered the timeline and named the wrong current owner.
- **Qwen, `sp-rotation` stock (0/2) while wording passed 2/2:** noise. Qwen passed the same stock
  case 2/2 in the earlier `v1` run, which used identical Reflect fixtures.
- **Qwen, `multi-speaker-attribution` wording (1/2):** a borderline extraction error. Jonas
  says "Then I'll handle the failover checklist", and Qwen stored that Jonas "offered to handle" it.
  Two of three judges failed that softened commitment. Every speaker was attributed correctly.

### Reading

- **Contamination:** neither model shows the memorization pattern: a high stock score with
  counterfactuals that fail or repeat the stock answer. Inkling's paired scores sit at or above
  its stock score in every run, and it never gave a stale answer. Qwen's ±4.2-point swings
  come from cases it gets wrong for visible, ordinary reasons.
- **Ability:** the ordering matches the main evals. Inkling is near ceiling on these 12 cases in
  both modes, and Qwen loses about one case in six. At this size the paired check cannot separate
  Inkling's two modes.
- **Limits:**
  - The paired check covers 12% of the stock suite, and none of Consolidation or Refresh pages,
    where Qwen's losses are largest. One execution is 4.2 points per arm, so only a large effect
    would show.
  - The `date-borrowed` trap needs a sturdier criterion before its results count for anything.
  - Qwen's paired runs were graded only by Sol. Inkling thinking-off's identical results under
    Sol and Terra suggest the judge change does not matter here.

## Sources

| What | Where |
| --- | --- |
| System-eval results | `results/leaderboard/evals/local-<profile>-sol62-high.json` |
| Paired reports (compact rows, no model text) | `results/paired/local-<profile>-<judge>-paired-7b46a07c-v2-9548f9fb.json` |
| AMB summaries | `results/runs/system-evals-20261124-e5Vai1/<profile>/amb/*/summary.json` (local, not committed) |
| Raw traces, outputs and judge reasons | `benchmark-runner/.hindsight/runs/` (local, not committed) |
| Runbook, queue and grader check | [LOCAL_SYSTEM_EVALS.md](../benchmark-runner/LOCAL_SYSTEM_EVALS.md) |

Regenerate the paired tables with:

```bash
benchmark-runner/.venv/bin/python benchmark-runner/report_paired.py \
  results/paired/local-*-paired-7b46a07c-v2-9548f9fb.json
```
