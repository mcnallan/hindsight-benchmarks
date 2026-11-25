"""Paired Reflect: each selected stock question, then its wording and counterfactual arms.

Mirrors ``test_reflect._run`` and ``_run_sources`` line for line, except that the
corpus is the arm's own instead of the module-level stock corpus. Each arm seeds a
fresh bank with its complete corpus, asks once, and grades trap and correctness
with the stock criteria wording.
"""

from __future__ import annotations

import dataclasses

import pytest
from evals.reflect.test_reflect import _assert_sound, _assert_sources_sound
from hindsight_client import Hindsight
from hindsight_system_evals import ask, evaluate, seed_bank, sources
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

from hindsight_benchmark.paired.runner import Execution, params, recording, stale_check


@pytest.mark.parametrize("execution", params("reflect"))
async def test_paired_reflect(client: Hindsight, bank_id: str, settled: SettleFn, execution: Execution) -> None:
    variant = execution.variant
    question, corpus = variant.case, list(variant.corpus)
    from_sources = variant.kind == "reflect_sources"
    with recording(execution, bank_id) as record:
        if from_sources:
            question = dataclasses.replace(question, question=sources.decorate_query(question.question))
            await sources.seed_bank(client, bank_id, corpus, settled)
        else:
            await seed_bank(client, bank_id, corpus, settled)
        outcome = await ask(client, bank_id, question, corpus)
        record.output = outcome.answer
        record.queries = outcome.queries
        record.gold_retrieved, record.gold_total = outcome.gold_retrieved, outcome.gold_total
        assert outcome.answer, f"{question.id}: reflect returned no text — bank {bank_id}"

        context = f"Question asked: {question.question}"
        verdict = await evaluate(outcome.answer, question.answer_criteria, context=context)
        outcome.correct, outcome.correct_reason = verdict.meets_criteria, verdict.reasoning
        # The corpus suite skips the trap when a question has none; the sources suite always asks it.
        if question.must_not_claim or from_sources:
            trap = await evaluate(
                outcome.answer, f"The answer asserts, as currently true, that {question.must_not_claim}.",
                context=context,
            )
            outcome.hit_trap, outcome.trap_reason = trap.meets_criteria, trap.reasoning
        record.hit_trap = outcome.hit_trap
        record.passed = outcome.correct and not outcome.hit_trap
        record.reasons = [r for r in (outcome.trap_reason if outcome.hit_trap else "", outcome.correct_reason) if r]
        if record.passed:
            record.blame = "ok"
        elif outcome.gold_total and outcome.gold_retrieved < outcome.gold_total:
            record.blame = "retrieval"
        else:
            record.blame = "generation"
        await stale_check(record, variant, outcome.answer, context)
    RECORDED.append(EvalRecord(kind="source_priority" if from_sources else "reflect",
                               question_id=f"{question.id}@{variant.arm}", category=question.category,
                               correct=outcome.correct, hit_trap=outcome.hit_trap, bank_id=bank_id,
                               reason=outcome.trap_reason if outcome.hit_trap else outcome.correct_reason))
    (_assert_sources_sound if from_sources else _assert_sound)(outcome)
