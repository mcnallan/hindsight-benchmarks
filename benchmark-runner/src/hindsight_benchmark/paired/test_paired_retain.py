"""Paired Retain: each selected stock case, then its wording and counterfactual arms.

Grading is the stock code path: ``retain_capabilities.run_case`` for capability
cases, ``test_retain._run_fidelity`` for no-invention cases, with the same
assertions in the same order (trap first). Each arm gets its own fresh bank.
"""

from __future__ import annotations

import pytest
from evals.retain.test_retain import _assert_fidelity_sound, _run_fidelity
from hindsight_client import Hindsight
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord
from hindsight_system_evals.retain_capabilities import _input, run_case

from hindsight_benchmark.paired.runner import Execution, params, recording, stale_check


async def _capability(client: Hindsight, bank_id: str, settled: SettleFn, execution: Execution) -> None:
    variant, case = execution.variant, execution.variant.case
    with recording(execution, bank_id) as record:
        outcome = await run_case(client, bank_id, settled, case)
        failed = outcome.missing + outcome.broken
        record.output = outcome.rendered
        record.hit_trap = bool(outcome.trapped)
        record.passed = not outcome.trapped and not failed
        record.reasons = outcome.trapped + failed
        if record.passed:
            record.blame = "ok"
        elif any("retain stored no facts" in b and "error=" in b for b in outcome.broken):
            record.blame = "infrastructure"
        elif outcome.trapped or outcome.missing:
            record.blame = "extraction"
        else:
            record.blame = "deterministic"
        context = f"These facts were extracted, by a memory system, from this input:\n\n{_input(case)}"
        await stale_check(record, variant, outcome.rendered if outcome.facts else "", context)
    RECORDED.append(EvalRecord(kind="retain", question_id=f"{case.id}@{variant.arm}", category=case.category,
                               correct=not failed, hit_trap=bool(outcome.trapped), bank_id=bank_id,
                               reason="; ".join(outcome.trapped or failed) or "ok"))
    evidence = f"bank {bank_id} ({case.origin}; arm {variant.arm})\nfacts:\n{outcome.rendered}"
    assert not outcome.trapped, f"{case.id}: stored a falsehood —\n" + "\n".join(outcome.trapped) + f"\n{evidence}"
    assert not failed, f"{case.id}:\n" + "\n".join(failed) + f"\n{evidence}"


async def _fidelity(client: Hindsight, bank_id: str, settled: SettleFn, execution: Execution) -> None:
    variant = execution.variant
    with recording(execution, bank_id) as record:
        outcome = await _run_fidelity(client, bank_id, settled, variant.case)
        record.output = outcome.facts
        record.hit_trap = outcome.record.hit_trap
        record.passed = outcome.record.correct and not outcome.record.hit_trap
        record.reasons = [outcome.record.reason]
        record.blame = "ok" if record.passed else "extraction"
        await stale_check(record, variant, outcome.facts, None)
    _assert_fidelity_sound(outcome)


@pytest.mark.parametrize("execution", params("retain"))
async def test_paired_retain(client: Hindsight, bank_id: str, settled: SettleFn, execution: Execution) -> None:
    if execution.variant.kind == "retain_fidelity":
        await _fidelity(client, bank_id, settled, execution)
    else:
        await _capability(client, bank_id, settled, execution)
