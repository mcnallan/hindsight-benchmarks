"""Shared machinery for the paired test modules: selection, stock lookup, records.

Runs inside the pinned evals venv, loaded by ``test_paired_retain.py`` and
``test_paired_reflect.py``. Selection comes from the environment the runner sets:

* ``HS_PAIRED_CASES`` — comma-separated stock case ids (default: every fixture case);
* ``HS_PAIRED_ARMS`` — comma-separated arms (default ``stock,wording,counterfactual``);
* ``HS_PAIRED_RECORDS`` — JSONL file that receives one record per case execution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from hindsight_benchmark.paired import FIXTURE_SET, revision
from hindsight_benchmark.paired.schema import ARMS, Arm, Kind, Variant

OPERATION_OF: dict[Kind, str] = {
    "retain_capability": "retain",
    "retain_fidelity": "retain",
    "reflect_corpus": "reflect",
    "reflect_sources": "reflect",
}


def _variants() -> tuple[Variant, ...]:
    from hindsight_benchmark.paired.fixtures import reflect_7b46a07c_v2 as reflect
    from hindsight_benchmark.paired.fixtures import retain_7b46a07c_v2 as retain

    return tuple(retain.VARIANTS) + tuple(reflect.VARIANTS)


def stock_variant(template: Variant) -> Variant:
    """The stock arm: the upstream case object and corpus, untouched."""
    from evals.retain.test_retain import FIDELITY_CASES
    from hindsight_system_evals import corpus, sources
    from hindsight_system_evals.retain_capabilities import CASES

    pools = {
        "retain_capability": (CASES, ()),
        "retain_fidelity": (FIDELITY_CASES, ()),
        "reflect_corpus": (corpus.build().questions, corpus.build().facts),
        "reflect_sources": (sources.build().questions, sources.build().facts),
    }
    cases, facts = pools[template.kind]
    case = next((c for c in cases if c.id == template.case_id), None)
    if case is None:
        raise RuntimeError(f"{template.case_id} is not a stock {template.kind} case at the pinned commit")
    return Variant(
        case_id=template.case_id, arm="stock", kind=template.kind, change="upstream case as pinned",
        stock_answer=template.stock_answer, expected_answer=template.stock_answer, case=case,
        corpus=tuple(facts),
    )


@dataclass(frozen=True)
class Execution:
    variant: Variant
    category: str

    @property
    def operation(self) -> str:
        return OPERATION_OF[self.variant.kind]


def executions(operation: str) -> list[Execution]:
    """Every selected (case, arm) for one operation, stock first within each case."""
    wanted_cases = {c for c in os.getenv("HS_PAIRED_CASES", "").split(",") if c}
    wanted_arms = [a for a in os.getenv("HS_PAIRED_ARMS", ",".join(ARMS)).split(",") if a]
    #: ``case@arm`` pairs, for rerunning only the executions infrastructure broke.
    only = {x for x in os.getenv("HS_PAIRED_ONLY", "").split(",") if x}
    unknown = set(wanted_arms) - set(ARMS)
    if unknown:
        raise RuntimeError(f"unknown arms {sorted(unknown)}")
    by_case: dict[str, dict[Arm, Variant]] = {}
    for v in _variants():
        by_case.setdefault(v.case_id, {})[v.arm] = v
    known = set(by_case)
    if wanted_cases - known:
        raise RuntimeError(f"no paired fixtures for {sorted(wanted_cases - known)}")
    out = []
    for case_id, arms in by_case.items():
        template = next(iter(arms.values()))
        if OPERATION_OF[template.kind] != operation or (wanted_cases and case_id not in wanted_cases):
            continue
        arms = {"stock": stock_variant(template), **arms}
        category = arms["stock"].case.category
        out.extend(Execution(arms[a], category) for a in ARMS
                   if a in wanted_arms and a in arms and (not only or f"{case_id}@{a}" in only))
    return out


def params(operation: str) -> list[object]:
    """pytest params id'd ``<category>/<case id>@<arm>`` so the stock tallies group by category."""
    return [pytest.param(e, id=f"{e.category}/{e.variant.case_id}@{e.variant.arm}") for e in executions(operation)]


def input_digest(variant: Variant) -> str:
    """sha256 of everything this execution feeds the model and the judge."""
    payload = {
        "case": dataclasses.asdict(variant.case),
        "corpus": [dataclasses.asdict(f) for f in variant.corpus],
        "stale_claim": variant.stale_claim,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Record:
    """One case execution. Graded fields stay None when infrastructure failed first."""

    case_id: str
    arm: str
    kind: str
    operation: str
    category: str
    input_sha: str
    fixture_revision: str
    fixture_set: str = FIXTURE_SET
    bank_id: str = ""
    passed: bool | None = None
    hit_trap: bool | None = None
    stale: bool | None = None
    stale_reason: str = ""
    #: extraction | retrieval | generation | deterministic | infrastructure | ok
    blame: str = ""
    reasons: list[str] = field(default_factory=list)
    output: str = ""
    gold_retrieved: int | None = None
    gold_total: int | None = None
    queries: list[str] = field(default_factory=list)
    infra_error: str = ""
    seconds: float = 0.0


@contextmanager
def recording(execution: Execution, bank_id: str) -> Iterator[Record]:
    """Yield a record to fill; write it whatever happens, classifying exceptions as infrastructure."""
    v = execution.variant
    record = Record(case_id=v.case_id, arm=v.arm, kind=v.kind, operation=execution.operation,
                    category=execution.category, input_sha=input_digest(v), fixture_revision=revision(),
                    bank_id=bank_id)
    start = time.monotonic()
    try:
        yield record
    except AssertionError as exc:
        # A graded failure raised inside the stock runner (no facts stored, no answer text).
        if record.passed is None:
            record.passed = False
            record.blame = record.blame or ("extraction" if execution.operation == "retain" else "generation")
            record.reasons.append(str(exc)[:1000])
        raise
    except BaseException as exc:
        # Anything but a graded assertion: the server, the model endpoint or the judge failed.
        record.blame = "infrastructure"
        record.passed = None
        record.infra_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        record.reasons.append(traceback.format_exc(limit=3)[-1500:])
        raise
    finally:
        record.seconds = round(time.monotonic() - start, 1)
        if path := os.getenv("HS_PAIRED_RECORDS"):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def stale_criterion(variant: Variant) -> str:
    """Retain stale claims are written about the facts; reflect ones are propositions, worded like the trap."""
    if variant.kind.startswith("reflect"):
        return f"The answer asserts, as currently true, that {variant.stale_claim.rstrip('.')}."
    return variant.stale_claim


async def stale_check(record: Record, variant: Variant, response: str, context: str | None) -> None:
    """Did the output repeat the stock answer despite the changed evidence? Recorded, never scored."""
    if variant.arm != "counterfactual" or not variant.stale_claim or not response.strip():
        return
    from hindsight_system_evals import evaluate

    verdict = await evaluate(response, stale_criterion(variant), context=context, confirm_met=True)
    record.stale, record.stale_reason = verdict.meets_criteria, verdict.reasoning
