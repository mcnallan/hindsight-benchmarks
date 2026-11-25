"""Grade every fixture probe with the stock criteria and the run's judge, before any model runs.

  uv run --project <pinned>/hindsight-system-evals python -m hindsight_benchmark.paired.check_graders OUT.json

Each probe is a canned output with an expected whole-case verdict. The criteria
are built exactly as the stock runners build them (trap first, then the
must/correct/answer criteria), so a probe that passes here exercises the same
judge prompts a live run sends. Deterministic retain checks (entities, labels,
contains) need a live bank and are checked statically by ``validate``.

Needs ``HINDSIGHT_EVAL_JUDGE_MODEL``, ``HS_BENCH_CODEX_LB_URL`` and ``CODEX_LB_TOKEN``.
Exit status 1 when any probe disagrees with its expectation.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from hindsight_benchmark import evals_plugin  # noqa: F401  (routes the judge through codex-lb)
from hindsight_benchmark.paired import revision
from hindsight_benchmark.paired.runner import _variants, stale_criterion
from hindsight_benchmark.paired.schema import Probe, Variant


async def _grade(variant: Variant, response: str) -> dict:
    """The stock grading order for this kind: traps, then correctness. Returns every verdict."""
    from hindsight_system_evals import evaluate, sources
    from hindsight_system_evals.retain_capabilities import _input

    case, verdicts = variant.case, []

    async def judge(role: str, criterion: str, context: str | None, **kw) -> bool:
        v = await evaluate(response, criterion, context=context, **kw)
        verdicts.append({"role": role, "criterion": criterion, "met": v.meets_criteria, "reasoning": v.reasoning})
        return v.meets_criteria

    stale_context: str | None
    if variant.kind == "retain_capability":
        context = stale_context = f"These facts were extracted, by a memory system, from this input:\n\n{_input(case)}"
        trapped = [await judge("trap", t, context) for t in case.traps]
        musts = [await judge("must", m, context) for m in case.must]
        passed = not any(trapped) and all(musts)
    elif variant.kind == "retain_fidelity":
        stale_context = None
        trapped = await judge(
            "trap",
            f"Consider ONLY the fact about {case.subject}, read entirely on its own. "
            f"That single fact {case.must_not_carry}. "
            "Judge that fact in isolation: do not combine it with any other fact in the list, and do not "
            "reason about what the facts imply together. A value appearing only in a different fact does "
            "not count.",
            None,
        )
        correct = await judge("correct", f"One of these facts records that {case.must_extract}.",
                              f"These facts were extracted from this document:\n\n{case.document}")
        passed = correct and not trapped
    else:
        question = case.question
        if variant.kind == "reflect_sources":
            question = sources.decorate_query(question)
        context = stale_context = f"Question asked: {question}"
        correct = await judge("answer", case.answer_criteria, context)
        trapped = await judge("trap", f"The answer asserts, as currently true, that {case.must_not_claim}.", context)
        passed = correct and not trapped
    stale = None
    if variant.stale_claim:
        stale = await judge("stale", stale_criterion(variant), stale_context, confirm_met=True)
    return {"passed": passed, "stale": stale, "verdicts": verdicts}


async def _check(variant: Variant, probe: Probe) -> dict:
    graded = await _grade(variant, probe.response)
    ok = graded["passed"] == probe.expect_pass and (probe.expect_stale is None or graded["stale"] == probe.expect_stale)
    return {"case_id": variant.case_id, "arm": variant.arm, "probe": probe.label, "ok": ok,
            "expect_pass": probe.expect_pass, "expect_stale": probe.expect_stale, **graded}


async def main_async(out: Path, only: set[str]) -> int:
    jobs = [(v, p) for v in _variants() if not only or v.case_id in only for p in v.probes]
    results = await asyncio.gather(*(_check(v, p) for v, p in jobs), return_exceptions=True)
    rows = []
    for (v, p), r in zip(jobs, results):
        if isinstance(r, BaseException):
            r = {"case_id": v.case_id, "arm": v.arm, "probe": p.label, "ok": False,
                 "error": f"{type(r).__name__}: {str(r)[:300]}"}
        rows.append(r)
        mark = "ok  " if r["ok"] else "FAIL"
        detail = r.get("error") or f"passed={r['passed']} (want {p.expect_pass}) stale={r['stale']} (want {p.expect_stale})"
        print(f"{mark} {v.case_id}@{v.arm} :: {p.label} :: {detail}", flush=True)
    out.write_text(json.dumps({"fixture_revision": revision(), "probes": rows}, indent=1, ensure_ascii=False) + "\n")
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} probes graded as expected; wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(Path(sys.argv[1]), set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set())))
