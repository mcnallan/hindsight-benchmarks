"""Write the paired fixtures, stock beside each variant, as JSON for the visualizer.

  uv run --project <pinned>/hindsight-system-evals python -m hindsight_benchmark.paired.export_visualizer OUT.json

Every variant field carries a word-level diff against the same field of the stock
case, so the page can mark exactly what a wording or counterfactual arm changed.
Reflect corpora are large (hundreds of rows); only the rows an arm changed and the
gold rows are exported.
"""

from __future__ import annotations

import dataclasses
import datetime
import difflib
import json
import re
import sys
from pathlib import Path

from hindsight_benchmark.paired import FIXTURE_SET, PINNED_COMMIT, revision
from hindsight_benchmark.paired.runner import OPERATION_OF, _variants, stale_criterion, stock_variant
from hindsight_benchmark.paired.schema import Variant

_TOKEN = re.compile(r"\s+|[^\s\w]|\w+")


def diff(old: str, new: str) -> list[dict]:
    """Word-level segments turning ``old`` into ``new``: eq / del / ins."""
    a, b = _TOKEN.findall(old or ""), _TOKEN.findall(new or "")
    out: list[dict] = []

    def push(op: str, text: str) -> None:
        if not text:
            return
        if out and out[-1]["op"] == op:
            out[-1]["text"] += text
        else:
            out.append({"op": op, "text": text})

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            push("eq", "".join(a[i1:i2]))
        else:
            push("del", "".join(a[i1:i2]))
            push("ins", "".join(b[j1:j2]))
    return out


def _item(text: str, stock: str | None, tag: str = "") -> dict:
    row = {"text": text}
    if tag:
        row["tag"] = tag
    if stock is not None and stock != text:
        row["diff"] = diff(stock, text)
    return row


def _list(values: list[str], stock: list[str] | None) -> list[dict]:
    """Items aligned by position; an added item diffs against nothing."""
    if stock is None:
        return [_item(v, None) for v in values]
    return [_item(v, stock[i] if i < len(stock) else "") for i, v in enumerate(values)]


def _sections(v: Variant, stock: Variant | None) -> list[dict]:
    c, s = v.case, stock.case if stock else None
    out = []

    def text(label: str, value: str, old: str | None, hint: str = "") -> None:
        if value or old:
            out.append({"label": label, "hint": hint, "items": [_item(value or "", old if stock else None)]})

    def items(label: str, values: list[str], old: list[str] | None, hint: str = "") -> None:
        if values or old:
            out.append({"label": label, "hint": hint, "items": _list(list(values), list(old) if stock else None)})

    if v.kind == "retain_capability":
        for n, it in enumerate(c.items):
            old = s.items[n] if s and n < len(s.items) else None
            head = " · ".join(x for x in (f"event date {it.timestamp[:10]}" if it.timestamp else "",
                                          f"context: {it.context}" if it.context else "") if x)
            old_head = (" · ".join(x for x in (f"event date {old.timestamp[:10]}" if old.timestamp else "",
                                               f"context: {old.context}" if old.context else "") if x)
                        if old else None)
            label = "Retained input" if len(c.items) == 1 else f"Retained input {n + 1}"
            if head or old_head:
                text(f"{label} · metadata", head, old_head)
            text(label, it.content, old.content if old else None)
        items("Must record", c.must, s.must if s else None, "Each is its own judge call on the stored facts.")
        items("Traps", c.traps, s.traps if s else None, "Any one met fails the case, before the musts are read.")
        items("Entities the bank must hold", c.entities, s.entities if s else None)
        items("Text a fact must contain", c.contains, s.contains if s else None)
    elif v.kind == "retain_fidelity":
        text("Retained document", c.document, s.document if s else None)
        text("Must extract", c.must_extract, s.must_extract if s else None, "Judged with the document as context.")
        text("Fact the trap reads alone", c.subject, s.subject if s else None)
        text("Trap: that fact", c.must_not_carry, s.must_not_carry if s else None,
             "Judged on that one fact, without the document.")
    else:
        text("Question", c.question, s.question if s else None)
        text("Correct when", c.answer_criteria, s.answer_criteria if s else None)
        text("Trap: the answer asserts, as currently true, that", c.must_not_claim, s.must_not_claim if s else None)
        stock_facts = {f.id: f for f in (stock.corpus if stock else v.corpus)}
        gold = set(c.gold)
        rows = []
        for f in v.corpus:
            old = stock_facts.get(f.id)
            changed = stock is not None and (old is None or old.text != f.text or old.metadata != f.metadata)
            if f.id in gold or changed:
                tags = [f.id] + (["gold"] if f.id in gold else []) + [f"{k}={x}" for k, x in f.metadata.items()]
                rows.append(_item(f.text, old.text if (stock and old) else ("" if stock else None), " · ".join(tags)))
        hint = (f"{len(v.corpus)} rows seeded. Shown: gold rows and rows this arm changed."
                if stock else f"{len(v.corpus)} rows seeded. Shown: the gold rows.")
        if stock and not gold and not rows:
            hint = f"{len(v.corpus)} rows seeded. The bank holds no gold row for this question."
        out.append({"label": "Memory rows", "hint": hint, "items": rows})
    if v.stale_claim:
        out.append({"label": "Stale answer check", "hint": "Recorded beside the verdict, never scored.",
                    "items": [_item(stale_criterion(v), None)]})
    return out


def _arm(v: Variant, stock: Variant | None) -> dict:
    return {
        "arm": v.arm,
        "change": v.change if v.arm != "stock" else "",
        "answer": v.expected_answer,
        "answer_diff": diff(stock.expected_answer, v.expected_answer) if stock and v.arm == "counterfactual" else None,
        "notes": v.notes,
        "sections": _sections(v, stock),
        "probes": [dataclasses.asdict(p) for p in v.probes],
    }


def export() -> dict:
    by_case: dict[str, dict[str, Variant]] = {}
    for v in _variants():
        by_case.setdefault(v.case_id, {})[v.arm] = v
    cases = []
    for case_id, arms in by_case.items():
        template = next(iter(arms.values()))
        stock = stock_variant(template)
        c = stock.case
        cases.append({
            "id": case_id,
            "operation": OPERATION_OF[template.kind],
            "kind": template.kind,
            "category": c.category,
            "origin": getattr(c, "origin", ""),
            "prompt": getattr(c, "question", ""),
            "arms": [_arm(stock, None)] + [_arm(arms[a], stock) for a in ("wording", "counterfactual") if a in arms],
        })
    return {
        "fixture_set": FIXTURE_SET,
        "fixture_revision": revision(),
        "pinned_commit": PINNED_COMMIT,
        "generated": datetime.date.today().isoformat(),
        "cases": cases,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.write_text(json.dumps(export(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
