"""The shape every paired-robustness fixture takes.

A paired case is one stock case run three ways in fresh banks:

* ``stock`` — the upstream case exactly as pinned;
* ``wording`` — names, identifiers or phrasing changed, relationships and correct
  answer preserved;
* ``counterfactual`` — one fact changed so the correct answer changes, with the
  traps rewritten to reject the stock answer.

These modules import from the pinned ``hindsight-system-evals`` checkout, so they
only load inside that venv (the runner puts ``src/`` and the checkout's
``hindsight-system-evals`` directory on ``PYTHONPATH``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Arm = Literal["stock", "wording", "counterfactual"]
VARIANT_ARMS: tuple[Arm, ...] = ("wording", "counterfactual")
ARMS: tuple[Arm, ...] = ("stock", *VARIANT_ARMS)

#: Which stock runner grades the case.
#:
#: * ``retain_capability`` — ``retain_capabilities.run_case`` on a ``Case``;
#: * ``retain_fidelity`` — ``test_retain._run_fidelity`` on a ``FidelityCase``;
#: * ``reflect_corpus`` — ``test_reflect._run`` over the hard corpus (``corpus.build()``);
#: * ``reflect_sources`` — ``test_reflect._run_sources`` over the handbook/chatter corpus.
Kind = Literal["retain_capability", "retain_fidelity", "reflect_corpus", "reflect_sources"]


@dataclass(frozen=True)
class Probe:
    """A canned output the stock grader must classify a fixed way before any model runs.

    ``response`` uses the format the grader sees for that kind: rendered stored facts
    for retain (``- [world] Tamsin, the user's roommate, ...``, one per line, the shape
    ``StoredFact.render`` produces), or reflect's answer text.

    ``expect_pass`` is the WHOLE-CASE verdict under the stock grading order: no trap
    met, then every ``must`` (retain capability), ``correct`` (retain fidelity) or
    ``answer_criteria`` (reflect) met. ``expect_stale`` is whether the variant's
    ``stale_claim`` should be met; leave it ``None`` where it is not meaningful.
    """

    label: str
    response: str
    expect_pass: bool
    expect_stale: bool | None = None


@dataclass(frozen=True)
class Variant:
    """One arm of one stock case, keyed to the pinned commit by ``case_id``."""

    #: The stock case id, exactly as upstream names it (``hq-owner-latest``).
    case_id: str
    arm: Arm
    kind: Kind
    #: One or two sentences: what changed. For a counterfactual, the single fact
    #: that changed and why the answer changes with it.
    change: str
    #: The stock case's correct answer, in a few words.
    stock_answer: str
    #: This arm's correct answer, in a few words. Equals ``stock_answer`` in meaning
    #: for a wording arm (with renamed entities); differs for a counterfactual.
    expected_answer: str
    #: The revised case object, the same type the stock runner takes: a
    #: ``retain_capabilities.Case``, a ``test_retain.FidelityCase`` or a
    #: ``corpus.HardQuestion``. Its ``id`` stays the stock id; the runner suffixes it.
    case: object
    #: Reflect only: the COMPLETE corpus the bank is seeded with, as
    #: ``corpus.HardFact`` rows (stock corpus with this arm's edits applied). Empty
    #: for retain. Gold ids in ``case.gold`` must point into it.
    corpus: tuple = ()
    #: A claim, phrased like a trap, that is met when an output repeats the stock
    #: answer despite the changed evidence. Required for a counterfactual; empty
    #: for a wording arm. Recorded beside the verdict, never scored.
    stale_claim: str = ""
    #: At least: the arm's correct answer (expect_pass=True) and the stock answer
    #: (expect_pass=False, expect_stale=True for a counterfactual). A wording arm
    #: also probes a wrong answer in the renamed vocabulary.
    probes: tuple[Probe, ...] = ()
    #: Free-form notes for the reviewer (what was kept fixed on purpose, risks).
    notes: str = ""
    #: Anything the runner should set on the bank beyond the stock case (rare).
    extra: dict = field(default_factory=dict)
