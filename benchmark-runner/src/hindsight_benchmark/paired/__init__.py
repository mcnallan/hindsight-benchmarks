"""Paired robustness fixtures for Hindsight's system evals (stock, wording, counterfactual).

Fixtures are versioned by the pinned upstream commit and a revision tag, so a
result always names exactly which inputs produced it. ``revision()`` hashes the
fixture sources; the runner records it with each job and refuses to resume a job
whose fixtures changed underneath it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_COMMIT = "7b46a07c82f997903230f0fb33deafe9dfd2e22b"
REVISION_TAG = "v2"
FIXTURE_SET = f"{PINNED_COMMIT[:8]}-{REVISION_TAG}"

HERE = Path(__file__).resolve().parent
#: Every file whose content decides what a paired run feeds the model or the judge.
FIXTURE_FILES = (
    HERE / "schema.py",
    HERE / "fixtures" / f"retain_{PINNED_COMMIT[:8]}_{REVISION_TAG}.py",
    HERE / "fixtures" / f"reflect_{PINNED_COMMIT[:8]}_{REVISION_TAG}.py",
    HERE / "runner.py",
    HERE / "test_paired_retain.py",
    HERE / "test_paired_reflect.py",
)


def revision() -> str:
    """``<fixture set>+<sha256 prefix>`` over the fixture sources, stable across machines."""
    digest = hashlib.sha256()
    for path in FIXTURE_FILES:
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return f"{FIXTURE_SET}+{digest.hexdigest()[:12]}"
