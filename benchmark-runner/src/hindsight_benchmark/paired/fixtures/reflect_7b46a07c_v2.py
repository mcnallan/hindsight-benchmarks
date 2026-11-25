"""Paired Reflect fixtures for the pinned 7b46a07c system evals.

Six stock cases, two arms each (``wording`` and ``counterfactual``). Every variant
carries the COMPLETE corpus its bank is seeded with: the stock corpus for its runner
(``corpus.build().facts`` for ``reflect_corpus``, ``sources.build().facts`` for
``reflect_sources``) with only that arm's edits applied. Fact ids never change; only
text does, so ``case.gold`` keeps pointing at the facts that answer the arm.

The stock source-priority corpus is built at import time (``sources`` computes a
date-relative exception window; we leave it alone).
"""

from __future__ import annotations

import dataclasses
import re

from hindsight_system_evals import corpus as hard_corpus
from hindsight_system_evals import sources
from hindsight_system_evals.corpus import HardFact, HardQuestion

from hindsight_benchmark.paired.schema import Probe, Variant

_HARD = hard_corpus.build()
_SOURCES = sources.build()
_HARD_Q = {q.id: q for q in _HARD.questions}
_SRC_Q = {q.id: q for q in _SOURCES.questions}
_MONTHS = hard_corpus._MONTHS

#: Ids each arm is allowed to differ on from the stock corpus (self-check only).
_CHANGED: dict[tuple[str, str], set[str]] = {}


def _edit(facts: list[HardFact], edits: dict[str, str]) -> list[HardFact]:
    """Replace the text of the named facts; ids, cluster, metadata and subject stay."""
    unknown = set(edits) - {f.id for f in facts}
    if unknown:
        raise RuntimeError(f"unknown fact ids: {sorted(unknown)}")
    return [dataclasses.replace(f, text=edits[f.id]) if f.id in edits else f for f in facts]


def _sub(facts: list[HardFact], pattern: str, repl: str, *, only: set[str] | None = None) -> list[HardFact]:
    """Regex-substitute text in every fact (or just ``only``); returns a new list."""
    out = []
    for f in facts:
        if (only is None or f.id in only) and re.search(pattern, f.text):
            out.append(dataclasses.replace(f, text=re.sub(pattern, repl, f.text)))
        else:
            out.append(f)
    return out


def _diff_ids(stock: list[HardFact], revised: list[HardFact]) -> set[str]:
    return {a.id for a, b in zip(stock, revised, strict=True) if a.text != b.text or a.metadata != b.metadata}


def _stock_ids(case_id: str) -> list[HardFact]:
    return list(_HARD.facts if case_id.startswith("hq-") else _SOURCES.facts)


def _register(case_id: str, arm: str, revised: list[HardFact]) -> tuple[HardFact, ...]:
    _CHANGED[(case_id, arm)] = _diff_ids(_stock_ids(case_id), revised)
    return tuple(revised)


# ── 1. hq-owner-latest ───────────────────────────────────────────────────────
_OWN = _HARD_Q["hq-owner-latest"]
_OWN_FACTS = list(_HARD.facts)

# wording: the service is renamed in the four handover facts (the filler never names it).
_own_w_facts = _sub(
    _OWN_FACTS, r"the ranking service", "the scoring service", only={f"own-00{i}" for i in range(1, 5)}
)
_own_w = dataclasses.replace(
    _OWN,
    question="Which team owns the scoring service at the moment?",
    ideal_query="scoring service ownership current team",
    answer_criteria=(
        "States that the core infra team currently owns the scoring service. Mentioning the earlier "
        "owners as history is fine; naming any of them as the current owner is not."
    ),
    must_not_claim="the platform team, the retrieval team or the search guild currently owns the scoring service",
)

# counterfactual: the final dated handover goes to a team that appears nowhere else, so the
# (system, owner) subject stays unique and the core infra team never held the ranking service.
_own_c_facts = _edit(
    _OWN_FACTS,
    {"own-004": "Ownership of the ranking service passed to the relevance engineering team in November 2024."},
)
_own_c = dataclasses.replace(
    _OWN,
    answer_criteria=(
        "States that the relevance engineering team currently owns the ranking service. Mentioning the "
        "earlier owners as history is fine; naming any of them as the current owner is not."
    ),
    must_not_claim=(
        "the core infra team, the platform team, the retrieval team or the search guild currently owns "
        "the ranking service"
    ),
)

# ── 2. hq-absent-2024 ────────────────────────────────────────────────────────
_ABS = _HARD_Q["hq-absent-2024"]
_ABS_FACTS = list(_HARD.facts)

_abs_w_facts = _edit(
    _ABS_FACTS,
    {
        f"abs-{i:03d}": f"The engineering team numbered {40 + i * 7} people at the end of {_MONTHS[i % 12]} 202{5 + i % 2}."
        for i in range(14)
    },
)
_abs_w = dataclasses.replace(
    _ABS,
    question="How large was the engineering team at the end of 2024?",
    ideal_query="engineering team size end of 2024",
    answer_criteria=(
        "Says the memory holds no engineering team size for 2024, rather than giving or estimating one. "
        "Citing 2025/2026 figures as context is fine only if it does not present one as the 2024 number."
    ),
    must_not_claim="a specific engineering team size for 2024",
)

# counterfactual: the first row now records the end-of-2024 figure (December 2024, 38). The
# other thirteen headcount rows are untouched, and December 2024 is not used by any of them.
_abs_c_facts = _edit(
    _ABS_FACTS,
    {"abs-000": "Engineering headcount reached 38 at the end of December 2024."},
)
_abs_c = dataclasses.replace(
    _ABS,
    gold=["abs-000"],
    answer_criteria=(
        "States that the engineering headcount at the end of 2024 was 38. Citing 2025/2026 figures as "
        "context is fine only if it does not present one as the 2024 number."
    ),
    must_not_claim="the memory holds no engineering headcount figure for 2024",
)

# ── 3. hq-scoped-2fa ─────────────────────────────────────────────────────────
_SCP = _HARD_Q["hq-scoped-2fa"]
_SCP_FACTS = list(_HARD.facts)

_scp_w_facts = _sub(
    _SCP_FACTS,
    r"^Two-factor authentication",
    "Multi-factor authentication",
    only={"scp-001", "scp-002", "scp-003"},
)
_scp_w = dataclasses.replace(
    _SCP,
    question="Is multi-factor authentication mandatory for our accounts?",
    ideal_query="multi-factor authentication mandatory optional region accounts",
    answer_criteria=(
        "Answers with the regional qualification rather than a flat yes or no: mandatory for EU "
        "accounts and optional for US accounts. Any accurate treatment of the UK is acceptable — "
        "the memory says UK accounts are mandatory FROM 2026, so stating it either as mandatory or "
        "as mandatory starting in 2026 is correct, and so is omitting the UK entirely."
    ),
    must_not_claim="multi-factor authentication is mandatory everywhere, with no regional distinction",
)

# counterfactual: the EU flips to optional, so the UK is the only mandatory region.
_scp_c_facts = _edit(
    _SCP_FACTS,
    {"scp-001": "Two-factor authentication remains optional for EU accounts."},
)
_scp_c = dataclasses.replace(
    _SCP,
    answer_criteria=(
        "Answers with the regional qualification rather than a flat yes or no: optional for EU "
        "accounts and optional for US accounts, with mandatory applying only to UK accounts. The memory "
        "says UK accounts are mandatory FROM 2026, so stating the UK either as mandatory or as mandatory "
        "starting in 2026 is correct."
    ),
    must_not_claim="two-factor authentication is mandatory for EU accounts",
)

# ── 4. sp-rotation ───────────────────────────────────────────────────────────
_ROT = _SRC_Q["sp-rotation"]
_ROT_FACTS = list(_SOURCES.facts)

_rot_w_facts = _edit(
    _ROT_FACTS,
    {
        "g-rotation": "Rotation of production service credentials happens every 90 days.",
        "c-rotation-decision": (
            "At the security audit on 12 February 2026 the head of security decided that rotation of "
            "production service credentials moves from every 90 days to every 30 days, effective "
            "immediately; the handbook has not been updated yet."
        ),
        "c-rotation-scope": "Staging credentials remain on a 90-day rotation cycle.",
    },
)
_rot_w = dataclasses.replace(
    _ROT,
    question="What is the current rotation interval for production service credentials?",
    ideal_query="current production credential rotation interval",
    answer_criteria=(
        "States that production credentials are now rotated every 30 days, following the decision "
        "taken at the 12 February 2026 security audit. Noting that the handbook still says 90 days, or "
        "that staging stays at 90 days, is correct; presenting 90 days as the current production "
        "interval is not."
    ),
)

# counterfactual: the dated decision moves rotation to 14 days (only c-rotation-decision states it);
# the handbook's 90 and the staging 90 stay.
_rot_c_facts = _edit(
    _ROT_FACTS,
    {
        "c-rotation-decision": (
            "At the security review on 9 March 2026 the security lead decided that credential rotation for "
            "production services moves from 90 days to 14 days, effective immediately; the handbook has not "
            "been updated yet."
        ),
    },
)
_rot_c = dataclasses.replace(
    _ROT,
    answer_criteria=(
        "States that production credentials are now rotated every 14 days, following the decision "
        "taken at the 9 March 2026 security review. Noting that the handbook still says 90 days, or "
        "that staging stays at 90 days, is correct; presenting 90 days as the current production "
        "interval is not."
    ),
    must_not_claim="production service credentials are currently rotated every 30 days or every 90 days",
)

# ── 5. sp-upgrade-owner ──────────────────────────────────────────────────────
_UPG = _SRC_Q["sp-upgrade-owner"]
_UPG_FACTS = list(_SOURCES.facts)

# wording: Farid is renamed Idris everywhere he is named (the volunteering fact, the decline, and
# every chatter row drawn from the name pool), so the person stays one person across the bank.
_upg_w_facts = _sub(_UPG_FACTS, r"\bFarid\b", "Idris")
_upg_w_facts = _edit(
    _upg_w_facts,
    {"c-upgrade-owner": "Idris volunteered to lead the PostgreSQL 16 upgrade for the payments database."},
)
_upg_w = dataclasses.replace(
    _UPG,
    question="Who volunteered to lead the PostgreSQL 16 upgrade for the payments database?",
    ideal_query="volunteer lead PostgreSQL 16 upgrade payments database",
    answer_criteria="Names Idris as the person who volunteered to lead the upgrade.",
    must_not_claim=(
        "someone other than Idris volunteered to lead the PostgreSQL 16 upgrade, or that the memory "
        "does not say who volunteered"
    ),
)

# counterfactual: Owen volunteers. Farid keeps his other (decline) fact and his chatter rows;
# no chatter template names a volunteer, so only c-upgrade-owner changes.
_upg_c_facts = _edit(
    _UPG_FACTS,
    {"c-upgrade-owner": "Owen volunteered to run the PostgreSQL 16 upgrade for the payments database."},
)
_upg_c = dataclasses.replace(
    _UPG,
    answer_criteria="Names Owen as the person who volunteered to run the upgrade.",
    must_not_claim=(
        "someone other than Owen, such as Farid, volunteered to run the PostgreSQL 16 upgrade for the "
        "payments database, or that the memory does not say who volunteered"
    ),
)

# ── 6. sp-postmortem ─────────────────────────────────────────────────────────
_PM = _SRC_Q["sp-postmortem"]
_PM_FACTS = list(_SOURCES.facts)

# wording: "incident commander" -> "incident lead" in the handbook sentence AND the chatter row
# that shares the phrase; "working days" -> "business days".
_pm_w_facts = _sub(_PM_FACTS, r"\bincident commander\b", "incident lead")
_pm_w_facts = _sub(_pm_w_facts, r"\bworking days\b", "business days", only={"g-postmortem"})
_pm_w = dataclasses.replace(
    _PM,
    question="Who writes the postmortem after an incident, and what is the deadline?",
    ideal_query="postmortem author deadline after incident",
    answer_criteria=(
        "States that the incident lead writes it and that it is due within five business days of "
        "the incident being resolved."
    ),
    must_not_claim=(
        "a postmortem owner other than the incident lead, or a deadline other than five business days"
    ),
)

# counterfactual: only the deadline changes (five -> ten working days); the owner, the guide
# provenance and the chatter are untouched, and no conversation fact restates the deadline.
_pm_c_facts = _edit(
    _PM_FACTS,
    {
        "g-postmortem": (
            "The incident commander writes the postmortem, and it is due within ten working days of the "
            "incident being resolved."
        )
    },
)
_pm_c = dataclasses.replace(
    _PM,
    answer_criteria=(
        "States that the incident commander writes it and that it is due within ten working days of "
        "the incident being resolved."
    ),
    must_not_claim=(
        "a postmortem owner other than the incident commander, or a deadline other than ten working "
        "days, such as five working days"
    ),
)


VARIANTS: tuple[Variant, ...] = (
    # ── 1. hq-owner-latest ───────────────────────────────────────────────────
    Variant(
        case_id="hq-owner-latest",
        arm="wording",
        kind="reflect_corpus",
        change='Renamed "the ranking service" to "the scoring service" in the four handover facts, the question, ideal_query, criteria and trap; same four owners, same dates, filler untouched.',
        stock_answer="the core infra team",
        expected_answer="the core infra team",
        case=_own_w,
        corpus=_register("hq-owner-latest", "wording", _own_w_facts),
        probes=(
            Probe(
                "correct",
                "The core infra team owns the scoring service now. Ownership passed to it in November 2024; "
                "before that it sat with the search guild (from April 2024), the retrieval team (from "
                "September 2023) and originally the platform team (March 2023).",
                True,
            ),
            Probe(
                "stock-style wrong (earlier owner as current)",
                "The scoring service is currently owned by the search guild, which took it over in April 2024.",
                False,
            ),
        ),
        notes="Filler facts never name the ranking service, so only the four own-00N facts change. Team names are shared with the filler and were left alone.",
    ),
    Variant(
        case_id="hq-owner-latest",
        arm="counterfactual",
        kind="reflect_corpus",
        change="The final dated handover (November 2024) goes to the relevance engineering team instead of the core infra team; the other three handovers and all filler are unchanged.",
        stock_answer="the core infra team",
        expected_answer="the relevance engineering team",
        case=_own_c,
        corpus=_register("hq-owner-latest", "counterfactual", _own_c_facts),
        stale_claim="the core infra team currently owns the ranking service",
        probes=(
            Probe(
                "correct (new final owner)",
                "The relevance engineering team owns the ranking service now; it took over in November 2024. "
                "Before that the search guild had it (April 2024), the retrieval team (September 2023) and "
                "the platform team (March 2023).",
                True,
                False,
            ),
            Probe(
                "stock answer (stale)",
                "The core infra team owns the ranking service. Ownership passed to it in November 2024, "
                "after the search guild held it from April 2024.",
                False,
                True,
            ),
            Probe(
                "earlier owner as current",
                "The ranking service is currently owned by the search guild, which took over in April 2024.",
                False,
                False,
            ),
        ),
        notes="New owner appears nowhere else in the corpus, so (system, owner) subjects stay unique. The core infra team still owns other systems in the filler, which is the bait for the stale answer.",
    ),
    # ── 2. hq-absent-2024 ────────────────────────────────────────────────────
    Variant(
        case_id="hq-absent-2024",
        arm="wording",
        kind="reflect_corpus",
        change='Reworded all 14 headcount facts ("The engineering team numbered N people at the end of ...") and the question ("How large was the engineering team at the end of 2024?"); numbers, months and years identical, still no 2024 row.',
        stock_answer="the memory holds no 2024 headcount figure",
        expected_answer="the memory holds no 2024 team-size figure",
        case=_abs_w,
        corpus=_register("hq-absent-2024", "wording", _abs_w_facts),
        probes=(
            Probe(
                "correct (absence)",
                "I don't have a team-size figure for the end of 2024. The engineering team records I can see "
                "cover 2025 and 2026 only, so I can't say how many people there were at the end of 2024.",
                True,
            ),
            Probe(
                "invented figure",
                "The engineering team numbered about 35 people at the end of 2024, extrapolating back from "
                "the growth seen through 2025.",
                False,
            ),
        ),
        notes="All headcount-cluster rows renamed together so the bait keeps the question's vocabulary. The stock corpus's two January 2025 rows (40 and 124) are preserved exactly.",
    ),
    Variant(
        case_id="hq-absent-2024",
        arm="counterfactual",
        kind="reflect_corpus",
        change='abs-000 changes from "reached 40 at the end of January 2025" to "reached 38 at the end of December 2024", so the bank now holds an end-of-2024 headcount.',
        stock_answer="the memory holds no 2024 headcount figure",
        expected_answer="38",
        case=_abs_c,
        corpus=_register("hq-absent-2024", "counterfactual", _abs_c_facts),
        stale_claim="the memory holds no engineering headcount figure for 2024",
        probes=(
            Probe(
                "correct (38)",
                "Engineering headcount was 38 at the end of 2024 (December 2024). The 2025 and 2026 rows "
                "show it growing from there.",
                True,
                False,
            ),
            Probe(
                "stock answer (absence)",
                "The memory holds no engineering headcount figure for the end of 2024; the records only "
                "cover 2025 and 2026.",
                False,
                True,
            ),
            Probe(
                "wrong number",
                "Engineering headcount at the end of 2024 was 40.",
                False,
                False,
            ),
        ),
        notes="One fact changed. 38 is not used elsewhere in the corpus; December 2024 is not used by any other headcount row. The criterion mirrors the stock context allowance for 2025/2026 figures. Gold changes from [] to [abs-000].",
    ),
    # ── 3. hq-scoped-2fa ─────────────────────────────────────────────────────
    Variant(
        case_id="hq-scoped-2fa",
        arm="wording",
        kind="reflect_corpus",
        change='"Two-factor authentication" renamed "Multi-factor authentication" in the three regional facts, the question, ideal_query, criteria and trap; regions and mandatory/optional values unchanged (SSO and password-rotation near-misses untouched).',
        stock_answer="mandatory for EU, optional for US (UK mandatory from 2026)",
        expected_answer="mandatory for EU, optional for US (UK mandatory from 2026)",
        case=_scp_w,
        corpus=_register("hq-scoped-2fa", "wording", _scp_w_facts),
        probes=(
            Probe(
                "correct (regional split)",
                "It depends on the region. Multi-factor authentication is mandatory for EU accounts and, from "
                "2026, for UK accounts, but it remains optional for US accounts.",
                True,
            ),
            Probe(
                "flat yes (the trap)",
                "Yes, multi-factor authentication is mandatory for all of our accounts.",
                False,
            ),
        ),
        notes="The near-misses (SSO, password rotation) do not share the entity name, so they stay as stock.",
    ),
    Variant(
        case_id="hq-scoped-2fa",
        arm="counterfactual",
        kind="reflect_corpus",
        change='scp-001 changes from "mandatory for all EU accounts" to "remains optional for EU accounts", so the UK is the only mandatory region.',
        stock_answer="mandatory for EU, optional for US (UK mandatory from 2026)",
        expected_answer="optional for EU and US, mandatory only for the UK (from 2026)",
        case=_scp_c,
        corpus=_register("hq-scoped-2fa", "counterfactual", _scp_c_facts),
        stale_claim="two-factor authentication is mandatory for EU accounts",
        probes=(
            Probe(
                "correct (new split)",
                "Not across the board. Two-factor authentication is optional for EU and US accounts; the "
                "only region where it is mandatory is the UK, from 2026.",
                True,
                False,
            ),
            Probe(
                "stock split (stale)",
                "It depends on the region: two-factor authentication is mandatory for EU accounts, optional "
                "for US accounts, and mandatory for UK accounts from 2026.",
                False,
                True,
            ),
            Probe(
                "flat yes",
                "Yes, two-factor authentication is mandatory for all of our accounts.",
                False,
                # "Mandatory for all" includes the EU, so it also repeats the stock EU claim.
                True,
            ),
        ),
        notes="One fact changed. The stock criterion tolerated omitting the UK; here the UK is the only mandatory region so the answer must state it. scp-001 and scp-002 stay distinct texts.",
    ),
    # ── 4. sp-rotation ───────────────────────────────────────────────────────
    Variant(
        case_id="sp-rotation",
        arm="wording",
        kind="reflect_sources",
        change='Reworded the three rotation facts (handbook, dated decision, staging note), changed the decision from "9 March 2026 security review/security lead" to "12 February 2026 security audit/head of security", and reworded the question; the 90 to 30 day values are unchanged.',
        stock_answer="30 days",
        expected_answer="30 days",
        case=_rot_w,
        corpus=_register("sp-rotation", "wording", _rot_w_facts),
        probes=(
            Probe(
                "correct (30 days)",
                "Production service credentials are now rotated every 30 days. The handbook still says 90 "
                "days, but the head of security decided at the 12 February 2026 security audit to move to 30. "
                "Staging credentials remain on 90 days.",
                True,
            ),
            Probe(
                "handbook-only (the trap)",
                "Production service credentials are rotated every 90 days, according to the handbook.",
                False,
            ),
        ),
        notes='"credential"/"rotation" vocabulary is kept in the facts because the chatter rows share it (renaming only the gold would make retrieval easier). The security-audit date is within the past and does not coincide with any chatter-specific claim.',
    ),
    Variant(
        case_id="sp-rotation",
        arm="counterfactual",
        kind="reflect_sources",
        change='The dated decision in c-rotation-decision now moves rotation "from 90 days to 14 days" (was 30); the handbook (90), the staging note (90), the date and the speaker are unchanged.',
        stock_answer="30 days",
        expected_answer="14 days",
        case=_rot_c,
        corpus=_register("sp-rotation", "counterfactual", _rot_c_facts),
        stale_claim="production service credentials are currently rotated every 30 days",
        probes=(
            Probe(
                "correct (14 days)",
                "Production service credentials are now rotated every 14 days, following the security lead's "
                "decision at the 9 March 2026 security review. The handbook still says 90 days and has not "
                "been updated, and staging stays on 90 days.",
                True,
                False,
            ),
            Probe(
                "stock answer (30 days, stale)",
                "Production service credentials are rotated every 30 days, per the decision at the 9 March "
                "2026 security review; the handbook still says 90.",
                False,
                True,
            ),
            Probe(
                "handbook-only (90 days)",
                "Production service credentials are rotated every 90 days, per the handbook.",
                False,
                False,
            ),
        ),
        notes="Only c-rotation-decision states the new interval, so one fact changes. The trap names both 30 and 90 days because both are wrong now; the stale_claim is the stock answer only.",
    ),
    # ── 5. sp-upgrade-owner ──────────────────────────────────────────────────
    Variant(
        case_id="sp-upgrade-owner",
        arm="wording",
        kind="reflect_sources",
        change='Farid renamed Idris in c-upgrade-owner, c-upgrade-decline and every chatter row drawn from the name pool; "run" became "lead" in the volunteering fact, question, ideal_query, criteria and trap.',
        stock_answer="Farid",
        expected_answer="Idris",
        case=_upg_w,
        corpus=_register("sp-upgrade-owner", "wording", _upg_w_facts),
        probes=(
            Probe(
                "correct (Idris)",
                "Idris volunteered to lead the PostgreSQL 16 upgrade for the payments database. Lena offered "
                "to review the upgrade plan.",
                True,
            ),
            Probe(
                "reviewer instead of volunteer",
                "Lena volunteered to lead the PostgreSQL 16 upgrade for the payments database.",
                False,
            ),
        ),
        notes="The chatter pool is renamed too so Idris stays one person across the bank (Farid would otherwise survive as chatter-only). Texts stay unique because Idris is not in the stock pool.",
    ),
    Variant(
        case_id="sp-upgrade-owner",
        arm="counterfactual",
        kind="reflect_sources",
        change="c-upgrade-owner now says Owen (not Farid) volunteered to run the PostgreSQL 16 upgrade; Farid keeps his unrelated decline fact and chatter rows, and Lena still offers to review.",
        stock_answer="Farid",
        expected_answer="Owen",
        case=_upg_c,
        corpus=_register("sp-upgrade-owner", "counterfactual", _upg_c_facts),
        stale_claim="Farid volunteered to run the PostgreSQL 16 upgrade for the payments database",
        probes=(
            Probe(
                "correct (Owen)",
                "Owen volunteered to run the PostgreSQL 16 upgrade for the payments database; Lena offered "
                "to review the plan.",
                True,
                False,
            ),
            Probe(
                "stock answer (Farid, stale)",
                "Farid volunteered to run the PostgreSQL 16 upgrade for the payments database.",
                False,
                True,
            ),
            Probe(
                "memory silent",
                "The memory does not say who volunteered to run the PostgreSQL 16 upgrade.",
                False,
                False,
            ),
        ),
        notes="No chatter template hard-codes a volunteer (only {name}), so only c-upgrade-owner changes. Owen is in the chatter name pool, same as Farid was, so difficulty is preserved. c-upgrade-decline (Farid, search reindex) stays as a distractor.",
    ),
    # ── 6. sp-postmortem ─────────────────────────────────────────────────────
    Variant(
        case_id="sp-postmortem",
        arm="wording",
        kind="reflect_sources",
        change='"incident commander" became "incident lead" in the handbook sentence and the chatter rows that share the phrase, "working days" became "business days", and the question was reworded; owner and the five-day deadline are unchanged.',
        stock_answer="incident commander, five working days",
        expected_answer="incident lead, five business days",
        case=_pm_w,
        corpus=_register("sp-postmortem", "wording", _pm_w_facts),
        probes=(
            Probe(
                "correct",
                "The incident lead writes the postmortem, and it is due within five business days of the "
                "incident being resolved.",
                True,
            ),
            Probe(
                "wrong deadline",
                "The incident lead writes the postmortem, and it is due within ten business days of the "
                "incident being resolved.",
                False,
            ),
        ),
        notes="Source structure intact: the handbook fact is still the only guide fact that answers it, the chatter never restated owner or deadline. The chatter row that says someone had been \"incident lead twice this month\" is renamed so it keeps sharing vocabulary with the gold.",
    ),
    Variant(
        case_id="sp-postmortem",
        arm="counterfactual",
        kind="reflect_sources",
        change='g-postmortem now says the postmortem is due within ten working days (was five); the owner (incident commander), guide provenance and all chatter are unchanged.',
        stock_answer="incident commander, five working days",
        expected_answer="incident commander, ten working days",
        case=_pm_c,
        corpus=_register("sp-postmortem", "counterfactual", _pm_c_facts),
        stale_claim="the postmortem is due within five working days of the incident being resolved",
        probes=(
            Probe(
                "correct (ten days)",
                "The incident commander writes the postmortem, and it is due within ten working days of the "
                "incident being resolved.",
                True,
                False,
            ),
            Probe(
                "stock answer (five days, stale)",
                "The incident commander writes the postmortem, and it is due within five working days of "
                "the incident being resolved.",
                False,
                True,
            ),
            Probe(
                "wrong owner",
                "The on-call lead writes the postmortem, due within ten working days of the incident being "
                "resolved.",
                False,
                False,
            ),
        ),
        notes="Only one attribute (deadline) changes. No conversation fact restates owner or deadline (checked chatter templates), so the handbook remains the sole authority.",
    ),
)


def _self_check() -> None:
    stock_by_case = {
        "hq-owner-latest": _HARD_Q,
        "hq-absent-2024": _HARD_Q,
        "hq-scoped-2fa": _HARD_Q,
        "sp-rotation": _SRC_Q,
        "sp-upgrade-owner": _SRC_Q,
        "sp-postmortem": _SRC_Q,
    }
    order = list(stock_by_case)
    assert len(VARIANTS) == 12, len(VARIANTS)
    for i, case_id in enumerate(order):
        w, c = VARIANTS[2 * i], VARIANTS[2 * i + 1]
        assert (w.case_id, w.arm) == (case_id, "wording"), (w.case_id, w.arm)
        assert (c.case_id, c.arm) == (case_id, "counterfactual"), (c.case_id, c.arm)

    # What each arm's gold facts must (and must not) contain, per (case, arm).
    present = {
        ("hq-owner-latest", "wording"): ["scoring service", "core infra team"],
        ("hq-owner-latest", "counterfactual"): ["relevance engineering team"],
        ("hq-absent-2024", "counterfactual"): ["38", "December 2024"],
        ("hq-scoped-2fa", "wording"): ["Multi-factor authentication"],
        ("hq-scoped-2fa", "counterfactual"): ["optional for EU"],
        ("sp-rotation", "wording"): ["30 days", "90 days"],
        ("sp-rotation", "counterfactual"): ["14 days", "90 days"],
        ("sp-upgrade-owner", "wording"): ["Idris"],
        ("sp-upgrade-owner", "counterfactual"): ["Owen"],
        ("sp-postmortem", "wording"): ["incident lead", "five business days"],
        ("sp-postmortem", "counterfactual"): ["incident commander", "ten working days"],
    }
    absent = {
        ("hq-owner-latest", "wording"): ["ranking service"],
        ("hq-owner-latest", "counterfactual"): ["core infra team"],
        ("hq-absent-2024", "counterfactual"): ["January 2025"],
        ("hq-scoped-2fa", "wording"): ["Two-factor"],
        ("hq-scoped-2fa", "counterfactual"): ["mandatory for all EU"],
        ("sp-rotation", "wording"): ["9 March"],
        ("sp-rotation", "counterfactual"): ["30 days"],
        ("sp-upgrade-owner", "wording"): ["Farid"],
        ("sp-upgrade-owner", "counterfactual"): ["Farid"],
        ("sp-postmortem", "wording"): ["incident commander", "working days"],
        ("sp-postmortem", "counterfactual"): ["five working days"],
    }

    for v in VARIANTS:
        key = (v.case_id, v.arm)
        stock_q = stock_by_case[v.case_id][v.case_id]
        stock_facts = _stock_ids(v.case_id)
        assert v.case.id == v.case_id == stock_q.id, key
        assert isinstance(v.case, HardQuestion), key
        corpus_facts = list(v.corpus)
        ids = [f.id for f in corpus_facts]
        assert ids == [f.id for f in stock_facts], f"{key}: ids/order differ from stock"
        assert len({f.text for f in corpus_facts}) == len(corpus_facts), f"{key}: duplicate fact text"
        assert all(f.cluster == s.cluster and f.metadata == s.metadata for f, s in zip(corpus_facts, stock_facts)), (
            f"{key}: cluster/metadata drift"
        )
        if v.kind == "reflect_corpus":
            hard_corpus._assert_subjects_are_unique(corpus_facts)
        assert v.kind == ("reflect_corpus" if v.case_id.startswith("hq-") else "reflect_sources"), key
        assert set(v.case.gold) <= set(ids), f"{key}: gold id missing"
        assert _diff_ids(stock_facts, corpus_facts) == _CHANGED[key]
        assert _CHANGED[key], f"{key}: no facts changed"
        assert v.case.question and v.case.ideal_query and v.case.answer_criteria and v.case.must_not_claim, key
        assert not v.extra, key

        gold_text = " ".join(f.text for f in corpus_facts if f.id in set(v.case.gold))
        if key == ("hq-owner-latest", "counterfactual"):
            gold_text = next(f.text for f in corpus_facts if f.id == "own-004")
        if key == ("hq-absent-2024", "counterfactual"):
            gold_text = next(f.text for f in corpus_facts if f.id == "abs-000")
        if key == ("hq-owner-latest", "wording"):
            gold_text = " ".join(f.text for f in corpus_facts if f.id.startswith("own-00"))
        if key == ("hq-scoped-2fa", "counterfactual"):
            gold_text = next(f.text for f in corpus_facts if f.id == "scp-001")
        if key == ("hq-scoped-2fa", "wording"):
            gold_text = next(f.text for f in corpus_facts if f.id == "scp-001")
        for needle in present.get(key, []):
            assert needle in gold_text, f"{key}: {needle!r} missing from gold facts"
        for needle in absent.get(key, []):
            assert needle not in gold_text, f"{key}: {needle!r} still in gold facts"
        if key == ("sp-rotation", "wording"):
            assert "9 March" not in " ".join(f.text for f in corpus_facts if f.id.startswith("c-rotation"))
        if key == ("sp-upgrade-owner", "wording"):
            assert not any("Farid" in f.text for f in corpus_facts), "Farid survived the rename"
        if key == ("sp-postmortem", "wording"):
            assert not any("incident commander" in f.text for f in corpus_facts), "incident commander survived"

        assert v.arm in ("wording", "counterfactual")
        assert v.change and v.stock_answer and v.expected_answer and v.notes, key
        passes = [p for p in v.probes if p.expect_pass]
        fails = [p for p in v.probes if not p.expect_pass]
        assert len(passes) >= 1 and len(fails) >= 1, f"{key}: need a passing and a failing probe"
        assert all(p.response.strip() and p.label for p in v.probes), key
        if v.arm == "counterfactual":
            assert v.expected_answer != v.stock_answer, key
            assert v.stale_claim, key
            assert all(p.expect_stale is not None for p in v.probes), key
            assert all(p.expect_stale is False for p in passes), key
            assert any(p.expect_stale is True and not p.expect_pass for p in v.probes), key
            assert v.case.must_not_claim != stock_q.must_not_claim, f"{key}: trap not rewritten"
            assert v.case.answer_criteria != stock_q.answer_criteria, f"{key}: criteria not rewritten"
        else:
            assert not v.stale_claim, key
            assert len(fails) >= 1
            assert v.case.question != stock_q.question, f"{key}: question not reworded"

    # Cross-checks that the criteria/trap/probes agree with the revised facts.
    cf = {v.case_id: v for v in VARIANTS if v.arm == "counterfactual"}
    assert "relevance engineering" in cf["hq-owner-latest"].case.answer_criteria
    assert "core infra team" in cf["hq-owner-latest"].case.must_not_claim
    assert "38" in cf["hq-absent-2024"].case.answer_criteria and cf["hq-absent-2024"].case.gold == ["abs-000"]
    assert "14 days" in cf["sp-rotation"].case.answer_criteria and "30 days" in cf["sp-rotation"].case.must_not_claim
    assert "Owen" in cf["sp-upgrade-owner"].case.answer_criteria and "Farid" in cf["sp-upgrade-owner"].case.must_not_claim
    assert "ten working days" in cf["sp-postmortem"].case.answer_criteria
    assert "five working days" in cf["sp-postmortem"].case.must_not_claim
    # The three corpus-case counterfactuals each change exactly one fact; the sources ones too.
    for case_id, v in cf.items():
        assert len(_CHANGED[(case_id, "counterfactual")]) == 1, case_id
    print(f"ok: {len(VARIANTS)} variants")
    for (case_id, arm), ids in _CHANGED.items():
        print(f"  {case_id:18s} {arm:15s} changed {len(ids)} fact(s)")


if __name__ == "__main__":
    _self_check()
