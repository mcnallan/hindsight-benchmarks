"""Paired fixtures for six Retain cases, pinned to hindsight-system-evals 7b46a07c.

Cases, in order: coreference-roles, multi-speaker-attribution,
context-names-the-speaker, in-text-correction (``retain_capability``);
date-borrowed-from-nearby-subject, owner-borrowed-from-nearby-subject
(``retain_fidelity``).

Every arm is built from the upstream object with ``dataclasses.replace``, so a field
not named here is identical to upstream. Probes for ``retain_capability`` use
``StoredFact.render`` format; probes for ``retain_fidelity`` use the
``_extracted_facts`` format (``- <text>`` with an optional ``  [occurred_start=...]``).
"""

from __future__ import annotations

from dataclasses import replace

from evals.retain.test_retain import FIDELITY_CASES, FidelityCase
from hindsight_system_evals.retain_capabilities import CASES, Case, Item

from hindsight_benchmark.paired.schema import Probe, Variant

_CAPS: dict[str, Case] = {case.id: case for case in CASES}
_FIDELITY: dict[str, FidelityCase] = {case.id: case for case in FIDELITY_CASES}


def _cap(case_id: str, **changes: object) -> Case:
    return replace(_CAPS[case_id], **changes)


def _fid(case_id: str, **changes: object) -> FidelityCase:
    return replace(_FIDELITY[case_id], **changes)


# ── 1. coreference-roles ─────────────────────────────────────────────────────

_COREF_WORDING = _cap(
    "coreference-roles",
    items=(
        Item(
            "My flatmate is leaving next month. Her name is Odette, and she got a job in Bristol. "
            "My supervisor, Anil, said I can work from home while I look for a new place."
        ),
    ),
    must=(
        "A fact names Odette and identifies her as the user's flatmate.",
        "A fact records that Odette got a job in Bristol or is leaving.",
        "A fact names Anil and identifies him as the user's supervisor.",
    ),
    traps=(
        "A fact line mentions 'my flatmate', 'she' or 'my supervisor' and nowhere on that whole line names the person.",
    ),
)

# One relationship moves: the manager, not the roommate, is the one who got the job in Leeds.
_COREF_CF = _cap(
    "coreference-roles",
    items=(
        Item(
            "My roommate is moving out next month. Her name is Tamsin. "
            "My manager, Ravi, said I can work remotely while I look for a new place. "
            "He got a job in Leeds."
        ),
    ),
    must=(
        "A fact names Tamsin and identifies her as the user's roommate.",
        "A fact records that Tamsin is moving out next month.",
        # v2: split in two, as the stock case splits name+role from the event. v1 demanded one fact
        # with all three, which failed correct extractions that stored the role and the job apart.
        "A fact names Ravi and identifies him as the user's manager.",
        "A fact records that Ravi got a job in Leeds.",
    ),
    traps=(
        "A fact line mentions 'my roommate', 'she', 'he' or 'my manager' and nowhere on that whole line names the person.",
        "A fact states that Tamsin got a job in Leeds.",
    ),
)

# ── 2. multi-speaker-attribution ─────────────────────────────────────────────

_SPEAKER_WORDING = _cap(
    "multi-speaker-attribution",
    items=(
        Item(
            "Mei: I'll take the schema upgrade, I've done three of these before.\n"
            "Jonas: Then I'll handle the failover checklist. I'm out on Friday though.\n"
            "Mei: Fine, I'm allergic to on-call weekends anyway, so let's ship Thursday.",
            context="platform team daily sync",
        ),
    ),
    must=(
        "A fact records that Mei will take the schema upgrade.",
        "A fact records that Jonas will handle the failover checklist.",
        "A fact records that Jonas is out on Friday.",
    ),
    traps=("A fact attributes the upgrade to Jonas, or the checklist or Friday off to Mei.",),
)

# One relationship moves: Priya, not Tom, is the one off on Friday.
_SPEAKER_CF = _cap(
    "multi-speaker-attribution",
    items=(
        Item(
            "Priya: I'll take the database migration, I've done three of these before.\n"
            "Tom: Then I'll handle the rollback runbook.\n"
            "Priya: Fine, I'm off on Friday and I'm allergic to on-call weekends anyway, so let's ship Thursday.",
            context="engineering stand-up",
        ),
    ),
    must=(
        "A fact records that Priya will take the database migration.",
        "A fact records that Tom will handle the rollback runbook.",
        "A fact records that Priya is off on Friday.",
    ),
    traps=("A fact attributes the migration to Tom, the runbook to Priya, or Friday off to Tom.",),
)

# ── 3. context-names-the-speaker ─────────────────────────────────────────────

_CONTEXT_WORDING = _cap(
    "context-names-the-speaker",
    items=(
        Item(
            "I've been a subscriber for three years. I run a pastry shop in Coimbra and I need the bills in "
            "Portuguese from now on.",
            context="transcript of a customer service call; the first-person speaker is the customer, Marta Seixas",
        ),
    ),
    must=(
        "A fact attributes running a pastry shop in Coimbra to Marta Seixas.",
        "A fact records that Marta Seixas needs bills in Portuguese.",
    ),
    traps=("A fact attributes the pastry shop or the bills request to the assistant or agent itself.",),
)

# One fact moves: the context names a different customer as the first-person speaker.
_CONTEXT_CF = _cap(
    "context-names-the-speaker",
    items=(
        Item(
            "I've been a customer for three years. I run a bakery in Porto and I need the invoices in "
            "Portuguese from now on.",
            context="transcript of a support call; the first-person speaker is the customer, Rui Matos",
        ),
    ),
    must=(
        "A fact attributes running a bakery in Porto to Rui Matos.",
        "A fact records that Rui Matos needs invoices in Portuguese.",
    ),
    traps=(
        "A fact attributes the bakery or the invoice request to the assistant or agent itself.",
        "A fact attributes the bakery or the invoice request to Inês Duarte.",
    ),
)

# ── 4. in-text-correction ────────────────────────────────────────────────────
# Item timestamp stays 2026-03-02 (a Monday): Tuesday = 3 March, Wednesday = 4 March,
# Thursday = 5 March.

_CORRECTION_WORDING = _cap(
    "in-text-correction",
    items=(
        Item(
            "The budget review is on Tuesday — apologies, I meant Wednesday. It's in room 7C with the whole "
            "infrastructure team.",
            timestamp="2026-03-02T09:00:00",
        ),
    ),
    must=("A fact records that the budget review is on Wednesday (4 March 2026).",),
    traps=("A fact states that the budget review is on Tuesday.",),
)

# One fact moves: the speaker corrects to Thursday, not Wednesday.
_CORRECTION_CF = _cap(
    "in-text-correction",
    items=(
        Item(
            "The design review is on Tuesday — sorry, I mean Thursday. It's in room 4B with the whole "
            "platform team.",
            timestamp="2026-03-02T09:00:00",
        ),
    ),
    must=("A fact records that the design review is on Thursday (5 March 2026).",),
    traps=(
        "A fact states that the design review is on Tuesday.",
        "A fact states that the design review is on Wednesday.",
    ),
)

# ── 5. date-borrowed-from-nearby-subject ─────────────────────────────────────

_DATE_WORDING = _fid(
    "date-borrowed-from-nearby-subject",
    document=(
        "Survey planning note.\n\n"
        "The programme has a target start of October 6.\n"
        "Resident notifications must be completed before fieldwork begins.\n"
        "The equipment check finished last quarter."
    ),
    must_extract="resident notifications have to be completed before fieldwork begins",
    subject="the resident notifications",
    must_not_carry="carries a date or deadline of its own, such as October 6",
)

# One fact moves: the notifications now have their own explicit deadline, which is not the project date.
_DATE_CF = _fid(
    "date-borrowed-from-nearby-subject",
    document=(
        "Fieldwork planning note.\n\n"
        "The project has a target start of September 14.\n"
        "Customer notifications must be completed by August 28, before fieldwork begins.\n"
        "The equipment audit finished last quarter."
    ),
    must_extract="customer notifications have to be completed by August 28, before fieldwork begins",
    subject="the customer notifications",
    must_not_carry="gives September 14 as the date or deadline of the notifications",
)

# ── 6. owner-borrowed-from-nearby-subject ────────────────────────────────────

_OWNER_WORDING = _fid(
    "owner-borrowed-from-nearby-subject",
    document=(
        "Release status.\n\n"
        "Amara owns the index rebuild and signed off on the cutover plan.\n"
        "The worker-thread cap still needs to be raised before go-live.\n"
        "Both items are tracked in the release checklist."
    ),
    must_extract="the worker-thread cap still needs to be raised before go-live",
    subject="raising the worker-thread cap",
    must_not_carry="names a person as its owner, assignee, or the one responsible, such as Amara",
)

# One fact moves: the pool-limit item now has its own named owner, who is not Priya.
_OWNER_CF = _fid(
    "owner-borrowed-from-nearby-subject",
    document=(
        "Migration status.\n\n"
        "Priya owns the schema migration and signed off on the rollback plan.\n"
        "Dmitri owns raising the connection-pool limit, which still needs to happen before launch.\n"
        "Both items are tracked in the launch checklist."
    ),
    must_extract="Dmitri owns raising the connection-pool limit, which still needs to happen before launch",
    subject="raising the connection-pool limit",
    must_not_carry="names Priya as its owner, assignee, or the one responsible",
)


VARIANTS: tuple[Variant, ...] = (
    # 1 ── coreference-roles
    Variant(
        case_id="coreference-roles",
        arm="wording",
        kind="retain_capability",
        change="Roommate/Tamsin/Leeds/manager/Ravi renamed to flatmate/Odette/Bristol/supervisor/Anil with rephrased verbs; all relationships kept.",
        stock_answer="Tamsin is the user's roommate and got a job in Leeds; Ravi is the user's manager.",
        expected_answer="Odette is the user's flatmate and got a job in Bristol; Anil is the user's supervisor.",
        case=_COREF_WORDING,
        probes=(
            Probe(
                "correct, names resolved",
                "- [world] Odette, the user's flatmate, is leaving next month and got a job in Bristol  {entities: Odette, user}\n"
                "- [world] Anil, the user's supervisor, said the user can work from home while looking for a new place  {entities: Anil, user}",
                True,
            ),
            Probe(
                "wrong: unresolved 'my flatmate' / 'she'",
                "- [world] The user's flatmate is leaving next month and she got a job in Bristol  {entities: user}\n"
                "- [world] Anil, the user's supervisor, said the user can work from home while looking for a new place  {entities: Anil, user}",
                False,
            ),
        ),
        notes="Same three relationships, same pronoun-resolution difficulty (she, my flatmate, my supervisor).",
    ),
    Variant(
        case_id="coreference-roles",
        arm="counterfactual",
        kind="retain_capability",
        change="The job in Leeds now belongs to the manager, Ravi (stock: the roommate Tamsin); everything else unchanged.",
        stock_answer="Tamsin (roommate) got a job in Leeds.",
        expected_answer="Ravi (manager) got a job in Leeds; Tamsin is only moving out.",
        case=_COREF_CF,
        stale_claim="A fact states that Tamsin got a job in Leeds.",
        probes=(
            Probe(
                "correct: Ravi got the Leeds job",
                "- [world] Tamsin, the user's roommate, is moving out next month  {entities: Tamsin, user}\n"
                "- [world] Ravi, the user's manager, got a job in Leeds and said the user can work remotely while looking for a new place  {entities: Ravi, Leeds, user}",
                True,
                False,
            ),
            Probe(
                "correct: role and job stored as separate facts (qwen3.5-4b v1 pilot output)",
                "- [world] Ravi got a job in Leeds.  {entities: Ravi, Leeds}\n"
                "- [world] User's manager Ravi said the user can work remotely while looking for a new place.  {entities: user, Ravi}\n"
                "- [world] User's roommate named Tamsin is moving out next month.  {entities: user, Tamsin}",
                True,
                False,
            ),
            Probe(
                "stock: Tamsin got the Leeds job",
                "- [world] Tamsin, the user's roommate, is moving out next month and got a job in Leeds  {entities: Tamsin, Leeds, user}\n"
                "- [world] Ravi, the user's manager, said the user can work remotely while looking for a new place  {entities: Ravi, user}",
                False,
                True,
            ),
        ),
        notes="The 'moving out' fact stays Tamsin's; only the job moves. 'He' was added to the pronoun trap so an unresolved 'he' is also caught.",
    ),
    # 2 ── multi-speaker-attribution
    Variant(
        case_id="multi-speaker-attribution",
        arm="wording",
        kind="retain_capability",
        change="Priya/Tom/migration/runbook renamed to Mei/Jonas/schema upgrade/failover checklist; 'off' became 'out'; same speaker structure.",
        stock_answer="Priya takes the migration; Tom handles the runbook and is off Friday.",
        expected_answer="Mei takes the schema upgrade; Jonas handles the failover checklist and is out Friday.",
        case=_SPEAKER_WORDING,
        probes=(
            Probe(
                "correct",
                "- [world] Mei said she will take the schema upgrade  {entities: Mei}\n"
                "- [world] Jonas will handle the failover checklist and is out on Friday  {entities: Jonas}\n"
                "- [world] The team plans to ship on Thursday  {entities: Mei}",
                True,
            ),
            Probe(
                "wrong: Friday off and allergy swapped to Mei",
                "- [world] Mei will take the schema upgrade and is out on Friday  {entities: Mei}\n"
                "- [world] Jonas will handle the failover checklist  {entities: Jonas}",
                False,
            ),
        ),
    ),
    Variant(
        case_id="multi-speaker-attribution",
        arm="counterfactual",
        kind="retain_capability",
        change="Priya, not Tom, is the one off on Friday (the clause moves from Tom's line to Priya's); migration and runbook owners unchanged.",
        stock_answer="Tom is off on Friday.",
        expected_answer="Priya is off on Friday.",
        case=_SPEAKER_CF,
        stale_claim="A fact states that Tom is off on Friday.",
        probes=(
            Probe(
                "correct: Priya off Friday",
                "- [world] Priya will take the database migration  {entities: Priya}\n"
                "- [world] Tom will handle the rollback runbook  {entities: Tom}\n"
                "- [world] Priya is off on Friday and wants to ship on Thursday to avoid on-call weekends  {entities: Priya}",
                True,
                False,
            ),
            Probe(
                "stock: Tom off Friday",
                "- [world] Priya will take the database migration  {entities: Priya}\n"
                "- [world] Tom will handle the rollback runbook and is off on Friday  {entities: Tom}",
                False,
                True,
            ),
        ),
        notes="Single relationship change; allergy/Thursday distractor kept with Priya.",
    ),
    # 3 ── context-names-the-speaker
    Variant(
        case_id="context-names-the-speaker",
        arm="wording",
        kind="retain_capability",
        change="Customer renamed Marta Seixas, bakery/Porto/invoices became pastry shop/Coimbra/bills; context rephrased; speaker still named only through context.",
        stock_answer="Inês Duarte runs a bakery in Porto and needs invoices in Portuguese.",
        expected_answer="Marta Seixas runs a pastry shop in Coimbra and needs bills in Portuguese.",
        case=_CONTEXT_WORDING,
        probes=(
            Probe(
                "correct",
                "- [world] Marta Seixas has been a subscriber for three years  {entities: Marta Seixas}\n"
                "- [world] Marta Seixas runs a pastry shop in Coimbra  {entities: Marta Seixas, Coimbra}\n"
                "- [world] Marta Seixas needs her bills in Portuguese from now on  {entities: Marta Seixas}",
                True,
            ),
            Probe(
                "wrong: attributed to the agent",
                "- [world] The assistant runs a pastry shop in Coimbra and needs the bills in Portuguese  {entities: Coimbra}",
                False,
            ),
        ),
    ),
    Variant(
        case_id="context-names-the-speaker",
        arm="counterfactual",
        kind="retain_capability",
        change="The context names Rui Matos (stock: Inês Duarte) as the first-person speaker; the content is unchanged.",
        stock_answer="Inês Duarte runs the Porto bakery and needs Portuguese invoices.",
        expected_answer="Rui Matos runs the Porto bakery and needs Portuguese invoices.",
        case=_CONTEXT_CF,
        stale_claim="A fact attributes the bakery or the invoice request to Inês Duarte.",
        probes=(
            Probe(
                "correct: Rui Matos",
                "- [world] Rui Matos has been a customer for three years  {entities: Rui Matos}\n"
                "- [world] Rui Matos runs a bakery in Porto  {entities: Rui Matos, Porto}\n"
                "- [world] Rui Matos needs invoices in Portuguese from now on  {entities: Rui Matos}",
                True,
                False,
            ),
            Probe(
                "stock: Inês Duarte",
                "- [world] Inês Duarte runs a bakery in Porto  {entities: Inês Duarte, Porto}\n"
                "- [world] Inês Duarte needs invoices in Portuguese from now on  {entities: Inês Duarte}",
                False,
                True,
            ),
        ),
        notes="Weak-ish: the content never names the speaker, so a stale answer needs the model to ignore the changed context (or recall the stock name from training data).",
    ),
    # 4 ── in-text-correction
    Variant(
        case_id="in-text-correction",
        arm="wording",
        kind="retain_capability",
        change="Design review/room 4B/platform team became budget review/room 7C/infrastructure team; 'sorry, I mean' became 'apologies, I meant'; same Tuesday-to-Wednesday correction and timestamp.",
        stock_answer="Wednesday 4 March 2026 (not Tuesday).",
        expected_answer="Wednesday 4 March 2026 (not Tuesday).",
        case=_CORRECTION_WORDING,
        probes=(
            Probe(
                "correct",
                "- [world] The budget review is on Wednesday, 4 March 2026, in room 7C with the whole infrastructure team  [occurred_start=2026-03-04]  {entities: infrastructure team}",
                True,
            ),
            Probe(
                "wrong: the slip stored",
                "- [world] The budget review is on Tuesday, 3 March 2026, in room 7C with the whole infrastructure team  [occurred_start=2026-03-03]  {entities: infrastructure team}",
                False,
            ),
        ),
    ),
    Variant(
        case_id="in-text-correction",
        arm="counterfactual",
        kind="retain_capability",
        change="The speaker corrects Tuesday to Thursday (stock: Wednesday), so the resolved date moves from 4 March to 5 March 2026 (item timestamp 2026-03-02, a Monday).",
        stock_answer="Wednesday 4 March 2026.",
        expected_answer="Thursday 5 March 2026.",
        case=_CORRECTION_CF,
        stale_claim="A fact states that the design review is on Wednesday.",
        probes=(
            Probe(
                "correct: Thursday 5 March",
                "- [world] The design review is on Thursday, 5 March 2026, in room 4B with the whole platform team  [occurred_start=2026-03-05]  {entities: platform team}",
                True,
                False,
            ),
            Probe(
                "stock: Wednesday 4 March",
                "- [world] The design review is on Wednesday, 4 March 2026, in room 4B with the whole platform team  [occurred_start=2026-03-04]  {entities: platform team}",
                False,
                True,
            ),
            Probe(
                "wrong: the slip (Tuesday) stored",
                "- [world] The design review is on Tuesday, 3 March 2026, in room 4B with the whole platform team  [occurred_start=2026-03-03]  {entities: platform team}",
                False,
                False,
            ),
        ),
    ),
    # 5 ── date-borrowed-from-nearby-subject
    Variant(
        case_id="date-borrowed-from-nearby-subject",
        arm="wording",
        kind="retain_fidelity",
        change="Customer notifications/project/September 14/equipment audit became resident notifications/programme/October 6/equipment check; same bait structure and sentence shape. (v1 review: a 'partner notices ... before the survey begins' rewording made the stock trap judge flag a correct output 4/4 against 2/4 for stock, so the sentence shape is kept.)",
        stock_answer="Notifications must be completed before fieldwork begins, carrying no date of their own.",
        expected_answer="Resident notifications must be completed before fieldwork begins, carrying no date of their own.",
        case=_DATE_WORDING,
        probes=(
            Probe(
                "correct: no borrowed date",
                "- Resident notifications must be completed before fieldwork begins\n"
                "- The programme has a target start of October 6\n"
                "- The equipment check finished last quarter",
                True,
            ),
            Probe(
                "wrong: project date borrowed into the notices",
                "- Resident notifications must be completed by October 6, before fieldwork begins\n"
                "- The equipment check finished last quarter",
                False,
            ),
        ),
    ),
    Variant(
        case_id="date-borrowed-from-nearby-subject",
        arm="counterfactual",
        kind="retain_fidelity",
        change="The notifications now carry their own explicit deadline, August 28, distinct from the project's September 14 start; the stock no-date output now fails correctness.",
        stock_answer="Notifications carry no date of their own.",
        expected_answer="Notifications are due August 28, before fieldwork begins; September 14 is the project start, not their deadline.",
        case=_DATE_CF,
        stale_claim="The customer notifications have no deadline of their own, only that they come before fieldwork begins.",
        probes=(
            Probe(
                "correct: own deadline August 28",
                "- Customer notifications must be completed by August 28, before fieldwork begins\n"
                "- The project has a target start of September 14\n"
                "- The equipment audit finished last quarter",
                True,
                False,
            ),
            Probe(
                "stock: no deadline stored",
                "- Customer notifications must be completed before fieldwork begins\n"
                "- The project has a target start of September 14\n"
                "- The equipment audit finished last quarter",
                False,
                True,
            ),
            Probe(
                "wrong: project start date borrowed as the deadline",
                "- Customer notifications must be completed by September 14, before fieldwork begins\n"
                "- The equipment audit finished last quarter",
                False,
                False,
            ),
        ),
        notes="The trap targets the neighbour's date (September 14) as the notifications' date; August 28 is the legitimate value. Risk: a fact that mentions both 'by August 28' and 'fieldwork begins September 14' could be judged either way.",
    ),
    # 6 ── owner-borrowed-from-nearby-subject
    Variant(
        case_id="owner-borrowed-from-nearby-subject",
        arm="wording",
        kind="retain_fidelity",
        change="Priya/schema migration/connection-pool limit/launch became Amara/index rebuild/worker-thread cap/go-live; same bait structure.",
        stock_answer="The pool limit still needs raising before launch, with no owner named.",
        expected_answer="The worker-thread cap still needs raising before go-live, with no owner named.",
        case=_OWNER_WORDING,
        probes=(
            Probe(
                "correct: no borrowed owner",
                "- Amara owns the index rebuild and signed off on the cutover plan\n"
                "- The worker-thread cap still needs to be raised before go-live\n"
                "- Both items are tracked in the release checklist",
                True,
            ),
            Probe(
                "wrong: Amara made the owner of the cap",
                "- Amara is responsible for raising the worker-thread cap before go-live\n"
                "- Amara signed off on the cutover plan",
                False,
            ),
        ),
    ),
    Variant(
        case_id="owner-borrowed-from-nearby-subject",
        arm="counterfactual",
        kind="retain_fidelity",
        change="The pool-limit item now names its own owner, Dmitri (stock: no owner), distinct from Priya who owns the schema migration.",
        stock_answer="The pool limit has no named owner.",
        expected_answer="Dmitri owns raising the connection-pool limit before launch; Priya owns the migration only.",
        case=_OWNER_CF,
        stale_claim="The connection-pool limit has no named owner.",
        probes=(
            Probe(
                "correct: Dmitri owns the pool limit",
                "- Priya owns the schema migration and signed off on the rollback plan\n"
                "- Dmitri owns raising the connection-pool limit, which still needs to happen before launch\n"
                "- Both items are tracked in the launch checklist",
                True,
                False,
            ),
            Probe(
                "stock: pool limit with no owner",
                "- Priya owns the schema migration and signed off on the rollback plan\n"
                "- The connection-pool limit still needs to be raised before launch\n"
                "- Both items are tracked in the launch checklist",
                False,
                True,
            ),
            Probe(
                "wrong: Priya made the owner of the pool limit",
                "- Priya owns the schema migration and signed off on the rollback plan\n"
                "- Priya is responsible for raising the connection-pool limit before launch",
                False,
                False,
            ),
        ),
    ),
)


if __name__ == "__main__":
    import datetime

    expected_ids = [
        "coreference-roles",
        "multi-speaker-attribution",
        "context-names-the-speaker",
        "in-text-correction",
        "date-borrowed-from-nearby-subject",
        "owner-borrowed-from-nearby-subject",
    ]
    assert len(VARIANTS) == 12
    assert [v.case_id for v in VARIANTS][::2] == expected_ids
    assert [v.case_id for v in VARIANTS][1::2] == expected_ids
    assert [v.arm for v in VARIANTS] == ["wording", "counterfactual"] * 6
    for v in VARIANTS:
        assert v.case.id == v.case_id, v.case_id  # type: ignore[attr-defined]
        if v.case_id in _CAPS:
            assert v.kind == "retain_capability"
            assert isinstance(v.case, Case)
            stock = _CAPS[v.case_id]
            assert v.case != stock
            text = "\n".join(i.content + (i.context or "") for i in v.case.items)
            for needle in (*v.case.entities, *v.case.contains):
                assert needle.lower() in text.lower(), needle
        else:
            assert v.kind == "retain_fidelity"
            assert isinstance(v.case, FidelityCase)
            assert v.case != _FIDELITY[v.case_id]
            assert v.case.must_not_carry
            assert v.case.subject
        assert v.stock_answer and v.expected_answer and v.change
        assert any(p.expect_pass for p in v.probes) and any(not p.expect_pass for p in v.probes)
        if v.arm == "counterfactual":
            assert v.expected_answer != v.stock_answer
            assert v.stale_claim
            assert any(p.expect_stale is True and not p.expect_pass for p in v.probes)
            assert all(p.expect_stale is False for p in v.probes if p.expect_pass)
        else:
            assert not v.stale_claim
    # Weekday arithmetic for the in-text-correction counterfactual.
    stamp = datetime.date(2026, 3, 2)
    assert stamp.strftime("%A") == "Monday"
    assert (stamp + datetime.timedelta(days=3)).strftime("%A %Y-%m-%d") == "Thursday 2026-03-05"
    cf = next(v for v in VARIANTS if v.case_id == "in-text-correction" and v.arm == "counterfactual")
    assert "Thursday" in cf.case.items[0].content and "5 March 2026" in cf.case.must[0]  # type: ignore[attr-defined]
    assert cf.case.items[0].timestamp == _CAPS["in-text-correction"].items[0].timestamp  # type: ignore[attr-defined]
    assert any("occurred_start=2026-03-05" in p.response for p in cf.probes if p.expect_pass)
    # Fidelity counterfactuals: the target fact's own value is in the document and in must_extract.
    date_cf = _DATE_CF
    assert "August 28" in date_cf.document and "August 28" in date_cf.must_extract
    assert "September 14" in date_cf.document and "September 14" in date_cf.must_not_carry
    owner_cf = _OWNER_CF
    assert "Dmitri" in owner_cf.document and "Dmitri" in owner_cf.must_extract
    assert "Priya" in owner_cf.document and "Priya" in owner_cf.must_not_carry
    # Wording fidelity arms keep the bait value in the document and the trap.
    assert "October 6" in _DATE_WORDING.document and "October 6" in _DATE_WORDING.must_not_carry
    assert "Amara" in _OWNER_WORDING.document and "Amara" in _OWNER_WORDING.must_not_carry
    # Only the intended fields differ from upstream.
    for v in VARIANTS:
        if v.case_id in _CAPS:
            s = _CAPS[v.case_id]
            assert (v.case.category, v.case.origin, v.case.bank_config, v.case.minimum) == (  # type: ignore[attr-defined]
                s.category, s.origin, s.bank_config, s.minimum
            )
        else:
            s = _FIDELITY[v.case_id]
            assert v.case.category == s.category  # type: ignore[attr-defined]
    print("ok", len(VARIANTS), "variants")
