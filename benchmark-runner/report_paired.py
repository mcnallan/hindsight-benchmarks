"""Summarize paired robustness results: pass-rate change, flips, stale answers, failure stages.

  python report_paired.py ../results/paired/local-*-paired-7b46a07c-v2-*.json [--out REPORT.md]

Reads the committed per-job reports (``results/paired/*.json``). A flip compares the
stock arm with a variant arm in the same profile, case and repeat. Executions with an
infrastructure failure are counted apart and excluded from every rate and flip.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ("wording", "counterfactual")


def profile_of(report: dict) -> str:
    """The profile, plus the judge when it is not the default Sol high, so two judges never merge."""
    m = re.match(r"local-(.+?)-([a-z]+\d+-[a-z]+)-(?:paired|custom)-", report["result_id"])
    if not m:
        return report["result_id"]
    return m[1] if m[2] == "sol62-high" else f"{m[1]} · {m[2]}"


def pct(passed: int, graded: int) -> str:
    return f"{100 * passed / graded:.1f}% ({passed}/{graded})" if graded else "n/a"


def summarize(reports: list[dict]) -> str:
    lines = ["# Paired robustness results", ""]
    revisions = sorted({r["fixture_revision"] for r in reports})
    lines += [f"Fixture revision: {', '.join(revisions)}. Pinned Hindsight {reports[0]['pinned_commit'][:12]}.", ""]
    lines += ["| Profile | Stock | Wording | Δ wording | Counterfactual | Δ counterfactual | Stale CF answers | Infra |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    flips, stale_rows, blames, infra_rows = [], [], defaultdict(Counter), []
    for report in reports:
        profile = profile_of(report)
        rows = report["executions"]
        graded = [e for e in rows if e["passed"] is not None]
        infra_rows += [(profile, e) for e in rows if e["passed"] is None]
        by_key = {(e["case_id"], e["arm"], e["repeat"]): e for e in graded}
        cells, deltas = {}, {}
        for arm in ("stock", *VARIANTS):
            arm_rows = [e for e in graded if e["arm"] == arm]
            cells[arm] = pct(sum(e["passed"] for e in arm_rows), len(arm_rows))
        for arm in VARIANTS:
            # Matched pairs only: both arms graded for the same case and repeat.
            pairs = [(by_key[(c, "stock", r)], e) for (c, a, r), e in by_key.items()
                     if a == arm and (c, "stock", r) in by_key]
            if pairs:
                d = 100 * (sum(v["passed"] for _, v in pairs) - sum(s["passed"] for s, _ in pairs)) / len(pairs)
                deltas[arm] = f"{d:+.1f} pp (n={len(pairs)})"
            else:
                deltas[arm] = "n/a"
            for s, v in pairs:
                if s["passed"] != v["passed"]:
                    flips.append((profile, v["case_id"], arm, v["repeat"], s, v))
        cf = [e for e in graded if e["arm"] == "counterfactual"]
        stale = [e for e in cf if e.get("stale")]
        stale_rows += [(profile, e) for e in stale]
        for e in graded:
            if not e["passed"]:
                blames[profile][(e["arm"], e["blame"])] += 1
        lines.append(f"| {profile} | {cells['stock']} | {cells['wording']} | {deltas['wording']} | "
                     f"{cells['counterfactual']} | {deltas['counterfactual']} | {len(stale)}/{len(cf)} | "
                     f"{sum(1 for e in rows if e['passed'] is None)} |")
    lines += ["", "## Flips against the matched stock execution", ""]
    if flips:
        lines += ["| Profile | Case | Arm | Repeat | Stock | Variant | Variant failure stage |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for profile, case, arm, rep, s, v in sorted(flips, key=lambda f: (f[1], f[2], f[0], f[3])):
            lines.append(f"| {profile} | {case} | {arm} | {rep + 1} | {'pass' if s['passed'] else 'fail'} | "
                         f"{'pass' if v['passed'] else 'fail'} | {'' if v['passed'] else v['blame']} |")
    else:
        lines.append("None.")
    lines += ["", "## Counterfactuals that repeated the stock answer", ""]
    if stale_rows:
        for profile, e in sorted(stale_rows, key=lambda x: (x[1]["case_id"], x[0], x[1]["repeat"])):
            lines.append(f"- {profile} `{e['case_id']}` repeat {e['repeat'] + 1}: "
                         f"{'passed anyway' if e['passed'] else 'failed'} ({e['blame']})")
    else:
        lines.append("None.")
    lines += ["", "## Failure stage by arm", "",
              "Extraction is Retain's own output; retrieval means a gold fact never reached Reflect's model; "
              "generation means every gold fact arrived and the answer was still wrong.", ""]
    for profile, counts in blames.items():
        parts = ", ".join(f"{arm} {blame} {n}" for (arm, blame), n in sorted(counts.items()))
        lines.append(f"- {profile}: {parts}")
    lines += ["", "## Infrastructure failures (excluded from rates)", ""]
    if infra_rows:
        for profile, e in infra_rows:
            lines.append(f"- {profile} `{e['case_id']}@{e['arm']}` repeat {e['repeat'] + 1}: {e.get('infra_error', '?')}")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    text = summarize([json.loads(path.read_text()) for path in args.reports])
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
