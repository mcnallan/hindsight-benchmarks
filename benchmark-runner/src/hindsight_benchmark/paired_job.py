"""Custom and paired suites for ``hindsight-evals``: selection, durable job state, retries, report.

``--suite stock`` (the default) is untouched by anything here. ``custom`` runs only
the variant arms (wording, counterfactual); ``paired`` runs the stock arm beside
them, in fresh banks, through the same stock grading code. Neither writes a
leaderboard result: each job writes ``results/paired/<result id>.json`` with the
fixture revision, the effective settings, the tallies and one compact row per case
execution. Model outputs, traces and judge reasoning stay in the ignored run dir.

``--state DIR`` makes a job resumable: the first start records the run dir, the
fixture revision and a digest of the effective settings in ``DIR/job.json``; a
later start with the same DIR reuses that run dir (skipping finished passes) and
refuses to continue if the fixtures or settings changed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from hindsight_benchmark.paired import FIXTURE_SET, PINNED_COMMIT, revision

RUNNER_DIR = Path(__file__).resolve().parents[2]
PAIRED_DIR = Path(__file__).resolve().parent / "paired"
REPORT_DIR = RUNNER_DIR.parent / "results" / "paired"
SUITES = ("stock", "custom", "paired")
ARMS_FOR = {"custom": ("wording", "counterfactual"), "paired": ("stock", "wording", "counterfactual")}
#: The operations that have paired fixtures, and the test module that runs each.
OPERATIONS = {
    "retain": [str(PAIRED_DIR / "test_paired_retain.py")],
    "reflect": [str(PAIRED_DIR / "test_paired_reflect.py")],
}
#: Reruns of executions that hit an infrastructure error, per pass.
INFRA_RETRIES = int(os.getenv("HS_PAIRED_INFRA_RETRIES", "2"))
INFRA_RETRY_WAIT = float(os.getenv("HS_PAIRED_INFRA_RETRY_WAIT", "120"))


def operations(selected: str | None) -> dict[str, list[str]]:
    if not selected:
        return dict(OPERATIONS)
    names = selected.split(",")
    unknown = [n for n in names if n not in OPERATIONS]
    if unknown:
        raise SystemExit(f"no paired fixtures for operations {unknown}; available: {sorted(OPERATIONS)}")
    return {n: OPERATIONS[n] for n in names}


def pytest_env(suite: str, case_ids: list[str], records: Path, only: list[str] | None = None) -> dict[str, str]:
    env = {"HS_PAIRED_ARMS": ",".join(ARMS_FOR[suite]), "HS_PAIRED_CASES": ",".join(case_ids),
           "HS_PAIRED_RECORDS": str(records)}
    if only:
        env["HS_PAIRED_ONLY"] = ",".join(only)
    return env


def effective_settings(cfg: dict, args, provider: dict, repeats: int, ops: dict, case_ids: list[str]) -> dict:
    """Everything that decides what a job measures. Credentials are named, never included."""
    return {
        "suite": args.suite,
        "arms": list(ARMS_FOR[args.suite]),
        "operations": list(ops),
        "case_ids": case_ids,
        "repeats": repeats,
        "hindsight_commit": cfg["hindsight_commit"],
        "provider": args.provider,
        "hindsight_provider": provider.get("hindsight_provider"),
        "base_url": provider.get("base_url"),
        "model": args.model,
        "model_settings": cfg.get("model_settings", {}).get(args.model, {}),
        "decision_model": args.decision,
        "embedder": args.embedder,
        "reranker": args.reranker,
        "judge_model": cfg["judge_model"],
        "judge_url": os.getenv("HS_BENCH_CODEX_LB_URL"),
        "judge_reasoning_effort": os.getenv("HS_BENCH_JUDGE_REASONING_EFFORT", "high"),
        "full": bool(cfg.get("full")),
    }


def digest(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def open_job(state: Path | None, settings: dict, fresh_run_dir) -> tuple[Path, dict]:
    """The job's run dir and record: created on first start, verified and reused on resume."""
    fixture_revision = revision()
    if state and (state / "job.json").exists():
        job = json.loads((state / "job.json").read_text())
        problems = []
        if job["fixture_revision"] != fixture_revision:
            problems.append(f"fixtures changed: recorded {job['fixture_revision']}, now {fixture_revision}")
        if job["settings_digest"] != digest(settings):
            changed = sorted(k for k in settings if job["settings"].get(k) != settings[k])
            problems.append(f"settings changed: {changed}")
        if problems:
            sys.exit(f"refusing to resume {state}: " + "; ".join(problems) +
                     ". Start a new job (new --state and --result-id) to measure different inputs.")
        run_dir = RUNNER_DIR / job["run_dir"]
        print(f"resuming paired job {state} in {run_dir}", flush=True)
        return run_dir, job
    run_dir = fresh_run_dir()
    job = {"fixture_set": FIXTURE_SET, "fixture_revision": fixture_revision, "pinned_commit": PINNED_COMMIT,
           "settings": settings, "settings_digest": digest(settings),
           "run_dir": str(run_dir.relative_to(RUNNER_DIR)),
           "started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    if state:
        state.mkdir(parents=True, exist_ok=True)
        pending = state / "job.json.tmp"
        pending.write_text(json.dumps(job, indent=2) + "\n")
        pending.replace(state / "job.json")
    return run_dir, job


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def infra_failures(junit: Path, records: Path) -> list[str]:
    """``case@arm`` executions that infrastructure broke: an error record, or a junit case with no record."""
    rows = read_records(records)
    bad = {f"{r['case_id']}@{r['arm']}" for r in rows if r["blame"] == "infrastructure"}
    recorded = {f"{r['case_id']}@{r['arm']}" for r in rows}
    if junit.exists():
        for case in ET.parse(junit).iter("testcase"):
            name = case.get("name", "")
            if "[" in name:
                key = name[name.index("[") + 1 : -1].split("/", 1)[-1]
                if key not in recorded or case.find("error") is not None:
                    bad.add(key)
    return sorted(bad)


def judge_healthy() -> bool:
    """A real structured completion from the judge, not just a listing."""
    from hindsight_benchmark.codex_lb_judge import generate_json

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    try:
        return generate_json(os.getenv("HS_PAIRED_JUDGE_MODEL", "gpt-6.2-sol"), 'Reply {"ok": true}.', schema)["ok"] is True
    except Exception as exc:  # noqa: BLE001 — any failure means "not yet"
        print(f"judge health check failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr, flush=True)
        return False


def splice_records(records: Path, retry_records: Path, keys: set[str]) -> None:
    rows = [r for r in read_records(records) if f"{r['case_id']}@{r['arm']}" not in keys]
    rows += [r for r in read_records(retry_records) if f"{r['case_id']}@{r['arm']}" in keys]
    records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    retry_records.unlink(missing_ok=True)


def retry_infra(i: int, junit: Path, tokens: Path, records: Path, rerun, splice) -> None:
    """Rerun, in fresh banks, only the executions an endpoint or judge error broke."""
    for attempt in range(INFRA_RETRIES):
        keys = infra_failures(junit, records)
        if not keys:
            return
        print(f"pass {i + 1}: {len(keys)} executions hit infrastructure errors {keys}; retry {attempt + 1} "
              f"in {INFRA_RETRY_WAIT:.0f}s", file=sys.stderr, flush=True)
        time.sleep(INFRA_RETRY_WAIT)
        if os.getenv("HS_BENCH_CODEX_LB_URL") and not judge_healthy():
            continue
        retry_junit, retry_tokens = junit.with_suffix(".retry.xml"), tokens.with_suffix(".retry.jsonl")
        retry_records = records.with_suffix(".retry.jsonl")
        for p in (retry_tokens, retry_records):
            p.unlink(missing_ok=True)
        rerun(retry_junit, retry_tokens, retry_records, keys)
        if not retry_junit.exists():
            continue
        names = {c.get("name") for c in ET.parse(retry_junit).iter("testcase")}
        splice(junit, tokens, retry_junit, retry_tokens, names)
        splice_records(records, retry_records, set(keys))


def compact(row: dict) -> dict:
    """A record without model output, queries or long judge text: safe to commit."""
    keep = ("case_id", "arm", "kind", "operation", "category", "input_sha", "fixture_revision", "passed",
            "hit_trap", "stale", "blame", "gold_retrieved", "gold_total", "seconds")
    out = {k: row.get(k) for k in keep}
    if row.get("infra_error"):
        out["infra_error"] = row["infra_error"].split(":", 1)[0]
    return out


def write_report(result_id: str, job: dict, run_dir: Path, ops: dict, repeats: int, stamp: str) -> Path:
    executions = []
    for i in range(repeats):
        for row in read_records(run_dir / f"records-{i}.jsonl"):
            executions.append({"repeat": i, **compact(row)})
    revisions = {e["fixture_revision"] for e in executions}
    if revisions - {job["fixture_revision"]}:
        raise RuntimeError(f"records carry fixture revisions {sorted(revisions)}, job {job['fixture_revision']}")
    by_arm: dict[str, dict] = {}
    for e in executions:
        a = by_arm.setdefault(e["arm"], {"passed": 0, "graded": 0, "infrastructure": 0, "stale": 0})
        if e["passed"] is None:
            a["infrastructure"] += 1
        else:
            a["graded"] += 1
            a["passed"] += bool(e["passed"])
            a["stale"] += bool(e["stale"])
    for a in by_arm.values():
        a["pass_rate"] = round(100 * a["passed"] / a["graded"], 1) if a["graded"] else None
    report = {
        "result_id": result_id,
        "timestamp": stamp,
        **{k: job[k] for k in ("fixture_set", "fixture_revision", "pinned_commit", "settings", "settings_digest",
                               "run_dir", "started")},
        "repeats": repeats,
        "arms": by_arm,
        "operations": ops,
        "executions": executions,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{result_id}.json"
    pending = out.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, indent=1) + "\n")
    pending.replace(out)
    return out


