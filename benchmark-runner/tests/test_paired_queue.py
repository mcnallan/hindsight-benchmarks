"""Queue ordering, interruption and resume, result-ID isolation, and paired job state.

The queue runs with a stub interpreter that records each benchmark invocation and
hands everything else (config generation, fixture-set lookup) to the real one, so
no endpoint, judge or Hindsight server is touched.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hindsight_benchmark import paired_job

RUNNER = Path(__file__).resolve().parents[1]
QUEUE = RUNNER / "run_local_system_evals.sh"
STUB = """#!/usr/bin/env bash
if [[ $1 == -m && $2 == hindsight_benchmark.evals ]]; then
  printf 'evals %s\\n' "${*:3}" >> "$FAKE_LOG"
  if [[ -n ${FAKE_FAIL_MATCH:-} && "$*" == *"$FAKE_FAIL_MATCH"* && ! -f $FAKE_LOG.failed ]]; then
    touch "$FAKE_LOG.failed"; exit 1
  fi
  exit 0
elif [[ $1 == */run_local_amb.py ]]; then
  printf 'amb %s\\n' "$2" >> "$FAKE_LOG"; exit 0
fi
exec "$REAL_PYTHON" "$@"
"""


def run_queue(tmp_path, *args, fail=None, smoke_done=True):
    run_dir = tmp_path / "run"
    if smoke_done:
        for profile in ("qwen35-4b-off", "inkling-2-mini-off", "inkling-2-mini-minimal"):
            (run_dir / profile).mkdir(parents=True, exist_ok=True)
            (run_dir / profile / "smoke.done").touch()
    stub = tmp_path / "python"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {**os.environ, "HS_QUEUE_PYTHON": str(stub), "REAL_PYTHON": sys.executable, "FAKE_LOG": str(log),
           "CODEX_LB_TOKEN": "test", "EVAL_RETRIEVAL_KEY": "test", "INKLING_API_KEY": "test",
           "FAKE_FAIL_MATCH": fail or ""}
    done = subprocess.run(["bash", str(QUEUE), str(run_dir), *args], env=env, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return done, calls, run_dir


def flag(call, name):
    parts = call.split()
    return parts[parts.index(name) + 1] if name in parts else None


def test_default_queue_is_unchanged(tmp_path):
    done, calls, _ = run_queue(tmp_path, "qwen")
    assert done.returncode == 0, done.stderr
    ops = [flag(c, "--operations") for c in calls if c.startswith("evals")]
    assert ops == ["retain", "reflect", "consolidation", "refresh_pages"]
    assert {flag(c, "--result-id") for c in calls if c.startswith("evals")} == {"local-qwen35-4b-off-sol62-high"}
    assert all("--suite" not in c for c in calls)
    assert calls[-1].startswith("amb ") and calls[-1].endswith("qwen35-4b-off")


def test_paired_selection_orders_profiles_and_skips_stock_and_amb(tmp_path):
    done, calls, run_dir = run_queue(tmp_path, "inkling", "--select", "paired", "--operations", "retain,reflect")
    assert done.returncode == 0, done.stderr
    assert len(calls) == 2 and all(c.startswith("evals") for c in calls)
    assert [flag(c, "--model") for c in calls] == ["inkling-2-mini", "inkling-2-mini"]
    states = [flag(c, "--state") for c in calls]
    assert "inkling-2-mini-off" in states[0] and "inkling-2-mini-minimal" in states[1]
    assert all(flag(c, "--suite") == "paired" and flag(c, "--repeats") == "2" for c in calls)
    ids = [flag(c, "--result-id") for c in calls]
    assert len(set(ids)) == 2
    assert all(i.startswith("local-inkling-2-mini-") and "-paired-" in i for i in ids)
    assert not (run_dir / "inkling-2-mini-off" / "retain.done").exists()


def test_interrupted_queue_resumes_where_it_stopped(tmp_path):
    args = ("inkling", "--select", "paired")
    done, calls, run_dir = run_queue(tmp_path, *args, fail="inkling-2-mini-minimal")
    assert done.returncode != 0
    assert len(calls) == 2  # off finished, minimal failed
    off_jobs = list((run_dir / "inkling-2-mini-off").glob("paired-*/done"))
    assert len(off_jobs) == 1
    assert not list((run_dir / "inkling-2-mini-minimal").glob("paired-*/done"))

    (tmp_path / "calls.log").unlink()
    done, calls, _ = run_queue(tmp_path, *args, fail="inkling-2-mini-minimal")  # the stub fails only once
    assert done.returncode == 0, done.stderr
    assert len(calls) == 1 and "inkling-2-mini-minimal" in flag(calls[0], "--state")
    assert flag(calls[0], "--state") == str(next((run_dir / "inkling-2-mini-minimal").glob("paired-*")))


def test_subsets_and_suites_get_isolated_result_ids(tmp_path):
    _, full, _ = run_queue(tmp_path / "a", "qwen", "--select", "paired")
    _, subset, _ = run_queue(tmp_path / "b", "qwen", "--select", "paired", "--cases", "hq-owner-latest")
    _, smoke, _ = run_queue(tmp_path / "c", "qwen", "--select", "paired", "--cases", "hq-owner-latest", "--repeats", "1")
    _, custom, _ = run_queue(tmp_path / "d", "qwen", "--select", "custom")
    ids = [flag(c[0], "--result-id") for c in (full, subset, smoke, custom)]
    assert len(set(ids)) == 4
    assert "local-qwen35-4b-off-sol62-high" not in ids
    assert flag(subset[0], "--case-ids") == "hq-owner-latest" and flag(full[0], "--case-ids") is None


def test_judge_choice_isolates_result_ids_state_and_config(tmp_path):
    _, sol, run_dir = run_queue(tmp_path, "qwen", "--select", "paired")
    (tmp_path / "calls.log").unlink()
    _, terra, _ = run_queue(tmp_path, "qwen", "--select", "paired", "--judge", "gpt-6.1-terra:medium")
    assert len(sol) == len(terra) == 1  # the finished Sol job does not hide the Terra one
    assert flag(sol[0], "--result-id").startswith("local-qwen35-4b-off-sol62-high-paired-")
    assert flag(terra[0], "--result-id").startswith("local-qwen35-4b-off-terra61-medium-paired-")
    assert Path(flag(terra[0], "--state")).name.startswith("terra61-medium-paired-")
    assert flag(sol[0], "--state") != flag(terra[0], "--state")
    profile = run_dir / "qwen35-4b-off"
    assert json.loads((profile / "evals.json").read_text())["judge_model"] == "gpt-6.2-sol"
    assert json.loads((profile / "evals-terra61-medium.json").read_text())["judge_model"] == "gpt-6.1-terra"
    assert flag(terra[0], "--config").endswith("evals-terra61-medium.json")


def test_amb_is_selectable_alone(tmp_path):
    done, calls, _ = run_queue(tmp_path, "qwen", "--select", "amb", smoke_done=False)
    assert done.returncode == 0, done.stderr
    assert calls == [f"amb {tmp_path / 'run' / 'qwen35-4b-off'}"]


@pytest.mark.parametrize("args", [
    ("--select", "stock", "--cases", "x"),
    ("--select", "custom,paired"),
    ("--select", "nonsense"),
    ("--bogus",),
    ("--judge", "gpt-6.1-terra"),
    ("--judge", "gpt-6.1-terra:extreme"),
    ("--judge", "terra:medium"),
])
def test_invalid_selections_are_refused(tmp_path, args):
    done, calls, _ = run_queue(tmp_path, "qwen", *args)
    assert done.returncode == 2 and calls == []


def test_after_session_returns_when_session_is_absent(tmp_path):
    done, calls, _ = run_queue(tmp_path, "qwen", "--select", "paired", "--after-session", "no-such-session-xyz")
    assert done.returncode == 0, done.stderr
    assert len(calls) == 1


# ── Job state ────────────────────────────────────────────────────────────────

def settings(**over):
    return {"suite": "paired", "model": "m", "repeats": 2, **over}


def test_job_state_resumes_same_run_dir_and_refuses_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_job, "RUNNER_DIR", tmp_path)
    monkeypatch.setattr(paired_job, "revision", lambda: "rev-a")
    state = tmp_path / "state"
    made = []

    def fresh():
        d = tmp_path / ".hindsight" / "runs" / f"x-{len(made)}"
        d.mkdir(parents=True)
        made.append(d)
        return d

    run_dir, job = paired_job.open_job(state, settings(), fresh)
    assert json.loads((state / "job.json").read_text())["run_dir"] == ".hindsight/runs/x-0"
    again, _ = paired_job.open_job(state, settings(), fresh)
    assert again == run_dir and len(made) == 1

    with pytest.raises(SystemExit, match="settings changed: \\['repeats'\\]"):
        paired_job.open_job(state, settings(repeats=3), fresh)
    monkeypatch.setattr(paired_job, "revision", lambda: "rev-b")
    with pytest.raises(SystemExit, match="fixtures changed"):
        paired_job.open_job(state, settings(), fresh)


def record(case, arm, blame="ok", passed=True, **extra):
    return {"case_id": case, "arm": arm, "kind": "reflect_corpus", "operation": "reflect", "category": "update",
            "input_sha": "s", "fixture_revision": "rev", "passed": passed, "hit_trap": False, "stale": None,
            "blame": blame, "output": "secret model text", "reasons": ["judge reasoning"], **extra}


def test_infra_failures_and_record_splice(tmp_path):
    records = tmp_path / "records-0.jsonl"
    rows = [record("a", "stock"), record("a", "wording", "infrastructure", None, infra_error="RuntimeError: 429")]
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    junit = tmp_path / "junit-0.xml"
    junit.write_text('<testsuites><testsuite>'
                     '<testcase name="test_paired_reflect[update/a@stock]"/>'
                     '<testcase name="test_paired_reflect[update/a@wording]"><error/></testcase>'
                     '<testcase name="test_paired_reflect[update/a@counterfactual]"><error/></testcase>'
                     '</testsuite></testsuites>')
    assert paired_job.infra_failures(junit, records) == ["a@counterfactual", "a@wording"]

    retry = tmp_path / "records-0.retry.jsonl"
    retry.write_text(json.dumps(record("a", "wording", "generation", False)) + "\n")
    paired_job.splice_records(records, retry, {"a@wording"})
    merged = {(r["case_id"], r["arm"]): r for r in paired_job.read_records(records)}
    assert merged[("a", "wording")]["blame"] == "generation" and not retry.exists()


def test_infra_retry_reruns_only_broken_executions(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_job, "INFRA_RETRY_WAIT", 0)
    monkeypatch.delenv("HS_BENCH_CODEX_LB_URL", raising=False)
    records, junit, tokens = tmp_path / "records-0.jsonl", tmp_path / "junit-0.xml", tmp_path / "tokens-0.jsonl"
    records.write_text(json.dumps(record("a", "stock")) + "\n" + json.dumps(record("a", "wording", "infrastructure", None)) + "\n")
    junit.write_text("<testsuites><testsuite/></testsuites>")
    seen = []

    def rerun(retry_junit, retry_tokens, retry_records, keys):
        seen.append(keys)
        retry_junit.write_text('<testsuites><testsuite><testcase name="test_paired_reflect[update/a@wording]"/>'
                               "</testsuite></testsuites>")
        retry_records.write_text(json.dumps(record("a", "wording")) + "\n")

    spliced = []
    paired_job.retry_infra(0, junit, tokens, records, rerun, lambda *a: spliced.append(a[-1]))
    assert seen == [["a@wording"]] and spliced == [{"test_paired_reflect[update/a@wording]"}]
    assert all(r["blame"] == "ok" for r in paired_job.read_records(records))


def test_report_keeps_outputs_local_and_separates_infrastructure(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_job, "REPORT_DIR", tmp_path / "reports")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [record("a", "stock"), record("a", "counterfactual", "generation", False, stale=True),
            record("a", "wording", "infrastructure", None, infra_error="RuntimeError: codex-lb grading failed")]
    (run_dir / "records-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    job = {"fixture_set": "fs", "fixture_revision": "rev", "pinned_commit": "c", "settings": {}, "settings_digest": "d",
           "run_dir": "run", "started": "t"}
    out = paired_job.write_report("rid", job, run_dir, {}, 1, "stamp")
    text = out.read_text()
    assert "secret model text" not in text and "judge reasoning" not in text
    report = json.loads(text)
    assert report["arms"]["counterfactual"] == {"passed": 0, "graded": 1, "infrastructure": 0, "stale": 1, "pass_rate": 0.0}
    assert report["arms"]["wording"]["infrastructure"] == 1 and report["arms"]["wording"]["pass_rate"] is None
    assert report["executions"][2]["infra_error"] == "RuntimeError"


def test_cli_rejects_paired_flags_on_stock(tmp_path):
    for args in (["--case-ids", "x"], ["--state", str(tmp_path)], ["--suite", "paired", "--cases", "x"]):
        done = subprocess.run([sys.executable, "-m", "hindsight_benchmark.evals", "--model", "m", *args],
                              capture_output=True, text=True, cwd=RUNNER)
        assert done.returncode == 2, (args, done.stderr)


def test_paired_operations_are_limited_to_fixture_operations():
    assert set(paired_job.operations(None)) == {"retain", "reflect"}
    with pytest.raises(SystemExit):
        paired_job.operations("consolidation")
    args = types.SimpleNamespace(suite="custom", provider="p", model="m", decision=None, embedder="e", reranker="r")
    s = paired_job.effective_settings({"hindsight_commit": "c", "judge_model": "j"}, args, {"base_url": "u"}, 2,
                                      {"retain": []}, ["x"])
    assert s["arms"] == ["wording", "counterfactual"] and "api_key" not in json.dumps(s).lower()


def test_paired_summary_counts_flips_stale_and_infra():
    import importlib.util
    spec = importlib.util.spec_from_file_location("report_paired", RUNNER / "report_paired.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def e(arm, rep, passed, blame="ok", stale=None):
        return {"case_id": "hq-owner-latest", "arm": arm, "repeat": rep, "passed": passed, "blame": blame,
                "stale": stale, "infra_error": "RuntimeError" if passed is None else None}
    report = {"result_id": "local-qwen35-4b-off-sol62-high-paired-x", "fixture_revision": "r", "pinned_commit": "c" * 40,
              "executions": [e("stock", 0, True), e("wording", 0, True), e("counterfactual", 0, False, "generation", True),
                             e("stock", 1, True), e("wording", 1, None), e("counterfactual", 1, True, stale=False)]}
    text = mod.summarize([report])
    assert "| qwen35-4b-off | 100.0% (2/2) | 100.0% (1/1) | +0.0 pp (n=1) | 50.0% (1/2) | -50.0 pp (n=2) | 1/2 | 1 |" in text
    assert "| qwen35-4b-off | hq-owner-latest | counterfactual | 1 | pass | fail | generation |" in text
    assert "`hq-owner-latest@wording` repeat 2: RuntimeError" in text


def test_tally_maps_paired_cases_without_classname(tmp_path):
    from hindsight_benchmark.evals import tally
    junit = tmp_path / "junit-0.xml"
    junit.write_text('<testsuites><testsuite>'
                     '<testcase classname="" name="test_paired_retain[fidelity/in-text-correction@stock]"><failure/></testcase>'
                     '<testcase classname="src.hindsight_benchmark.paired.test_paired_reflect" '
                     'name="test_paired_reflect[update/hq-owner-latest@wording]"/>'
                     '</testsuite></testsuites>')
    tokens = tmp_path / "tokens-0.jsonl"
    tokens.write_text(json.dumps({"nodeid": "::test_paired_retain[fidelity/in-text-correction@stock]", "calls": 1,
                                  "input_tokens": 10, "cached_tokens": 0, "output_tokens": 2}) + "\n")
    ops = tally([junit], [tokens], paired_job.OPERATIONS)
    assert (ops["retain"]["failed"], ops["reflect"]["passed"], ops["retain"]["tokens"]["calls"]) == (1, 1, 1)
