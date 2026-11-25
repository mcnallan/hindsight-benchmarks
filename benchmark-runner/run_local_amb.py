"""Run the pinned AMB memory suites on an isolated Hindsight server per split."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from hindsight_benchmark.codex_lb_judge import generate_json

RUNNER = Path(__file__).resolve().parent
AMB = RUNNER / ".hindsight" / "amb"
AMB_COMMIT = "01bd83f3fa5245d6be3986cfab81fdd828f26acd"
LIFEBENCH_EN_URL = (
    "https://raw.githubusercontent.com/1754955896/LifeBench/095fa48a169cb88488d56e73001c8ad6397d3a13"
    "/life_bench_data/locomo_format/our_en.json"
)
SUITES = (
    ("precisionmembench", "single-turn", "retrieval"),
    ("locomo-fixed", "locomo10", "agent"),
    ("locomo", "locomo10", "rag"),
    ("beam", "100k", "rag"),
    ("personamem", "32k", "rag"),
    ("lifebench", "en", "rag"),
    ("longmemeval", "s", "rag"),
)


def score(dataset_name: str, split: str, mode: str, output: Path, model: str, *, smoke: bool) -> None:
    from memory_bench.dataset import get_dataset
    from memory_bench.dataset._cache import dataset_cache_dir
    from memory_bench.judge import GeminiJudge
    from memory_bench.llm.base import LLM
    from memory_bench.memory import get_memory_provider
    from memory_bench.modes import get_mode
    from memory_bench.runner import EvalRunner

    class CodexLB(LLM):
        @property
        def model_id(self) -> str:
            effort = os.getenv("HS_BENCH_JUDGE_REASONING_EFFORT", "high")
            return f"codex-lb:{model}:{effort}"

        def generate(self, prompt, schema):
            return generate_json(model, prompt, {
                "type": "object", "properties": schema.properties,
                "required": schema.required, "additionalProperties": False,
            })

    class LocalRunner(EvalRunner):
        def _get_judge(self, dataset):
            return grader

    if dataset_name == "lifebench" and not os.getenv("LIFEBENCH_DATA_PATH"):
        # AMB's original URL moved when LifeBench reorganized the repository.
        cache = dataset_cache_dir("lifebench")
        target = cache / "our_en.json"
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                try:
                    with urlopen(LIFEBENCH_EN_URL, timeout=60) as source:
                        shutil.copyfileobj(source, tmp)
                    tmp.flush()
                    Path(tmp.name).replace(target)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

    llm = CodexLB()
    grader = GeminiJudge(llm=llm)  # Upstream scoring and rubrics, with a shared independent judge.
    result = LocalRunner(output_dir=output).run(
        dataset=get_dataset(dataset_name), split=split,
        memory=get_memory_provider("hindsight-http"), mode=get_mode(mode, llm=llm if mode == "rag" else None),
        query_limit=1 if smoke else None,
        description=f"Local Hindsight backend comparison, AMB {AMB_COMMIT}",
    )
    if result.total_queries == 0:
        raise RuntimeError("AMB selected no queries")
    summary = {k: v for k, v in vars(result).items() if k != "results"}
    summary["amb_commit"] = AMB_COMMIT
    pending = output / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pending.replace(output / "summary.json")
    print(f"{dataset_name}/{split} {mode}: {result.correct}/{result.total_queries}", flush=True)


def run(profile_dir: Path) -> None:
    from hindsight_system_evals.server import start_eval_server

    config = json.loads((profile_dir / "evals.json").read_text())
    [(model, settings)] = [(name, value) for name, value in config["model_settings"].items()
                           if "RETAIN_MAX_COMPLETION_TOKENS" in value]
    provider = config["providers"]["local-eval"]
    retrieval = config["providers"]["retrieval"]
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AMB, text=True).strip()
    if rev != AMB_COMMIT:
        raise RuntimeError(f"AMB must be checked out at {AMB_COMMIT}, found {rev}")

    for name in list(os.environ):
        if name.startswith(("HINDSIGHT_API_", "HINDSIGHT_EVAL_", "AMB_", "OMB_")):
            del os.environ[name]
    os.environ.update(
        HINDSIGHT_EVAL_LLM_PROVIDER=provider["hindsight_provider"],
        HINDSIGHT_EVAL_LLM_MODEL=model,
        HINDSIGHT_EVAL_LLM_BASE_URL=provider["base_url"],
        HINDSIGHT_EVAL_LLM_API_KEY=os.environ[provider["api_key_env"]],
        **{f"HINDSIGHT_EVAL_SET_{key}": value for key, value in settings.items()},
        HINDSIGHT_EVAL_SET_EMBEDDINGS_PROVIDER="openai",
        HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_MODEL="BAAI/bge-small-en-v1.5",
        HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_BASE_URL=retrieval["base_url"],
        HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_API_KEY=os.environ[retrieval["api_key_env"]],
        HINDSIGHT_EVAL_SET_RERANKER_PROVIDER="cohere",
        HINDSIGHT_EVAL_SET_RERANKER_COHERE_MODEL="cross-encoder/ms-marco-MiniLM-L6-v2",
        HINDSIGHT_EVAL_SET_RERANKER_COHERE_BASE_URL=retrieval["rerank_url"],
        HINDSIGHT_EVAL_SET_RERANKER_COHERE_API_KEY=os.environ[retrieval["api_key_env"]],
        HINDSIGHT_EVAL_SET_WORKER_MAX_SLOTS="4",
    )
    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("HINDSIGHT_EVAL_", "TARGET_API_KEY", "EVAL_RETRIEVAL_KEY", "INKLING_API_KEY"))}
    child_env.update(PYTHONPATH=str(RUNNER / "src"), AMB_CONCURRENCY="4", AMB_UNIT_CONCURRENCY="1")

    jobs = [("smoke", SUITES[0])] + [("full", suite) for suite in SUITES]
    for phase, (dataset, split, mode) in jobs:
        output = profile_dir / "amb" / (f"{dataset}-{split}" if phase == "full" else "smoke")
        output.mkdir(parents=True, exist_ok=True)
        if (output / "summary.json").exists():
            continue
        print(f"{phase} {dataset}/{split} {mode}: {output}", flush=True)
        server = start_eval_server(log_path=output / "hindsight.log")
        try:
            with (output / "amb.log").open("a") as log:
                result = subprocess.run(
                    [str(AMB / ".venv/bin/python"), __file__, "--score", dataset, split, mode, str(output),
                     config["judge_model"], *(["--smoke"] if phase == "smoke" else [])],
                    cwd=AMB, env=child_env | {"HINDSIGHT_HTTP_URL": server.url},
                    stdout=log, stderr=subprocess.STDOUT,
                )
        finally:
            server.stop()
        if result.returncode:
            message = f"AMB {phase} {dataset}/{split} exited {result.returncode}. See {output / 'amb.log'}"
            print(message, file=sys.stderr, flush=True)
            (output / "failed.txt").write_text(message + "\n")
        else:
            (output / "failed.txt").unlink(missing_ok=True)
            report = json.loads((output / "summary.json").read_text())
            print(f"{phase} {dataset}/{split}: {report['correct']}/{report['total_queries']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, nargs="?")
    parser.add_argument("--score", nargs=5, metavar=("DATASET", "SPLIT", "MODE", "OUTPUT", "MODEL"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.score:
        dataset, split, mode, output, model = args.score
        score(dataset, split, mode, Path(output), model, smoke=args.smoke)
    else:
        if args.profile_dir is None:
            parser.error("profile_dir is required")
        profile_dir = args.profile_dir.resolve()
        if not os.getenv("HS_BENCH_AMB_EVAL_ENV"):
            commit = json.loads((profile_dir / "evals.json").read_text())["hindsight_commit"]
            project = RUNNER / ".hindsight" / commit / "hindsight-system-evals"
            env = os.environ | {"PYTHONPATH": str(RUNNER / "src"), "HS_BENCH_AMB_EVAL_ENV": "1"}
            os.execvpe("uv", ["uv", "run", "--project", str(project), "python", __file__, str(profile_dir)], env)
        run(profile_dir)
