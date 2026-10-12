"""
Reflect benchmark — measures answer accuracy via Hindsight reflect() on LoComo conv-43.

The retain step uses a fixed model (gemini-2.5-flash) as a constant baseline.
The reflect step tests the model under evaluation via client.reflect().
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .benchmark import DATASETS_DIR, LEADERBOARD_DIR
from .locomo import parse_locomo_date, LOCOMO_JUDGE_PROMPT

JUDGE_MODEL = "gemini-2.5-flash-lite"


def _save_reflect_result(provider_id: str, model_id: str, result: dict):
    """Merge reflect result into the unified leaderboard file."""
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{provider_id}-{model_id}.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    existing["reflect"] = result
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path


class ReflectBenchmark:
    """Measures answer accuracy via Hindsight reflect() on LoComo conv-43.

    The retain/ingestion step always uses a constant model (gemini-2.5-flash).
    The reflect() call uses the model configured in the Hindsight instance
    (HINDSIGHT_API_REFLECT_LLM_* env vars).
    """

    DATASET_PATH = DATASETS_DIR / "locomo_quality.json"
    TARGET_CONVERSATION = "conv-43"

    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None, *,
                 judge_base_url: str = None, judge_api_key: str = None,
                 judge_model: str = None, judge_reasoning_effort: str = None,
                 strict_judge: bool = False):
        self.judge_reasoning_effort = judge_reasoning_effort
        self.strict_judge = strict_judge
        self.judge_errors = 0
        self.last_judge_reasoning = None
        if judge_base_url:
            if not judge_api_key or not judge_model:
                raise ValueError("Custom judge requires base URL, API key and model")
            self.llm_client = OpenAI(api_key=judge_api_key, base_url=judge_base_url, timeout=120.0)
            self.judge_model = judge_model
            print(f"Using {judge_model} for judge")
        elif gemini_api_key:
            print(f"Using {JUDGE_MODEL} for judge")
            self.llm_client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=120.0,
            )
            self.judge_model = JUDGE_MODEL
        elif openai_api_key:
            print("GEMINI_API_KEY not set, falling back to OpenAI gpt-4o-mini for judge")
            self.llm_client = OpenAI(api_key=openai_api_key, timeout=90.0)
            self.judge_model = "gpt-4o-mini"
        else:
            self.llm_client = None
            self.judge_model = None

    def run(self, model_id: str, provider_id: str, api_url: str, max_questions: Optional[int] = None) -> Dict[str, Any]:
        try:
            from hindsight_client import Hindsight
        except ImportError:
            print("Error: hindsight_client not installed.")
            sys.exit(1)

        with open(self.DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == self.TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {self.TARGET_CONVERSATION} not found in dataset")

        sample_id = item["sample_id"]
        print(f"\n=== Reflect Benchmark ===")
        print(f"Model: {provider_id}/{model_id}")
        print(f"Conversation: {sample_id}")
        print(f"Hindsight API: {api_url}")
        print(f"Retain LLM: gemini-2.5-flash (constant baseline)")
        print(f"Reflect LLM: {model_id} (model under test)")

        client = Hindsight(base_url=api_url)
        run_ts = int(time.time())
        bank_id = (
            f"rb_{provider_id}_{model_id}_{sample_id}_{run_ts}"
            .replace("/", "_").replace("-", "_").replace(".", "_")
        )
        print(f"\nCreating fresh bank: {bank_id}")
        client.create_bank(bank_id=bank_id)

        print("\nIngesting conversation (retain with gemini-2.5-flash)...")
        conv = item["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]
        session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))
        last_session_date = None

        for session_key in session_keys:
            if not isinstance(conv.get(session_key), list):
                continue
            session_date = parse_locomo_date(conv[f"{session_key}_date_time"])
            last_session_date = session_date
            client.retain(
                bank_id=bank_id,
                content=json.dumps(conv[session_key]),
                context=f"Conversation between {speaker_a} and {speaker_b} ({session_key})",
                timestamp=session_date,
                document_id=f"{sample_id}_{session_key}_{run_ts}",
            )
            print(f"  ✓ Ingested {session_key}", flush=True)

        qa_pairs = item["qa"][:max_questions] if max_questions else item["qa"]
        print(f"\nEvaluating {len(qa_pairs)} questions via reflect()...")

        correct = 0
        total_latency = 0.0
        successful_calls = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected = qa.get("answer", "You did not mention this information.")

            predicted = None
            latency = 0.0
            call_succeeded = False
            for attempt in range(2):
                try:
                    t0 = time.time()
                    reflect_response = client.reflect(
                        bank_id=bank_id,
                        query=question,
                        budget="low",
                    )
                    latency = time.time() - t0
                    predicted = reflect_response.text
                    call_succeeded = True
                    break
                except Exception as e:
                    err_str = str(e)
                    is_connection_err = "Connect call failed" in err_str or "Cannot connect" in err_str or "Connection refused" in err_str
                    if attempt < 1:
                        print(f"  Reflect attempt {attempt + 1} failed ({err_str[:80]}), retrying...", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  Reflect failed after 2 attempts: {err_str[:80]}", flush=True)
                        predicted = "Error: reflect call failed"
                        latency = 0.0
                        if is_connection_err:
                            consecutive_failures += 1

            if call_succeeded:
                consecutive_failures = 0

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n  ABORT: {consecutive_failures} consecutive connection failures — daemon likely crashed. Saving partial results.", flush=True)
                break

            if call_succeeded:
                total_latency += latency
                successful_calls += 1
            is_correct = self._judge_answer(question, expected, predicted)
            print(f"  {'✓' if is_correct else '✗'} Q{i} ({latency:.1f}s): {question[:60]}...", flush=True)
            if is_correct:
                correct += 1

        total = len(qa_pairs)
        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        # avg_latency_s is over successful calls only (failed calls have latency=0 and skew the average)
        avg_latency_s = round(total_latency / successful_calls, 3) if successful_calls > 0 else 0
        print(f"\n=== Results ===")
        print(f"Accuracy: {accuracy}% ({correct}/{total})")
        print(f"Avg latency: {avg_latency_s}s per question")

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_s": avg_latency_s,
            "model_id": model_id,
            "provider_id": provider_id,
            "sample_id": sample_id,
        }
        path = _save_reflect_result(provider_id, model_id, result)
        print(f"Results saved to {path}")
        return result

    def run_with_bank(self, model_id: str, provider_id: str, api_url: str, bank_id: str,
                      max_questions: Optional[int] = None, *, no_save: bool = False,
                      question_trace_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run reflect benchmark against an already-ingested shared bank (skips retain step)."""
        try:
            from hindsight_client import Hindsight
        except ImportError:
            print("Error: hindsight_client not installed.")
            sys.exit(1)

        with open(self.DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == self.TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {self.TARGET_CONVERSATION} not found in dataset")

        sample_id = item["sample_id"]
        print(f"\n=== Reflect Benchmark (shared bank) ===")
        print(f"Model: {provider_id}/{model_id}")
        print(f"Conversation: {sample_id}")
        print(f"Hindsight API: {api_url}")
        print(f"Bank: {bank_id} (pre-ingested, shared)")
        print(f"Reflect LLM: {model_id} (model under test)")

        client = Hindsight(base_url=api_url)
        qa_pairs = item["qa"][:max_questions] if max_questions else item["qa"]
        print(f"\nEvaluating {len(qa_pairs)} questions via reflect()...")

        correct = 0
        question_records = []
        judge_errors_before = self.judge_errors
        run_started = time.monotonic()

        def save_question_trace(complete=False, result=None):
            if question_trace_path is None:
                return
            path = Path(question_trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {"complete": complete, "questions": question_records}
            if result is not None:
                value["result"] = result
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.chmod(0o600)
            temporary.replace(path)
        total_latency = 0.0
        successful_calls = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected = qa.get("answer", "You did not mention this information.")

            predicted = None
            latency = 0.0
            call_succeeded = False
            for attempt in range(2):
                try:
                    t0 = time.time()
                    reflect_response = client.reflect(
                        bank_id=bank_id,
                        query=question,
                        budget="low",
                    )
                    latency = time.time() - t0
                    predicted = reflect_response.text
                    call_succeeded = True
                    break
                except Exception as e:
                    err_str = str(e)
                    is_connection_err = "Connect call failed" in err_str or "Cannot connect" in err_str or "Connection refused" in err_str
                    if attempt < 1:
                        print(f"  Reflect attempt {attempt + 1} failed ({err_str[:80]}), retrying...", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  Reflect failed after 2 attempts: {err_str[:80]}", flush=True)
                        predicted = "Error: reflect call failed"
                        latency = 0.0
                        if is_connection_err:
                            consecutive_failures += 1

            if call_succeeded:
                consecutive_failures = 0

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n  ABORT: {consecutive_failures} consecutive connection failures — daemon likely crashed. Saving partial results.", flush=True)
                break

            if call_succeeded:
                total_latency += latency
                successful_calls += 1
            is_correct = self._judge_answer(question, expected, predicted)
            print(f"  {'✓' if is_correct else '✗'} Q{i} ({latency:.1f}s): {question[:60]}...", flush=True)
            if is_correct:
                correct += 1
            question_records.append({"question_index": i, "question": question,
                                     "gold_answer": expected, "generated_answer": predicted,
                                     "call_succeeded": call_succeeded, "latency_s": latency,
                                     "correct": is_correct, "judge_reasoning": self.last_judge_reasoning})
            save_question_trace()

        total = len(question_records)
        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        avg_latency_s = round(total_latency / successful_calls, 3) if successful_calls > 0 else 0
        print(f"\n=== Results ===")
        print(f"Accuracy: {accuracy}% ({correct}/{total})")
        print(f"Avg latency: {avg_latency_s}s per question")

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_s": avg_latency_s,
            "model_id": model_id,
            "provider_id": provider_id,
            "sample_id": sample_id,
            "dataset": "locomo_conv43",
            "selected_questions": len(qa_pairs),
            "dataset_questions": len(item["qa"]),
            "partial": total != len(item["qa"]),
            "successful_calls": successful_calls,
            "reflect_errors": total - successful_calls,
            "judge_errors": self.judge_errors - judge_errors_before,
            "judge_model": self.judge_model,
            "judge_reasoning_effort": self.judge_reasoning_effort,
            "budget": "low",
            "elapsed_seconds": round(time.monotonic() - run_started, 3),
        }
        save_question_trace(complete=total == len(qa_pairs), result=result)
        if not no_save:
            path = _save_reflect_result(provider_id, model_id, result)
            print(f"Results saved to {path}")
        return result

    def _judge_answer(self, question: str, expected: str, predicted: str) -> bool:
        self.last_judge_reasoning = None
        if not self.llm_client:
            if self.strict_judge:
                raise RuntimeError("Strict judging requires a configured judge")
            return expected.lower() in predicted.lower()

        prompt = f"""{LOCOMO_JUDGE_PROMPT}

Question: {question}
Gold answer: {expected}
Generated answer: {predicted}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true or false}}"""

        try:
            kwargs = {"reasoning_effort": self.judge_reasoning_effort} if self.judge_reasoning_effort else {}
            response = self.llm_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                **kwargs,
            )
            data = json.loads(response.choices[0].message.content)
            if self.strict_judge and not isinstance(data.get("correct"), bool):
                raise ValueError("Judge must return a boolean correct field")
            self.last_judge_reasoning = data.get("reasoning")
            return bool(data.get("correct", False))
        except Exception as e:
            self.judge_errors += 1
            if self.strict_judge:
                raise RuntimeError("Reflect judge failed; string-match fallback is disabled") from e
            print(f"Warning: Judge failed ({e}), falling back to string matching", flush=True)
            return expected.lower() in predicted.lower()
