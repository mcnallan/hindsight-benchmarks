import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import hindsight_client
from hindsight_benchmark import quality

ABILITIES = ['information_extraction', 'knowledge_update', 'multi_session_reasoning']


def frozen_dataset():
    return json.loads(quality.QualityBenchmark.DATASET_PATH.read_text())


def test_selection_preserves_original_questions_gold_rubric_and_order():
    dataset = frozen_dataset()[:1]
    before = json.dumps(dataset)
    selected = quality.select_question_pairs(dataset, ABILITIES, 6)[dataset[0]['sample_id']]
    assert [i for i, _ in selected] == [7, 8, 11, 12, 13, 14]
    for i, qa in selected:
        assert qa is dataset[0]['qa'][i - 1]
        assert (qa['question'], qa.get('answer'), qa.get('rubric')) == (
            dataset[0]['qa'][i - 1]['question'], dataset[0]['qa'][i - 1].get('answer'),
            dataset[0]['qa'][i - 1].get('rubric'))
    assert json.dumps(dataset) == before


def test_default_prefix_and_unfiltered_full_remain_original():
    dataset = frozen_dataset()
    prefix = quality.select_question_pairs(dataset[:1], max_questions=2)
    assert next(iter(prefix.values())) == list(enumerate(dataset[0]['qa'][:2], 1))
    full = quality.select_question_pairs(dataset)
    assert len(full) == 4
    assert sum(map(len, full.values())) == 80
    assert all(full[item['sample_id']] == list(enumerate(item['qa'], 1)) for item in dataset)


def test_invalid_or_empty_selection_rejected():
    with pytest.raises(ValueError, match='Unsupported question abilities'):
        quality.select_question_pairs(frozen_dataset(), ['unsupported'])
    with pytest.raises(ValueError, match='No questions selected'):
        quality.select_question_pairs([{'sample_id': 'empty', 'qa': []}])
    with pytest.raises(ValueError, match='positive'):
        quality.select_question_pairs(frozen_dataset(), max_questions=0)


def test_wrapper_forwards_filter_only_to_smoke():
    wrapper = Path(__file__).resolve().parents[1] / 'run_gpu1_quality.sh'
    text = wrapper.read_text()
    start = text.index('if [[ $MODE != full ]]; then\n  SMOKE_ARGS=')
    end = text.index('run_phase full || exit $?\n', start) + len('run_phase full || exit $?\n')
    fragment = text[start:end]
    setup = "run_phase() { printf '%s\\n' \"$*\"; }; QUALITY_SMOKE_QUESTION_ABILITIES=information_extraction,knowledge_update; QUALITY_SMOKE_MAX_QUESTIONS=6; "
    result = subprocess.run(['bash', '-c', setup + 'MODE=auto; ' + fragment], check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == [
        'smoke --max-conversations 1 --max-questions 6 --no-save --question-abilities information_extraction,knowledge_update', 'full']
    result = subprocess.run(['bash', '-c', setup + 'MODE=full; ' + fragment], check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == ['full']
    result = subprocess.run(['bash', '-c', 'run_phase() { printf \'%s\\n\' "$*"; }; MODE=smoke; ' + fragment], check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == ['smoke --max-conversations 1 --max-questions 2 --no-save']


def run_fake_benchmark(monkeypatch, tmp_path, abort=False):
    fake = SimpleNamespace(get_version=lambda: SimpleNamespace(api_version='test'))
    monkeypatch.setattr(hindsight_client, 'Hindsight', lambda **kwargs: fake)
    monkeypatch.setattr(quality, '_count_stored_fact_tokens', lambda *args: 10)
    recall = SimpleNamespace(results=[SimpleNamespace(text='Recalled source fact', context='session', occurred_start=None, occurred_end=None, mentioned_at=None)])
    monkeypatch.setattr(quality, '_facts_only_recall', lambda *args: recall)
    monkeypatch.setattr(quality, '_save_quality_result', lambda *args: pytest.fail('No-save wrote leaderboard'))
    monkeypatch.setattr(quality, 'LEADERBOARD_DIR', tmp_path / 'leaderboard' / 'llm')
    bench = object.__new__(quality.QualityBenchmark)
    bench.model_name = 'fake-judge'; bench.judge_base_url = 'fake'; bench.reasoning_effort = None
    def generate(*args):
        if abort:
            bench.generation_errors = quality.MAX_LLM_ERRORS + 1
        return 'Complete generated answer'
    bench._generate_answer = generate
    def judge(*args):
        bench.last_judge_reasoning = 'Complete judge reasoning'
        return True
    bench._judge_answer = judge
    trace = tmp_path / 'private' / 'questions.json'
    kwargs = dict(model_id='test', provider_id='local', api_url='fake', max_conversations=1,
                  max_questions_per_conversation=6, question_abilities=ABILITIES,
                  question_trace_path=trace, save=False, reuse_bank_ts=123)
    return bench, trace, kwargs


def test_no_save_captures_full_question_evidence_without_leaderboard(monkeypatch, tmp_path):
    bench, trace, kwargs = run_fake_benchmark(monkeypatch, tmp_path)
    result = bench.run(**kwargs)
    captured = json.loads(trace.read_text())
    assert result['total'] == 6 and result['partial'] is True
    assert captured['complete'] is True
    assert [q['question_index'] for q in captured['questions']] == [7, 8, 11, 12, 13, 14]
    for q in captured['questions']:
        assert q['question'] and q['gold_answer'] and q['rubric']
        assert 'Recalled source fact' in q['retrieved_context']
        assert q['generated_answer'] == 'Complete generated answer'
        assert q['judge_reasoning'] == 'Complete judge reasoning'
    assert stat.S_IMODE(trace.stat().st_mode) == 0o600
    assert not (tmp_path / 'leaderboard').exists()


def test_completed_question_trace_survives_abort(monkeypatch, tmp_path):
    bench, trace, kwargs = run_fake_benchmark(monkeypatch, tmp_path, abort=True)
    with pytest.raises(RuntimeError, match='Aborting'):
        bench.run(**kwargs)
    captured = json.loads(trace.read_text())
    assert captured['complete'] is False
    assert len(captured['questions']) == 1
    assert captured['questions'][0]['generated_answer'] == 'Complete generated answer'


@pytest.mark.parametrize('kv,expected_dtype', [('bf16kv', 'bfloat16'), ('fp8kv', 'fp8_e4m3')])
def test_gsq_profile_dashboard_ids_and_native_off_match(kv, expected_dtype):
    runner = Path(__file__).resolve().parents[1]
    text = (runner / 'run_gpu1_quality.sh').read_text()
    prefix = text[:text.index(': "${CODEX_LB_TOKEN:')]
    import os
    env = dict(os.environ, QUALITY_JUDGE_MODEL='gpt-5.6-luna')
    for key in ('QUALITY_LINEAR_ACTIVATIONS', 'QUALITY_KV_DTYPE', 'QUALITY_RESULT_ID_SUFFIX', 'QUALITY_RETAIN_UPSTREAM_URL'):
        env.pop(key, None)
    profile = f'qwen36-35b-gsq-{kv}'
    result = subprocess.run(['bash', '-c', prefix + '\nprintf "%s\\n" "$MODEL" "$RESULT_ID" "$EXTRA_BODY_PROFILE" "$THINKING_MODE" "$UPSTREAM"',
                             'test', profile, '/unused', 'smoke'], env=env, capture_output=True, text=True, check=True)
    model, result_id, extra, thinking, upstream = result.stdout.splitlines()
    assert model == 'qwen3.6-35b-gsq-test'
    assert result_id == f'qwen3.6-35b-gsq-no-thinking-mixed-w2a16-w8a16-{kv}-bge-minilm'
    assert extra == 'no-thinking' and thinking == 'off'
    assert upstream == 'http://127.0.0.1:18137'
    rows = json.loads((runner / 'benchmark_models.json').read_text())['models']
    matching = [r for r in rows if r['provider_id'] == 'local' and r['model_id'] == result_id]
    assert len(matching) == 1 and matching[0]['api_model'] == model
    assert 'INT4' not in matching[0]['model_name'] and 'W2A16/W8A16' in matching[0]['model_name']
    # Explicit GSQ profiles reject Mellum's independently variable activation matrix.
    env.update(QUALITY_LINEAR_ACTIVATIONS='int8', QUALITY_KV_DTYPE=expected_dtype)
    bad = subprocess.run(['bash', '-c', prefix, 'test', profile, '/unused', 'smoke'], env=env, capture_output=True, text=True)
    assert bad.returncode == 2 and 'native BF16 activations' in bad.stderr
    env.pop('QUALITY_LINEAR_ACTIVATIONS'); env.pop('QUALITY_KV_DTYPE')
    auto = subprocess.run(['bash', '-c', prefix, 'test', profile, '/unused', 'auto'], env=env, capture_output=True, text=True)
    assert auto.returncode == 2 and 'manual source-review' in auto.stderr
