import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from hindsight_benchmark import quality

RUNNER = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model', ['gpt-5.6-luna', 'gpt-6-sol'])
def test_combined_selector_routes_both_real_call_shapes_to_same_model(monkeypatch, model):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get('response_format') == {'type': 'json_object'}:
            content = json.dumps({'correct': True, 'reasoning': 'Matches gold.'})
        elif kwargs['messages'][0]['content'] == 'Reply with the single word: ok':
            content = 'ok'
        else:
            content = 'Four.'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(quality, 'OpenAI', lambda **kwargs: client)
    benchmark = quality.QualityBenchmark(openai_base_url='http://fake/v1', openai_api_key='fake',
                                         openai_model=model, reasoning_effort='medium')
    benchmark.verify_benchmark_calls()
    assert [c['model'] for c in calls] == [model, model, model]
    generation, judge = calls[-2:]
    assert generation['temperature'] == 0.1 and 'response_format' not in generation
    assert judge['temperature'] == 0 and judge['response_format'] == {'type': 'json_object'}
    assert all(c['extra_body'] == {'reasoning_effort': 'medium'} for c in calls)
    assert 'No memories found.' in generation['messages'][0]['content']
    assert quality.JUDGE_PROMPT in judge['messages'][0]['content']
    assert 'Gold answer: Four.' in judge['messages'][0]['content']


def test_future_local_default_and_explicit_legacy_override():
    env = dict(os.environ); env.pop('QUALITY_JUDGE_MODEL', None)
    command = [sys.executable, '-c', 'import run_local_quality as q; print(q.JUDGE_MODEL)']
    default = subprocess.run(command, cwd=RUNNER, env=env, capture_output=True, text=True, check=True)
    assert default.stdout.strip() == 'gpt-6-sol'
    env['QUALITY_JUDGE_MODEL'] = 'gpt-5.6-luna'
    legacy = subprocess.run(command, cwd=RUNNER, env=env, capture_output=True, text=True, check=True)
    assert legacy.stdout.strip() == 'gpt-5.6-luna'


def test_sol_wrapper_id_is_separate_from_historical_luna_catalog():
    wrapper = (RUNNER / 'run_gpu1_quality.sh').read_text()
    prefix = wrapper[:wrapper.index(': "${CODEX_LB_TOKEN:')]
    env = dict(os.environ)
    for key in ('QUALITY_JUDGE_MODEL', 'QUALITY_LINEAR_ACTIVATIONS', 'QUALITY_KV_DTYPE',
                'QUALITY_RESULT_ID_SUFFIX', 'QUALITY_RETAIN_UPSTREAM_URL'):
        env.pop(key, None)
    command = ['bash', '-c', prefix + '\nprintf "%s\\n" "$RESULT_ID" "$QUALITY_JUDGE_MODEL"',
               'test', 'qwen36-35b-gsq-bf16kv', '/unused', 'smoke']
    default = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    sol_id, model = default.stdout.splitlines()
    assert model == 'gpt-6-sol' and sol_id.endswith('-qa-gpt-6-sol')
    env['QUALITY_JUDGE_MODEL'] = 'gpt-5.6-luna'
    legacy = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    luna_id, model = legacy.stdout.splitlines()
    assert model == 'gpt-5.6-luna' and luna_id != sol_id
    assert sol_id == luna_id + '-qa-gpt-6-sol'
    rows = json.loads((RUNNER / 'benchmark_models.json').read_text())['models']
    assert len([r for r in rows if r['provider_id'] == 'local' and r['model_id'] == sol_id]) == 1
    assert len([r for r in rows if r['provider_id'] == 'local' and r['model_id'] == luna_id]) == 1
