import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hindsight_benchmark.reflect import ReflectBenchmark, _save_reflect_result
import hindsight_benchmark.reflect as module
from run_local_reflect import validate_wire


def test_custom_strict_judge_controls_and_no_silent_fallback():
    benchmark = ReflectBenchmark(judge_base_url='http://localhost/v1', judge_api_key='synthetic',
                                 judge_model='gpt-6-sol', judge_reasoning_effort='medium', strict_judge=True)
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"reasoning":"Exact match","correct":true}'))]))
    benchmark.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert benchmark._judge_answer('Q', 'A', 'A') is True
    kwargs = create.call_args.kwargs
    assert kwargs['model'] == 'gpt-6-sol' and kwargs['reasoning_effort'] == 'medium'
    assert kwargs['response_format'] == {'type': 'json_object'}
    create.return_value.choices[0].message.content = '{"correct":"false"}'
    with pytest.raises(RuntimeError, match='fallback is disabled'):
        benchmark._judge_answer('Q', 'A', 'A')
    assert benchmark.judge_errors == 1


def test_smoke_trace_does_not_publish_or_overwrite_quality(tmp_path, monkeypatch):
    fixture = tmp_path / 'dataset.json'
    fixture.write_text(json.dumps([{'sample_id':'conv-43', 'qa':[
        {'question':'Q1','answer':'A1'}, {'question':'Q2','answer':'A2'}, {'question':'Q3','answer':'A3'}]}]))
    monkeypatch.setattr(ReflectBenchmark, 'DATASET_PATH', fixture)
    reflect = Mock(side_effect=[SimpleNamespace(text='A1'),SimpleNamespace(text='A2')])
    monkeypatch.setattr('hindsight_client.Hindsight', lambda **_: SimpleNamespace(reflect=reflect))
    save = Mock(); monkeypatch.setattr(module, '_save_reflect_result', save)
    bench = ReflectBenchmark()
    trace = tmp_path / 'questions.private.json'
    result = bench.run_with_bank('test','local','http://unused','fixed-bank',max_questions=2,
                                 no_save=True,question_trace_path=trace)
    assert result['total'] == 2 and result['correct'] == 2 and result['partial'] is True
    assert result['successful_calls'] == 2 and result['reflect_errors'] == 0
    assert all(call.kwargs['bank_id']=='fixed-bank' and call.kwargs['budget']=='low' for call in reflect.call_args_list)
    saved = json.loads(trace.read_text()); assert saved['complete'] and len(saved['questions']) == 2
    assert trace.stat().st_mode & 0o777 == 0o600
    save.assert_not_called()


def test_reflect_merge_keeps_existing_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'LEADERBOARD_DIR', tmp_path)
    path = tmp_path/'local-test.json';path.write_text(json.dumps({'quality':{'correct':28,'total':80}}))
    _save_reflect_result('local','test',{'correct':1,'total':2})
    result=json.loads(path.read_text());assert result['quality']=={'correct':28,'total':80}
    assert result['reflect']=={'correct':1,'total':2}


def test_wire_accepts_native_tool_calls_and_rejects_invalid_controls(tmp_path):
    path=tmp_path/'wire.ndjson'
    request={'model':'test','chat_template_kwargs':{'enable_thinking':False},'max_tokens':4096,
             'tools':[{'type':'function','function':{'name':'done'}}]}
    response={'choices':[{'finish_reason':'tool_calls','message':{'content':None,
              'tool_calls':[{'function':{'name':'done','arguments':'{"answer":"ready"}'}}]}}],
              'usage':{'prompt_tokens':100,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':64}}}
    records=[{'event':'request','request_id':'1','method':'POST','path':'/v1/chat/completions','body':request},
             {'event':'response','request_id':'1','status':200,'body':response}]
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    summary=validate_wire(path,'test');assert summary['error_count']==0 and summary['tool_names']==['done']
    assert summary['cached_prompt_tokens']==64
    request['chat_template_kwargs']['enable_thinking']=True
    response['choices'][0]['message']['tool_calls'][0]['function']['arguments']='not JSON'
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    summary=validate_wire(path,'test');assert summary['error_count']==2


def test_retain_wire_ignores_availability_probe_but_checks_real_extraction(tmp_path):
    path=tmp_path/'wire.ndjson'
    schema={'type':'object','properties':{'facts':{'type':'array'}},'required':['facts']}
    records=[
        {'event':'request','request_id':'probe','method':'POST','path':'/v1/chat/completions',
         'body':{'model':'gpt-6-sol','messages':[{'role':'user','content':"Say 'ok'"}]}},
        {'event':'response','request_id':'probe','status':200,
         'body':{'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}},
        {'event':'request','request_id':'extract','method':'POST','path':'/v1/chat/completions',
         'body':{'response_format':{'type':'json_schema','json_schema':{'schema':schema}}}},
        {'event':'response','request_id':'extract','status':200,
         'body':{'choices':[{'finish_reason':'stop','message':{'content':'{"facts":[]}'}}]}},
    ]
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    summary=validate_wire(path,retain=True)
    assert summary['responses']==1 and summary['ignored_probe_requests']==1 and summary['error_count']==0
    records[-1]['body']['choices'][0]['message']['content']='not facts JSON'
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    assert validate_wire(path,retain=True)['error_count']==1
