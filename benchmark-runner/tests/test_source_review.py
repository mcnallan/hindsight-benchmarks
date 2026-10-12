import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
spec = importlib.util.spec_from_file_location('source_review', RUNNER / 'prepare_source_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def request(rid, source='{"role": "assistant", "content": "Ensure dependencies installed; plan Flask 2.3.1 setup."}'):
    return {'event': 'request', 'request_id': rid, 'body': {
        'messages': [{'role': 'user', 'content': source}],
        'response_format': {'type': 'json_schema', 'json_schema': {'strict': True, 'schema': {'properties': {'facts': {}}}}}}}


def response(rid, facts=None):
    if facts is None:
        facts = [{'what': 'Dependencies installed; Flask 2.4.0 configured.', 'when': '2024-03-22', 'who': 'user', 'fact_type': 'world'}]
    return {'event': 'response', 'request_id': rid, 'status': 200,
            'body': {'choices': [{'message': {'content': json.dumps({'facts': facts})}}]}}


def write_events(path, events, tail=b''):
    path.write_bytes(b''.join((json.dumps(e) + '\n').encode() for e in events) + tail)


def test_join_ids_keeps_exact_source_and_facts_ignores_partial_tail(tmp_path):
    p = tmp_path / 'smoke.ndjson'
    events = [request('one'), request('two'), response('two', []), response('one'), request('pending')]
    write_events(p, events, b'{"incomplete":')
    pairs, binding = review.read_pairs(p)
    assert [x['request_id'] for x in pairs] == ['one', 'two']
    assert pairs[0]['request'] == events[0] and pairs[0]['response'] == events[3]
    assert pairs[1]['facts'] == []
    assert binding['strict_pending_request_ids'] == ['pending']
    assert binding['trace_prefix_bytes'] == len(p.read_bytes()) - len(b'{"incomplete":')
    assert pairs[0]['exact_pair_sha256'] == review.canonical_hash([events[0], events[3]])


def test_duplicate_request_or_terminal_is_rejected(tmp_path):
    p = tmp_path / 'trace'
    write_events(p, [request('one'), request('one')])
    with pytest.raises(ValueError, match='Duplicate strict request'):
        review.read_pairs(p)
    write_events(p, [request('one'), response('one'), response('one')])
    with pytest.raises(ValueError, match='Duplicate terminal'):
        review.read_pairs(p)


def test_candidate_flags_require_manual_review_not_a_score():
    messages = request('one')['body']['messages']
    flags = review.review_candidates(messages, response('one')['body'] and json.loads(response('one')['body']['choices'][0]['message']['content'])['facts'])
    assert flags['semantic_result'] == 'pending'
    assert {x['candidate'] for x in flags['candidates']} == {
        'plan_or_proposal_source_with_completed_state_wording', 'assistant_source_attributed_to_user_world',
        'version_not_literal_in_source', 'date_requires_source_grounding_check'}
    assert not any(k in flags for k in ('accuracy', 'fidelity_score', 'pass'))


def test_private_bundle_hashes_detect_tampering_and_allow_trace_append(tmp_path):
    run = tmp_path / 'run'; run.mkdir(); traces = tmp_path / 'traces'; traces.mkdir()
    (run / 'trace-directory.txt').write_text(str(traces)); (run / 'deployment.json').write_text('{}')
    trace = traces / 'smoke.ndjson'; write_events(trace, [request('one'), response('one')])
    pairs, _ = review.read_pairs(trace)
    reference = {'cases': [{'label': 'case', 'source_messages_sha256': pairs[0]['source_messages_sha256']},
                           {'label': 'missing', 'source_messages_sha256': 'different'}]}
    out = tmp_path / 'review'
    status = review.prepare(reference, run, out)
    assert status['matched_pairs'] == 1 and status['unmatched_cases'] == 1
    assert status['semantic_result'] == 'pending'
    for name in ('source-pairs.private.json', 'reference-manifest.private.json', 'manual-source-review.template.json'):
        assert stat.S_IMODE((out / name).stat().st_mode) == 0o600
    assert review.verify_bundle(run, out)
    with trace.open('ab') as handle:
        handle.write((json.dumps(request('later')) + '\n').encode())
    assert review.verify_bundle(run, out)
    original = (out / 'source-pairs.private.json').read_bytes()
    (out / 'source-pairs.private.json').write_text('[]')
    with pytest.raises(ValueError, match='evidence changed'):
        review.verify_bundle(run, out)
    (out / 'source-pairs.private.json').write_bytes(original)
    trace.write_bytes(trace.read_bytes().replace(b'Ensure', b'ALTER!', 1))
    with pytest.raises(ValueError, match='prefix changed'):
        review.verify_bundle(run, out)
