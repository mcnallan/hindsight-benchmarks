#!/usr/bin/env python3
"""Prepare private matched source pairs and a manual review worklist; never score fidelity."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from handoff_gpu1_quality import digest, digest_messages, trace_path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_pairs(path):
    """Join strict extraction records by ID; keep a hash of the complete line prefix."""
    requests, terminals = {}, {}
    prefix, size = hashlib.sha256(), 0
    with path.open('rb') as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.endswith(b'\n'):
                break
            prefix.update(raw); size += len(raw)
            event = json.loads(raw)
            rid = event['request_id']
            if event['event'] == 'request':
                body = event.get('body', {})
                if 'facts' not in body.get('response_format', {}).get('json_schema', {}).get('schema', {}).get('properties', {}):
                    continue
                if rid in requests:
                    raise ValueError('Duplicate strict request ID')
                requests[rid] = (number, event)
            elif event['event'] in ('response', 'error'):
                if rid in terminals:
                    raise ValueError('Duplicate terminal ID')
                terminals[rid] = (number, event)
    pairs = []
    for rid, (line, request) in requests.items():
        if rid not in terminals:
            continue
        response_line, response = terminals[rid]
        if response['event'] != 'response' or response.get('status') != 200:
            continue
        try:
            facts = json.loads(response['body']['choices'][0]['message']['content'])['facts']
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
        pairs.append({'request_id': rid, 'request_line': line, 'response_line': response_line,
                      'request': request, 'response': response, 'facts': facts,
                      'source_messages_sha256': digest_messages(request['body']['messages']),
                      'exact_pair_sha256': canonical_hash([request, response])})
    return pairs, {'trace_path': str(path.resolve()), 'trace_prefix_bytes': size,
                   'trace_prefix_sha256': prefix.hexdigest(),
                   'strict_pending_request_ids': sorted(set(requests) - set(terminals))}


def review_candidates(messages, facts):
    """Coarse review prompts only: mixed source text can legitimately trigger these."""
    source = '\n'.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
    roles = sorted(set(re.findall(r'"role"\s*:\s*"(user|assistant)"', source)))
    planned = bool(re.search(r'\b(plan(?:ned|s)?|propos(?:e|ed|al)|want|trying|will|should|ensure|recommend(?:ed)?)\b', source, re.I))
    flags = []
    version_pattern = re.compile(r'\b(Python|Flask(?:-Migrate)?|SQLite|Redis|Bootstrap)\s*([0-9]+(?:\.[0-9]+)+)', re.I)
    source_versions = {(name.lower(), version) for name, version in version_pattern.findall(source)}
    for i, fact in enumerate(facts, 1):
        text = str(fact.get('what', ''))
        if planned and re.search(r'\b(completed|installed|implemented|configured|deployed|finished|created|added)\b', text, re.I):
            flags.append({'fact_index': i, 'candidate': 'plan_or_proposal_source_with_completed_state_wording',
                          'caveat': 'Source may also contain confirmed completed work; compare this exact fact and actor before deciding.'})
        if roles == ['assistant'] and re.search(r'\buser\b', str(fact.get('who', '')), re.I) and fact.get('fact_type') != 'assistant':
            flags.append({'fact_index': i, 'candidate': 'assistant_source_attributed_to_user_world',
                          'caveat': 'Project context may legitimately concern the user; inspect example/proposal framing.'})
        for name, version in version_pattern.findall(text):
            if (name.lower(), version) not in source_versions:
                flags.append({'fact_index': i, 'candidate': 'version_not_literal_in_source',
                              'software': name, 'version': version})
        for date in sorted(set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', str(fact.get('when', ''))))):
            if date not in source:
                flags.append({'fact_index': i, 'candidate': 'date_requires_source_grounding_check', 'date': date,
                              'caveat': 'Could be a valid relative or month-name date conversion; not automatically a contradiction.'})
    return {'source_roles': roles, 'source_has_plan_or_proposal_wording': planned,
            'candidates': flags, 'semantic_result': 'pending'}


def build_reference(reference_run, fixtures=()):
    pairs, binding = read_pairs(trace_path(reference_run))
    by_id = {p['request_id']: p for p in pairs}
    report_path = reference_run / 'manual-source-review.json'
    report = json.loads(report_path.read_text())
    cases = []
    for i, review in enumerate(report['case_reviews'], 1):
        pair = by_id[review['request_id']]
        cases.append({'label': f'matched-source-{i}', 'reference_request_id': pair['request_id'],
                      'source_messages_sha256': pair['source_messages_sha256'],
                      'reference_exact_pair_sha256': pair['exact_pair_sha256'],
                      'reference_review': review})
    empty_path = report.get('empty_facts_review', {}).get('private_pairs_file')
    seen_empty = set()
    if empty_path:
        for example in json.loads(Path(empty_path).read_text()):
            pair = by_id[example.get('new_request_id', example.get('request_id'))]
            key = pair['source_messages_sha256']
            if key in seen_empty:
                continue
            seen_empty.add(key)
            cases.append({'label': f'reviewed-empty-source-{len(seen_empty)}',
                          'reference_request_id': pair['request_id'],
                          'source_messages_sha256': key,
                          'reference_exact_pair_sha256': pair['exact_pair_sha256'],
                          'reference_review': 'Previously reviewed reasonable empty array; current output still requires review.'})
    for label, path in fixtures:
        payload = json.loads(path.read_text())
        key = digest_messages(payload['messages'])
        matched = [p for p in pairs if p['source_messages_sha256'] == key]
        cases.append({'label': label, 'source_messages_sha256': key,
                      'fixture_path': str(path.resolve()), 'fixture_sha256': digest(path),
                      'reference_request_ids': [p['request_id'] for p in matched],
                      'reference_review': 'Supplemental frozen dense-plan source; actual current output requires manual review.'})
    return {'reference_run': str(reference_run.resolve()), 'reference_report_sha256': digest(report_path),
            'reference_trace_binding': binding, 'cases': cases}


def prepare(reference, run, output):
    pairs, binding = read_pairs(trace_path(run))
    selected, worklist = [], []
    for case in reference['cases']:
        matched = [p for p in pairs if p['source_messages_sha256'] == case['source_messages_sha256']]
        if not matched:
            worklist.append({'label': case['label'], 'state': 'no_completed_exact_source_match',
                             'semantic_result': 'pending'})
        for pair in matched:
            selected.append({**pair, 'reference_label': case['label']})
            worklist.append({'label': case['label'], 'state': 'completed_exact_source_match',
                             'request_id': pair['request_id'], 'fact_count': len(pair['facts']),
                             'exact_pair_sha256': pair['exact_pair_sha256'],
                             **review_candidates(pair['request']['body']['messages'], pair['facts'])})
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    def save(name, value):
        path = output / name
        path.write_text(json.dumps(value, indent=2) + '\n'); path.chmod(0o600)
        return path
    manifest = save('reference-manifest.private.json', reference)
    pair_file = save('source-pairs.private.json', selected)
    work_file = save('review-worklist.private.json', {'semantic_result': 'pending',
        'scope': 'Bounded matched source review; candidate flags are prompts, not findings or a fidelity score.',
        'cases': worklist})
    template = save('manual-source-review.template.json', {'semantic_result': 'pending',
        'scope': 'Bounded manual review of exact captured source pairs, plans/status, actor, dates, versions and selective omissions.',
        'case_reviews': [{'request_id': x['request_id'], 'label': x['label'],
                         'semantic_result': 'pending', 'dates': '', 'status': '', 'ownership': '',
                         'completeness': '', 'finding_or_caveat': ''} for x in worklist if 'request_id' in x],
        'production_admitted': False, 'full_admitted': False})
    binding.update(deployment_sha256=digest(run / 'deployment.json'),
                   reference_manifest_sha256=digest(manifest), source_pairs_sha256=digest(pair_file),
                   worklist_sha256=digest(work_file))
    save('trace-binding.json', binding)
    return {'output_directory': str(output.resolve()), 'matched_pairs': len(selected),
            'unmatched_cases': sum(x['state'] != 'completed_exact_source_match' for x in worklist),
            'semantic_result': 'pending', 'manual_report_template': str(template.resolve())}


def verify_bundle(run, output):
    binding = json.loads((output / 'trace-binding.json').read_text())
    if digest(run / 'deployment.json') != binding['deployment_sha256']:
        raise ValueError('Review bundle deployment changed')
    for name, field in [('reference-manifest.private.json', 'reference_manifest_sha256'),
                        ('source-pairs.private.json', 'source_pairs_sha256'),
                        ('review-worklist.private.json', 'worklist_sha256')]:
        if digest(output / name) != binding[field]:
            raise ValueError('Review bundle evidence changed')
    prefix = hashlib.sha256()
    remaining = binding['trace_prefix_bytes']
    with Path(binding['trace_path']).open('rb') as handle:
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('Review trace prefix truncated')
            prefix.update(block); remaining -= len(block)
    if prefix.hexdigest() != binding['trace_prefix_sha256']:
        raise ValueError('Review trace prefix changed')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extra-fixture', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.verify_only:
        verify_bundle(args.run_dir.resolve(), args.output_dir.resolve())
        print(json.dumps({'evidence_integrity_verified': True, 'semantic_result': 'pending'}))
        return
    if not args.reference_run:
        parser.error('--reference-run is required when preparing a bundle')
    fixtures = []
    for value in args.extra_fixture:
        label, separator, path = value.partition('=')
        if not separator or not label or not path:
            parser.error('--extra-fixture requires LABEL=PATH')
        fixtures.append((label, Path(path)))
    reference = build_reference(args.reference_run.resolve(), fixtures)
    print(json.dumps(prepare(reference, args.run_dir.resolve(), args.output_dir.resolve())))


if __name__ == '__main__':
    main()
