#!/usr/bin/env python3
"""Paired local reflect comparison on one fixed, privately captured memory bank."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

from dotenv import dotenv_values
from hindsight_client import Hindsight
from jsonschema import Draft202012Validator
from hindsight_benchmark.reflect import ReflectBenchmark, _save_reflect_result
from hindsight_benchmark.locomo import parse_locomo_date
from hindsight_benchmark.quality import _retain_with_polling, _count_stored_fact_tokens
from run_all_quality import BASE_CONFIG
from run_local_quality import BGE_MINILM_CONFIG, PRODUCTION_LIKE_CONFIG, _write_env_file, _model_ids

RUNNER = Path(os.environ.get('REFLECT_RUNNER_DIR', Path(__file__).resolve().parent))
MODELS = {'redhat': ('qwen35-redhat-w8a8', 'http://127.0.0.1:18142'),
          'production': ('qwen3.5-4b', 'http://10.10.5.1:8000')}
RESULT_IDS = {'redhat': 'qwen3.5-4b-redhat-w8a8-fp8kv-no-thinking-bge-minilm-qa-gpt-6-sol',
              'production': 'qwen3.5-4b-production-w4a8-fp8kv-no-thinking-bge-minilm-qa-gpt-6-sol'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def port():
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 0))
        return sock.getsockname()[1]


def validate_wire(path, model=None, retain=False, start_offset=0):
    pending, errors, names = {}, [], set()
    ignored_probes = 0
    responses = cached = prompt = completion = facts = 0
    handle = Path(path).open()
    handle.seek(start_offset)
    for raw in handle:
        record = json.loads(raw)
        rid = record['request_id']
        if record['event'] == 'request':
            body = record.get('body') or {}
            if record['method'] != 'POST' or not record['path'].endswith('/chat/completions'):
                continue
            if retain and 'facts' not in body.get('response_format', {}).get('json_schema', {}).get('schema', {}).get('properties', {}):
                ignored_probes += 1
                continue  # Startup availability probes are not extraction attempts.
            messages = body.get('messages', [])
            if not retain and len(messages) == 1 and messages[0].get('role') == 'user' and not body.get('tools') and not body.get('response_format'):
                ignored_probes += 1
                continue
            pending[rid] = body
            if model and body.get('model') != model:
                errors.append(f'{rid}: wrong model')
            if not retain and body.get('chat_template_kwargs', {}).get('enable_thinking') is not False:
                errors.append(f'{rid}: thinking control missing')
            if not retain and body.get('max_completion_tokens', body.get('max_tokens')) != 4096:
                errors.append(f'{rid}: unexpected reflect completion cap')
        elif rid in pending:
            request = pending.pop(rid)
            if record['event'] != 'response' or record.get('status') != 200:
                errors.append(f'{rid}: unsuccessful transport'); continue
            responses += 1
            body = record['body']; choice = body['choices'][0]; message = choice['message']
            if choice.get('finish_reason') not in ('stop', 'tool_calls'):
                errors.append(f'{rid}: finish_reason={choice.get("finish_reason")}')
            if not retain and (message.get('reasoning') or message.get('reasoning_content')):
                errors.append(f'{rid}: reasoning leakage')
            calls = message.get('tool_calls') or []
            for call in calls:
                function = call['function']; names.add(function['name'])
                offered = {tool.get('function', {}).get('name') for tool in request.get('tools', [])}
                if function['name'] not in offered:
                    errors.append(f'{rid}: unoffered tool')
                try: json.loads(function['arguments'])
                except (TypeError, ValueError): errors.append(f'{rid}: invalid tool JSON')
                if re.search(r'\s{256,}', function.get('arguments', '')):
                    errors.append(f'{rid}: tool whitespace loop')
            content = message.get('content')
            if content and re.search(r'\s{256,}', content):
                errors.append(f'{rid}: content whitespace loop')
            if retain:
                try:
                    data = json.loads(content)
                    schema = request['response_format']['json_schema']['schema']
                    Draft202012Validator(schema).validate(data)
                    facts += len(data['facts'])
                except Exception: errors.append(f'{rid}: invalid retain facts')
            elif not calls and not content:
                errors.append(f'{rid}: empty assistant response')
            usage = body.get('usage') or {}
            prompt += usage.get('prompt_tokens') or 0
            completion += usage.get('completion_tokens') or 0
            cached += (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    handle.close()
    errors.extend(f'{rid}: missing response' for rid in pending)
    if not responses: errors.append('No captured LLM responses')
    return {'responses': responses, 'facts': facts, 'prompt_tokens': prompt,
            'completion_tokens': completion, 'cached_prompt_tokens': cached,
            'tool_names': sorted(names), 'ignored_probe_requests': ignored_probes,
            'errors': errors, 'error_count': len(errors)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--redhat-provenance', type=Path, required=True)
    parser.add_argument('--production-provenance', type=Path, required=True)
    parser.add_argument('--prepared-bank', type=Path,
                        help='Reuse a completed, validated shared-bank dump and metadata; do not re-ingest')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / 'status.json').exists(): raise SystemExit('Preserve previous run; use a fresh directory')
    run.mkdir(parents=True, exist_ok=True); run.chmod(0o700)
    token = os.environ.get('CODEX_LB_TOKEN')
    if not token: raise SystemExit('CODEX_LB_TOKEN is required')
    retrieval_key = os.environ.get('QUALITY_RETRIEVAL_API_KEY') or dotenv_values(
        Path.home() / 'Projects/agents-mono/platform/hindsight/.env').get('HINDSIGHT_API_LLM_API_KEY')
    if not retrieval_key: raise SystemExit('Retrieval credential missing')
    from urllib.request import urlopen
    with urlopen(MODELS['production'][1] + '/health', timeout=15) as response:
        health = json.load(response)
    gpu2 = [x for x in health['instances'] if x.get('endpoint') == 'generate'
            and '192.168.39.245:' in str(x.get('transport'))]
    if len(gpu2) != 1: raise SystemExit('Require one healthy GPU2 generation worker')
    worker_id = gpu2[0]['instance_id']
    for name, (model, endpoint) in MODELS.items():
        if model not in _model_ids(endpoint + '/v1', 'EMPTY'):
            raise SystemExit(f'{name}: intended model missing')
    if 'gpt-6-sol' not in _model_ids('http://127.0.0.1:2455/v1', token):
        raise SystemExit('GPT-6 Sol missing')
    image_info = json.loads(subprocess.check_output(
        ['docker', 'image', 'inspect', 'ghcr.io/vectorize-io/hindsight:0.9.2'], text=True))[0]
    image = next(x for x in image_info['RepoDigests'] if 'hindsight@' in x)
    project = f'reflect-local-{int(time.time())}'
    state = {'stage': 'preflight', 'models': {}, 'project': project, 'gpu2_worker_id': worker_id}
    state_lock = threading.Lock()

    def save(**updates):
        with state_lock:
            state.update(updates)
            tmp = run / 'status.json.tmp'; tmp.write_text(json.dumps(state, indent=2) + '\n')
            tmp.replace(run / 'status.json')

    def judge():
        return ReflectBenchmark(judge_base_url='http://127.0.0.1:2455/v1', judge_api_key=token,
                                judge_model='gpt-6-sol', judge_reasoning_effort='medium', strict_judge=True)

    if not judge()._judge_answer('What color was explicitly stated?', 'blue', 'blue'):
        raise SystemExit('Reflect rubric/call-shape judge preflight failed')
    proxies = []; handles = []; env_files = []; compose_env = dict(os.environ)
    compose_ready = False
    compose_env.update(REFLECT_PROJECT=project, REFLECT_HINDSIGHT_IMAGE=image)
    common = {**BASE_CONFIG, **PRODUCTION_LIKE_CONFIG, **BGE_MINILM_CONFIG,
              'HINDSIGHT_API_LLM_PROVIDER': 'openai', 'HINDSIGHT_API_LLM_MODEL': 'gpt-6-sol',
              'HINDSIGHT_API_LLM_API_KEY': token, 'HINDSIGHT_API_LLM_REASONING_EFFORT': 'medium',
              'HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY': retrieval_key,
              'HINDSIGHT_API_RERANKER_COHERE_API_KEY': retrieval_key,
              'HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN': 'true',
              'HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT': '8',
              'HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS': '16384'}

    def proxy(name, upstream, worker=None):
        listen = port(); log = (run / f'proxy-{name}.log').open('w'); handles.append(log)
        cmd = [sys.executable, str(RUNNER / 'capture_openai_proxy.py'), '--listen-port', str(listen),
               '--upstream', upstream, '--output', str(run / f'wire-{name}.private.ndjson')]
        if worker is not None: cmd += ['--dynamo-worker-id', str(worker)]
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT); proxies.append(process)
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None: raise RuntimeError(f'{name}: capture process failed')
            try:
                with socket.create_connection(('127.0.0.1', listen), timeout=1): break
            except OSError:
                if time.monotonic() > deadline: raise RuntimeError(f'{name}: capture process not ready')
                time.sleep(0.1)
        return f'http://host.docker.internal:{listen}/v1'

    def compose(*cmd, **kwargs):
        return subprocess.run(['docker', 'compose', '-p', project, '-f', str(RUNNER / 'compose.local-reflect.yml'), *cmd],
                              env=compose_env, check=True, **kwargs)

    try:
        retain_proxy = proxy('retain', 'http://127.0.0.1:2455')
        common['HINDSIGHT_API_LLM_BASE_URL'] = retain_proxy
        for name in ['ingest', 'redhat', 'production']:
            config = dict(common)
            if name != 'ingest':
                model, endpoint = MODELS[name]
                reflect_proxy = proxy(name, endpoint, worker_id if name == 'production' else None)
                config.update(HINDSIGHT_API_REFLECT_LLM_PROVIDER='litellm',
                              HINDSIGHT_API_REFLECT_LLM_MODEL='openai/' + model,
                              HINDSIGHT_API_REFLECT_LLM_BASE_URL=reflect_proxy,
                              HINDSIGHT_API_REFLECT_LLM_API_KEY='EMPTY',
                              HINDSIGHT_API_REFLECT_LLM_TIMEOUT='600',
                              HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS='4096',
                              HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"max_completion_tokens":4096}')
                config.pop('HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT', None)
            file = _write_env_file(config); env_files.append(file)
            compose_env[f'REFLECT_{name.upper()}_ENV_FILE'] = str(file)
            compose_env[f'REFLECT_{name.upper()}_PORT'] = str(port())
        (run / 'deployment.json').write_text(json.dumps({
            'hindsight_image': image, 'retain_model': 'gpt-6-sol', 'retain_reasoning_effort': 'medium',
            'judge_model': 'gpt-6-sol', 'judge_reasoning_effort': 'medium', 'retrieval_profile': 'bge-minilm',
            'reflect_provider': 'litellm', 'reflect_thinking': False, 'reflect_completion_cap': 4096,
            'budget': 'low', 'dataset_sha256': digest(ReflectBenchmark.DATASET_PATH),
            'models': {name: json.loads(path.read_text()) for name, path in
                       [('redhat', args.redhat_provenance), ('production', args.production_provenance)]},
            'gpu2_worker_id': worker_id}, indent=2) + '\n')
        compose_ready = True
        save(stage='prepare_running')
        compose('config', '--quiet'); compose('up', '-d', '--wait', 'db')
        prepared = json.loads(args.prepared_bank.read_text()) if args.prepared_bank else None
        if prepared:
            if prepared['dataset_sha256'] != digest(ReflectBenchmark.DATASET_PATH) or prepared['retain_model'] != 'gpt-6-sol' or prepared['retrieval_profile'] != 'bge-minilm':
                raise RuntimeError('Prepared bank has different dataset/retain provenance')
            if digest(prepared['dump_path']) != prepared['dump_sha256']:
                raise RuntimeError('Prepared bank dump hash mismatch')
            db = subprocess.check_output(['docker', 'compose', '-p', project, '-f', str(RUNNER / 'compose.local-reflect.yml'),
                                          'ps', '-q', 'db'], env=compose_env, text=True).strip()
            with Path(prepared['dump_path']).open('rb') as source:
                subprocess.run(['docker', 'exec', '-i', db, 'pg_restore', '-U', 'benchmark', '-d', 'benchmark', '--exit-on-error'],
                               stdin=source, check=True)
        compose('up', '-d', '--wait', 'ingest')
        client = Hindsight(base_url='http://127.0.0.1:' + compose_env['REFLECT_INGEST_PORT'])
        dataset = json.loads(ReflectBenchmark.DATASET_PATH.read_text())
        item = next(x for x in dataset if x['sample_id'] == ReflectBenchmark.TARGET_CONVERSATION)
        bank = prepared['bank_id'] if prepared else 'reflect_shared_' + str(int(time.time()))
        if not prepared: client.create_bank(bank_id=bank)
        (run / 'bank-id.txt').write_text(bank + '\n')
        conv = item['conversation']; sessions = sorted(k for k,v in conv.items() if k.startswith('session_') and isinstance(v, list))
        for index, key in enumerate([] if prepared else sessions, 1):
            _retain_with_polling(client, bank, json.dumps(conv[key]),
                                 f"Conversation between {conv['speaker_a']} and {conv['speaker_b']} ({key})",
                                 parse_locomo_date(conv[f'{key}_date_time']), f'conv43_{key}', poll_interval_s=2)
            save(stage='prepare_running', ingested_sessions=index, total_sessions=len(sessions))
            print(f'Shared bank ingested {index}/{len(sessions)}', flush=True)
        tokens = _count_stored_fact_tokens(client, bank)
        if not tokens: raise RuntimeError('Shared bank contains no stored facts')
        db = subprocess.check_output(['docker', 'compose', '-p', project, '-f', str(RUNNER / 'compose.local-reflect.yml'),
                                      'ps', '-q', 'db'], env=compose_env, text=True).strip()
        with (run / 'shared-bank.pg.dump.private').open('wb') as output:
            subprocess.run(['docker', 'exec', db, 'pg_dump', '-U', 'benchmark', '-d', 'benchmark', '-Fc'], stdout=output, check=True)
        validation = prepared['retain_trace_summary'] if prepared else validate_wire(run / 'wire-retain.private.ndjson', retain=True)
        (run / 'retain-trace-summary.json').write_text(json.dumps(validation, indent=2) + '\n')
        if validation['error_count']: raise RuntimeError('Shared bank retain trace failed; preserve evidence')
        save(ingested_sessions=len(sessions), total_sessions=len(sessions))
        (run / 'prepared-bank.json').write_text(json.dumps({
            'bank_id': bank, 'retain_model': 'gpt-6-sol', 'retrieval_profile': 'bge-minilm',
            'dataset_sha256': digest(ReflectBenchmark.DATASET_PATH),
            'dump_path': str(run / 'shared-bank.pg.dump.private'),
            'dump_sha256': digest(run / 'shared-bank.pg.dump.private'),
            'retain_trace_summary': validation, 'sessions': len(sessions), 'stored_fact_tokens': tokens}, indent=2) + '\n')
        compose('stop', 'ingest'); compose('up', '-d', '--wait', 'redhat'); compose('up', '-d', '--wait', 'production')

        def phase(name, mode):
            directory = run / name; directory.mkdir(exist_ok=True)
            wire = run / f'wire-{name}.private.ndjson'
            start_offset = wire.stat().st_size if wire.exists() else 0
            result = judge().run_with_bank(RESULT_IDS[name], 'local',
                'http://127.0.0.1:' + compose_env[f'REFLECT_{name.upper()}_PORT'], bank,
                max_questions=2 if mode == 'smoke' else None, no_save=True,
                question_trace_path=directory / f'{mode}-questions.private.json')
            summary = validate_wire(wire, MODELS[name][0], start_offset=start_offset)
            (directory / f'{mode}-trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            result.update(retain_model='gpt-6-sol', shared_bank_id=bank, shared_bank_tokens=tokens,
                          retrieval_profile='bge-minilm', reflect_provider='litellm', thinking_enabled=False,
                          max_completion_tokens=4096, deployment=json.loads((run / 'deployment.json').read_text()),
                          wire_trace=str(wire), wire_start_offset=start_offset, trace_errors=summary['error_count'])
            if mode == 'full': _save_reflect_result('local', RESULT_IDS[name], result)
            with state_lock: state['models'][name] = {'phase': mode, 'result': result, 'trace_errors': summary['error_count']}
            save()
            if mode == 'smoke' and (result['reflect_errors'] or result['judge_errors'] or summary['error_count']):
                raise RuntimeError(f'{name}: reflect smoke failed')
            if mode == 'full' and result['partial']:
                raise RuntimeError(f'{name}: incomplete reflect evaluation')
            return result

        save(stage='smoke_running', shared_bank_tokens=tokens)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(phase, name, 'smoke') for name in MODELS]
            for future in futures: future.result()
        save(stage='awaiting_source_review')
        expected = {name: digest(run / name / 'smoke-questions.private.json') for name in MODELS}
        deadline = time.monotonic() + 3600
        while not (run / 'review-approved.json').exists():
            if time.monotonic() > deadline: raise RuntimeError('Source-review checkpoint not supplied; full unstarted')
            time.sleep(2)
        review = json.loads((run / 'review-approved.json').read_text())
        if review.get('smoke_sha256') != expected: raise RuntimeError('Source review does not bind these smokes')
        save(stage='full_running')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(phase, name, 'full') for name in MODELS]
            for future in futures: future.result()
        save(stage='complete_with_trace_errors' if any(x['trace_errors'] or x['result']['reflect_errors']
                                                     for x in state['models'].values()) else 'complete')
    except Exception as error:
        save(stage='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if compose_ready:
            logs = compose('logs', '--no-color', stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode(errors='replace')
            (run / 'hindsight.private.log').write_text(logs.replace(token, 'REDACTED').replace(retrieval_key, 'REDACTED'))
            compose('down', '-v', '--remove-orphans')
        for process in proxies:
            if process.poll() is None: process.terminate()
            process.wait(timeout=10)
        for handle in handles: handle.close()
        for file in env_files: file.unlink(missing_ok=True)


if __name__ == '__main__':
    os.umask(0o077)
    main()
