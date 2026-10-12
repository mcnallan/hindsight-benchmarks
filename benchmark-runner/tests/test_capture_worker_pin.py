import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_openai_proxy import CaptureProxy


@pytest.mark.parametrize("worker_id", [None, 42])
def test_capture_preserves_payload_and_auth_with_optional_explicit_placement(tmp_path, worker_id):
    observed = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            observed.append((body, dict(self.headers)))
            payload = b'{"choices":[]}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    class Proxy(CaptureProxy):
        pass
    Proxy.upstream = f'http://127.0.0.1:{upstream.server_port}'
    Proxy.output_path = tmp_path / 'wire.ndjson'
    Proxy.timeout = 5
    Proxy.dynamo_worker_id = worker_id
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    servers = [upstream, proxy]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    original = {'model': 'test', 'messages': [{'role': 'user', 'content': 'Keep this prompt'}],
                'nvext': {'other_option': True}, 'max_tokens': 16384,
                'response_format': {'type': 'json_schema'}, 'chat_template_kwargs': {'enable_thinking': False}}
    try:
        response = requests.post(f'http://127.0.0.1:{proxy.server_port}/v1/chat/completions',
                                 json=original, headers={'Authorization': 'Bearer synthetic-test-secret'}, timeout=5)
        response.raise_for_status()
        actual, headers = observed[0]
        expected = json.loads(json.dumps(original))
        if worker_id is not None:
            expected['nvext']['decode_worker_id'] = worker_id
            assert headers['x-dynamo-worker-instance-id'] == str(worker_id)
        else:
            assert 'x-dynamo-worker-instance-id' not in headers
        assert actual == expected
        assert headers['Authorization'] == 'Bearer synthetic-test-secret'
        records = [json.loads(line) for line in Proxy.output_path.read_text().splitlines()]
        assert records[0]['body'] == actual
        assert records[1]['status'] == 200
        assert 'synthetic-test-secret' not in Proxy.output_path.read_text()
        assert original['nvext'] == {'other_option': True}
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
        for t in threads:
            t.join(timeout=5)
