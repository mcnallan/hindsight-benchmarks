import json
from types import SimpleNamespace

from run_local_quality import _effective_config


def test_native_medium_control_preserves_benchmark_settings():
    args = SimpleNamespace(retrieval_profile='bge-minilm',extra_body_profile='medium-thinking',
                           retain_base_url='http://isolated/v1',retain_model='candidate',
                           retain_concurrency=8,retain_max_completion_tokens=16384,strict_retain_schema=True)
    config = _effective_config(args,'test-retrieval','test-retain')
    for scope in ('RETAIN','REFLECT','CONSOLIDATION'):
        assert json.loads(config[f'HINDSIGHT_API_{scope}_LLM_EXTRA_BODY']) == {
            'chat_template_kwargs':{'enable_thinking':True,'medium_effort':True}}
    assert config['HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT'] == '8'
    assert config['HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS'] == '16384'
    assert config['HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN'] == 'true'
    assert config['HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL'] == 'BAAI/bge-small-en-v1.5'
