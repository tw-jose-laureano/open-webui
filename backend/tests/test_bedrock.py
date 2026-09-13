"""Tests for AWS Bedrock support in Open WebUI (open_webui/utils/bedrock.py).

All boto3/botocore calls are mocked — no AWS credentials or network required.
Run with:
    pytest backend/tests/test_bedrock.py -v
"""

import importlib
import importlib.util
import json
import sys
import types
from unittest.mock import MagicMock, patch


def _load_bedrock_module():
    """Load bedrock.py isolated from the rest of open_webui (avoids typer/config deps)."""
    # Stub out parent packages so open_webui/__init__.py is never executed
    for name in ('open_webui', 'open_webui.utils'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    spec = importlib.util.spec_from_file_location(
        'open_webui.utils.bedrock',
        'backend/open_webui/utils/bedrock.py',
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


b = _load_bedrock_module()


# ---------------------------------------------------------------------------
# has_aws_credentials
# ---------------------------------------------------------------------------

class TestHasAwsCredentials:
    def test_true_when_access_key_and_secret_set(self, monkeypatch):
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
        assert b.has_aws_credentials() is True

    def test_true_via_bearer_token(self, monkeypatch):
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'tok123')
        assert b.has_aws_credentials() is True

    def test_true_via_container_credentials_uri(self, monkeypatch):
        monkeypatch.setenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', '/v2/credentials/xyz')
        assert b.has_aws_credentials() is True

    def test_false_when_no_env_vars_and_no_iam(self, monkeypatch):
        for var in (
            'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_PROFILE',
            'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', 'AWS_WEB_IDENTITY_TOKEN_FILE',
            'AWS_BEARER_TOKEN_BEDROCK',
        ):
            monkeypatch.delenv(var, raising=False)
        # Patch the internal boto3 chain check directly — avoids requiring botocore installed
        with patch.object(b, '_boto3_chain_has_credentials', return_value=False):
            assert b.has_aws_credentials(env={}) is False

    def test_explicit_empty_env_dict(self):
        assert b.has_aws_credentials(env={}) is False


# ---------------------------------------------------------------------------
# resolve_aws_region
# ---------------------------------------------------------------------------

class TestResolveAwsRegion:
    def test_prefers_aws_region(self, monkeypatch):
        monkeypatch.setenv('AWS_REGION', 'eu-west-1')
        assert b.resolve_aws_region() == 'eu-west-1'

    def test_falls_back_to_default_region(self, monkeypatch):
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'ap-southeast-1')
        assert b.resolve_aws_region() == 'ap-southeast-1'

    def test_explicit_env_dict(self):
        assert b.resolve_aws_region(env={'AWS_REGION': 'us-west-2'}) == 'us-west-2'

    def test_final_fallback_is_us_east_1(self, monkeypatch):
        # When env dict has no region vars and botocore is unavailable (suppressed),
        # the function falls back to 'us-east-1'
        result = b.resolve_aws_region(env={})
        assert result == 'us-east-1'


# ---------------------------------------------------------------------------
# convert_messages_to_converse
# ---------------------------------------------------------------------------

class TestConvertMessagesToConverse:
    def test_basic_user_message(self):
        system, msgs = b.convert_messages_to_converse([
            {'role': 'user', 'content': 'Hello'},
        ])
        assert system is None
        assert msgs == [{'role': 'user', 'content': [{'text': 'Hello'}]}]

    def test_system_message_extracted(self):
        system, msgs = b.convert_messages_to_converse([
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hi'},
        ])
        assert system == [{'text': 'You are helpful.'}]
        assert msgs[0]['role'] == 'user'

    def test_tool_result_becomes_user_block(self):
        _, msgs = b.convert_messages_to_converse([
            {'role': 'user', 'content': 'call tool'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'tc1', 'function': {'name': 'fn', 'arguments': '{}'}}
            ]},
            {'role': 'tool', 'tool_call_id': 'tc1', 'content': 'result'},
        ])
        assert msgs[-1]['role'] == 'user'
        assert 'toolResult' in msgs[-1]['content'][0]

    def test_empty_text_replaced_with_placeholder(self):
        _, msgs = b.convert_messages_to_converse([
            {'role': 'user', 'content': '   '},
        ])
        assert msgs[0]['content'][0]['text'] == '(empty)'

    def test_strict_alternation_prepends_user_placeholder(self):
        # If messages start with assistant, a placeholder user turn is prepended
        _, msgs = b.convert_messages_to_converse([
            {'role': 'assistant', 'content': 'Hello'},
        ])
        assert msgs[0]['role'] == 'user'
        assert msgs[1]['role'] == 'assistant'

    def test_strict_alternation_appends_user_placeholder(self):
        # If messages end with assistant, a placeholder user turn is appended
        _, msgs = b.convert_messages_to_converse([
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello'},
        ])
        assert msgs[-1]['role'] == 'user'

    def test_same_role_turns_merged(self):
        _, msgs = b.convert_messages_to_converse([
            {'role': 'user', 'content': 'Part 1'},
            {'role': 'user', 'content': 'Part 2'},
        ])
        assert len(msgs) == 1
        assert len(msgs[0]['content']) == 2


# ---------------------------------------------------------------------------
# convert_tools_to_converse
# ---------------------------------------------------------------------------

class TestConvertToolsToConverse:
    def test_converts_openai_tool_to_toolspec(self):
        tools = [{
            'type': 'function',
            'function': {
                'name': 'search',
                'description': 'Search the web',
                'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}}},
            },
        }]
        result = b.convert_tools_to_converse(tools)
        assert len(result) == 1
        assert result[0]['toolSpec']['name'] == 'search'
        assert result[0]['toolSpec']['inputSchema']['json']['properties']['q']['type'] == 'string'

    def test_empty_tools_returns_empty_list(self):
        assert b.convert_tools_to_converse([]) == []
        assert b.convert_tools_to_converse(None) == []


# ---------------------------------------------------------------------------
# normalize_converse_response
# ---------------------------------------------------------------------------

class TestNormalizeConverseResponse:
    def test_basic_text_response(self):
        raw = {
            'output': {'message': {'content': [{'text': 'Hello!'}]}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 10, 'outputTokens': 5},
        }
        result = b.normalize_converse_response(raw)
        assert result['choices'][0]['message']['content'] == 'Hello!'
        assert result['choices'][0]['finish_reason'] == 'stop'
        assert result['usage']['prompt_tokens'] == 10
        assert result['usage']['completion_tokens'] == 5
        assert result['usage']['total_tokens'] == 15

    def test_tool_use_response(self):
        raw = {
            'output': {'message': {'content': [
                {'toolUse': {'toolUseId': 'tc1', 'name': 'search', 'input': {'q': 'hi'}}}
            ]}},
            'stopReason': 'tool_use',
            'usage': {'inputTokens': 20, 'outputTokens': 8},
        }
        result = b.normalize_converse_response(raw)
        assert result['choices'][0]['finish_reason'] == 'tool_calls'
        tc = result['choices'][0]['message']['tool_calls'][0]
        assert tc['function']['name'] == 'search'
        assert json.loads(tc['function']['arguments']) == {'q': 'hi'}

    def test_result_is_json_serialisable(self):
        raw = {
            'output': {'message': {'content': [{'text': 'ok'}]}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 1, 'outputTokens': 1},
        }
        result = b.normalize_converse_response(raw)
        # Must not raise
        json.dumps(result)

    def test_max_tokens_finish_reason(self):
        raw = {
            'output': {'message': {'content': [{'text': 'cut off'}]}},
            'stopReason': 'max_tokens',
            'usage': {'inputTokens': 5, 'outputTokens': 100},
        }
        result = b.normalize_converse_response(raw)
        assert result['choices'][0]['finish_reason'] == 'length'


# ---------------------------------------------------------------------------
# discover_bedrock_models
# ---------------------------------------------------------------------------

class TestDiscoverBedrockModels:
    def test_returns_active_streaming_text_models(self):
        b.reset_client_cache()
        b._discovery_cache.clear()
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {
                    'modelId': 'anthropic.claude-3-haiku-20240307-v1:0',
                    'modelName': 'Claude 3 Haiku',
                    'providerName': 'Anthropic',
                    'modelLifecycle': {'status': 'ACTIVE'},
                    'responseStreamingSupported': True,
                    'outputModalities': ['TEXT'],
                    'inputModalities': ['TEXT'],
                },
                {
                    # INACTIVE — should be filtered out
                    'modelId': 'old.model',
                    'modelName': 'Old Model',
                    'providerName': 'SomeProvider',
                    'modelLifecycle': {'status': 'LEGACY'},
                    'responseStreamingSupported': True,
                    'outputModalities': ['TEXT'],
                    'inputModalities': ['TEXT'],
                },
            ]
        }
        mock_client.list_inference_profiles.return_value = {'inferenceProfileSummaries': [], 'nextToken': None}

        with patch.object(b, 'get_bedrock_control_client', return_value=mock_client):
            models = b.discover_bedrock_models('us-east-1')

        assert len(models) == 1
        assert models[0]['id'] == 'anthropic.claude-3-haiku-20240307-v1:0'

    def test_inference_profiles_included(self):
        b.reset_client_cache()
        b._discovery_cache.clear()
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {'modelSummaries': []}
        mock_client.list_inference_profiles.return_value = {
            'inferenceProfileSummaries': [
                {
                    'inferenceProfileId': 'us.anthropic.claude-sonnet-4-6',
                    'inferenceProfileName': 'Claude Sonnet 4.6 (US)',
                    'status': 'ACTIVE',
                }
            ],
        }

        with patch.object(b, 'get_bedrock_control_client', return_value=mock_client):
            models = b.discover_bedrock_models('us-east-1')

        assert any(m['id'] == 'us.anthropic.claude-sonnet-4-6' for m in models)

    def test_results_cached(self):
        b.reset_client_cache()
        b._discovery_cache.clear()
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {'modelSummaries': []}
        mock_client.list_inference_profiles.return_value = {'inferenceProfileSummaries': []}

        with patch.object(b, 'get_bedrock_control_client', return_value=mock_client):
            b.discover_bedrock_models('us-east-1')
            b.discover_bedrock_models('us-east-1')

        # list_foundation_models called only once due to cache
        assert mock_client.list_foundation_models.call_count == 1

    def test_returns_empty_on_client_error(self):
        b.reset_client_cache()
        b._discovery_cache.clear()
        with patch.object(b, 'get_bedrock_control_client', side_effect=Exception('no creds')):
            models = b.discover_bedrock_models('us-east-1')
        assert models == []


# ---------------------------------------------------------------------------
# build_converse_kwargs
# ---------------------------------------------------------------------------

class TestBuildConverseKwargs:
    def test_basic_structure(self):
        kwargs = b.build_converse_kwargs(
            model='anthropic.claude-3-haiku',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        assert kwargs['modelId'] == 'anthropic.claude-3-haiku'
        assert 'messages' in kwargs
        assert kwargs['inferenceConfig']['maxTokens'] == 4096

    def test_system_block_included_when_present(self):
        kwargs = b.build_converse_kwargs(
            model='m',
            messages=[
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'hi'},
            ],
        )
        assert 'system' in kwargs
        assert kwargs['system'][0]['text'] == 'You are helpful.'

    def test_tools_included_when_provided(self):
        tools = [{'type': 'function', 'function': {'name': 'fn', 'description': 'd', 'parameters': {}}}]
        kwargs = b.build_converse_kwargs(model='m', messages=[{'role': 'user', 'content': 'hi'}], tools=tools)
        assert 'toolConfig' in kwargs
        assert kwargs['toolConfig']['tools'][0]['toolSpec']['name'] == 'fn'

    def test_no_inference_config_when_max_tokens_none(self):
        kwargs = b.build_converse_kwargs(model='m', messages=[{'role': 'user', 'content': 'hi'}], max_tokens=None)
        assert 'inferenceConfig' not in kwargs
