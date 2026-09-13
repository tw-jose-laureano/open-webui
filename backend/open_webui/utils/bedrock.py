"""AWS Bedrock Converse API support for Open WebUI.

Credential detection, region resolution, model discovery, OpenAI<->Converse format
conversion, and streaming response generation. Adapted from hermes-agent's
agent/bedrock_adapter.py.

boto3 is a hard dependency in open-webui's pyproject.toml so it is always available
in the standard Docker image. All boto3 client construction is deferred to first use
so this module can be imported without boto3 installed (tests, lean environments).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential / region detection
# ---------------------------------------------------------------------------

# Priority order: first group whose vars are ALL non-empty names the auth source.
_AWS_AUTH_ENV_CHAIN: Tuple[Tuple[str, ...], ...] = (
    ('AWS_BEARER_TOKEN_BEDROCK',),
    ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'),
    ('AWS_PROFILE',),
    ('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',),
    ('AWS_WEB_IDENTITY_TOKEN_FILE',),
)


def _boto3_chain_has_credentials() -> bool:
    """True if boto3's default credential chain resolves credentials (IMDS, task role, ...)."""
    with suppress(Exception):
        import botocore.session
        credentials = botocore.session.get_session().get_credentials()
        resolved = credentials.get_frozen_credentials() if credentials is not None else None
        return bool(resolved and resolved.access_key)
    return False


def has_aws_credentials(env: Optional[Mapping[str, str]] = None) -> bool:
    """True if any AWS credential source is available.

    Checks env-var groups first (fast, no I/O), then boto3's credential chain
    (covers EC2/ECS/EKS instance/task roles via IMDS).
    """
    env = env if env is not None else os.environ
    for group in _AWS_AUTH_ENV_CHAIN:
        if all(env.get(var, '').strip() for var in group):
            return True
    return _boto3_chain_has_credentials()


def resolve_aws_region(env: Optional[Mapping[str, str]] = None) -> str:
    """AWS_REGION → AWS_DEFAULT_REGION → botocore config region → 'us-east-1'."""
    env = env if env is not None else os.environ
    explicit = env.get('AWS_REGION', '').strip() or env.get('AWS_DEFAULT_REGION', '').strip()
    if explicit:
        return explicit
    with suppress(Exception):
        import botocore.session
        region = botocore.session.get_session().get_config_variable('region')
        if region:
            return region
    return 'us-east-1'


# ---------------------------------------------------------------------------
# boto3 client cache (per-region singletons)
# ---------------------------------------------------------------------------

_runtime_client_cache: Dict[str, Any] = {}
_control_client_cache: Dict[str, Any] = {}


def get_bedrock_runtime_client(region: str) -> Any:
    """Return a cached boto3 bedrock-runtime client for *region*."""
    if region not in _runtime_client_cache:
        import boto3
        _runtime_client_cache[region] = boto3.client('bedrock-runtime', region_name=region)
    return _runtime_client_cache[region]


def get_bedrock_control_client(region: str) -> Any:
    """Return a cached boto3 bedrock (control-plane) client for *region*."""
    if region not in _control_client_cache:
        import boto3
        _control_client_cache[region] = boto3.client('bedrock', region_name=region)
    return _control_client_cache[region]


def reset_client_cache() -> None:
    """Clear all cached boto3 clients. Used in tests and after credential changes."""
    _runtime_client_cache.clear()
    _control_client_cache.clear()


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_discovery_cache: Dict[str, Any] = {}
_DISCOVERY_CACHE_TTL = 3600  # seconds


def discover_bedrock_models(region: str) -> List[Dict[str, Any]]:
    """Return active, streaming, text-output foundation models + inference profiles.

    Results are cached per region for one hour. Returns [] on any error so callers
    can degrade gracefully.
    """
    cache_key = region
    cached = _discovery_cache.get(cache_key)
    if cached and (time.time() - cached['ts']) < _DISCOVERY_CACHE_TTL:
        return cached['models']

    models: List[Dict[str, Any]] = []
    try:
        client = get_bedrock_control_client(region)
    except Exception as exc:
        log.warning('bedrock: failed to create control client for %s: %s', region, exc)
        return []

    # Foundation models
    try:
        for summary in client.list_foundation_models().get('modelSummaries', []):
            model_id = (summary.get('modelId') or '').strip()
            if not model_id:
                continue
            output_mods = summary.get('outputModalities', [])
            lifecycle = summary.get('modelLifecycle', {}).get('status', '').upper()
            if (
                lifecycle != 'ACTIVE'
                or not summary.get('responseStreamingSupported', False)
                or 'TEXT' not in output_mods
            ):
                continue
            models.append({
                'id': model_id,
                'name': (summary.get('modelName') or model_id).strip(),
                'provider': (summary.get('providerName') or '').strip(),
                'input_modalities': summary.get('inputModalities', []),
                'output_modalities': output_mods,
            })
    except Exception as exc:
        log.warning('bedrock: list_foundation_models failed: %s', exc)

    # Inference profiles (cross-region)
    try:
        seen_ids = {m['id'].lower() for m in models}
        next_token = None
        while True:
            kwargs: Dict[str, Any] = {}
            if next_token:
                kwargs['nextToken'] = next_token
            resp = client.list_inference_profiles(**kwargs)
            for profile in resp.get('inferenceProfileSummaries', []):
                pid = (profile.get('inferenceProfileId') or '').strip()
                if not pid or profile.get('status') != 'ACTIVE' or pid.lower() in seen_ids:
                    continue
                models.append({
                    'id': pid,
                    'name': (profile.get('inferenceProfileName') or pid).strip(),
                    'provider': 'inference-profile',
                    'input_modalities': ['TEXT'],
                    'output_modalities': ['TEXT'],
                })
                seen_ids.add(pid.lower())
            next_token = resp.get('nextToken')
            if not next_token:
                break
    except Exception as exc:
        log.debug('bedrock: list_inference_profiles skipped: %s', exc)

    # Sort: global.* profiles first, then alphabetically by name
    models.sort(key=lambda m: (0 if m['id'].startswith('global.') else 1, m['name'].lower()))

    _discovery_cache[cache_key] = {'ts': time.time(), 'models': models}
    return models


# ---------------------------------------------------------------------------
# OpenAI → Bedrock Converse format conversion
# ---------------------------------------------------------------------------

_EMPTY_TEXT_PLACEHOLDER = '(empty)'
_PLACEHOLDER_BLOCK = {'text': _EMPTY_TEXT_PLACEHOLDER}


def _safe_text(text: Any) -> str:
    """Return text if it has non-whitespace content, else the placeholder."""
    text = '' if text is None else str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


def _image_block_from_data_url(url: str) -> Dict[str, Any]:
    """data:<mime>;base64,... → Converse image block with raw bytes."""
    import base64
    header, _, data = url.partition(',')
    media_type = (header[5:].split(';')[0] if header.startswith('data:') else '') or 'image/jpeg'
    try:
        raw_bytes = base64.b64decode(data)
    except Exception:
        raw_bytes = data.encode('utf-8')
    fmt = media_type.split('/')[-1] if '/' in media_type else 'jpeg'
    return {'image': {'format': fmt, 'source': {'bytes': raw_bytes}}}


def _convert_content_to_converse(content: Any) -> List[Dict[str, Any]]:
    """OpenAI user content → Converse blocks."""
    if not isinstance(content, list):
        return [{'text': _safe_text(content)}]
    blocks = []
    for part in content:
        if isinstance(part, str):
            blocks.append({'text': _safe_text(part)})
        elif isinstance(part, dict) and part.get('type') == 'text':
            blocks.append({'text': _safe_text(part.get('text', ''))})
        elif isinstance(part, dict) and part.get('type') == 'image_url':
            image_url = part.get('image_url', {})
            url_str = image_url.get('url', '') if isinstance(image_url, dict) else ''
            if url_str.startswith('data:'):
                blocks.append(_image_block_from_data_url(url_str))
            else:
                blocks.append({'text': f'[Image: {url_str}]'})
    return blocks or [dict(_PLACEHOLDER_BLOCK)]


def _system_blocks(content: Any) -> List[Dict[str, Any]]:
    """System content → text blocks (empty parts dropped)."""
    parts = [content] if isinstance(content, str) else (content if isinstance(content, list) else [])
    texts = [
        part.get('text', '') if isinstance(part, dict) and part.get('type') == 'text' else part
        for part in parts
    ]
    return [{'text': text} for text in texts if isinstance(text, str) and text.strip()]


def _parse_tool_args(args: Any) -> Any:
    """JSON-decode a tool argument string; {} on failure."""
    try:
        return json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return {}


def _assistant_blocks(msg: Dict[str, Any], content: Any) -> List[Dict[str, Any]]:
    """Assistant message → Converse blocks (text + tool_calls)."""
    blocks: List[Dict[str, Any]] = []
    if isinstance(content, str) and content.strip():
        blocks.append({'text': content})
    elif isinstance(content, list):
        blocks.extend(_convert_content_to_converse(content))
    for tc in (msg.get('tool_calls') or []):
        fn = tc.get('function', {})
        blocks.append({
            'toolUse': {
                'toolUseId': tc.get('id', ''),
                'name': fn.get('name', ''),
                'input': _parse_tool_args(fn.get('arguments', '{}')),
            }
        })
    return blocks


def convert_messages_to_converse(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """OpenAI messages → (system_blocks_or_None, converse_messages).

    Enforces Converse's strict alternation (user/assistant, user first and last).
    Same-role consecutive turns are merged. tool messages become toolResult user blocks.
    """
    system_blocks: List[Dict[str, Any]] = []
    converse_msgs: List[Dict[str, Any]] = []

    def _append(role: str, blocks: List[Dict[str, Any]]) -> None:
        if converse_msgs and converse_msgs[-1]['role'] == role:
            converse_msgs[-1]['content'].extend(blocks)
        else:
            converse_msgs.append({'role': role, 'content': blocks})

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content')

        if role == 'system':
            system_blocks.extend(_system_blocks(content))
        elif role == 'tool':
            result_text = content if isinstance(content, str) else json.dumps(content)
            _append('user', [{
                'toolResult': {
                    'toolUseId': msg.get('tool_call_id', ''),
                    'content': [{'text': _safe_text(result_text)}],
                }
            }])
        elif role == 'assistant':
            blocks = _assistant_blocks(msg, content)
            _append('assistant', blocks or [dict(_PLACEHOLDER_BLOCK)])
        elif role == 'user':
            _append('user', _convert_content_to_converse(content))

    # Converse requires first and last message to be 'user'
    if converse_msgs and converse_msgs[0]['role'] != 'user':
        converse_msgs.insert(0, {'role': 'user', 'content': [dict(_PLACEHOLDER_BLOCK)]})
    if converse_msgs and converse_msgs[-1]['role'] != 'user':
        converse_msgs.append({'role': 'user', 'content': [dict(_PLACEHOLDER_BLOCK)]})

    return (system_blocks or None, converse_msgs)


def convert_tools_to_converse(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI tool defs → Converse toolSpec list."""
    result = []
    for t in tools or []:
        fn = t.get('function', {})
        result.append({
            'toolSpec': {
                'name': fn.get('name', ''),
                'description': fn.get('description', ''),
                'inputSchema': {'json': fn.get('parameters', {'type': 'object', 'properties': {}})},
            }
        })
    return result


# ---------------------------------------------------------------------------
# Build Converse kwargs
# ---------------------------------------------------------------------------

_STOP_REASON_TO_FINISH_REASON = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'tool_use': 'tool_calls',
    'max_tokens': 'length',
    'content_filtered': 'content_filter',
    'guardrail_intervened': 'content_filter',
}


def build_converse_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build kwargs dict for boto3 bedrock-runtime converse() / converse_stream()."""
    system_prompt, converse_messages = convert_messages_to_converse(messages)

    inference_config: Dict[str, Any] = {}
    if max_tokens is not None:
        inference_config['maxTokens'] = max_tokens
    if temperature is not None:
        inference_config['temperature'] = temperature
    if top_p is not None:
        inference_config['topP'] = top_p
    if stop_sequences:
        inference_config['stopSequences'] = stop_sequences

    kwargs: Dict[str, Any] = {
        'modelId': model,
        'messages': converse_messages,
    }
    if inference_config:
        kwargs['inferenceConfig'] = inference_config
    if system_prompt:
        kwargs['system'] = system_prompt

    if tools:
        converse_tools = convert_tools_to_converse(tools)
        if converse_tools:
            kwargs['toolConfig'] = {'tools': converse_tools}

    return kwargs


# ---------------------------------------------------------------------------
# Response normalisation: Bedrock Converse → OpenAI dict
# ---------------------------------------------------------------------------

def normalize_converse_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Bedrock Converse response → OpenAI-shaped dict (plain dict, JSON-serialisable).

    Shape: {choices: [{message: {role, content, tool_calls}, finish_reason}], usage: {...}}
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in response.get('output', {}).get('message', {}).get('content', []):
        if 'text' in block:
            text_parts.append(block['text'])
        elif 'toolUse' in block:
            tu = block['toolUse']
            tool_calls.append({
                'id': tu.get('toolUseId', ''),
                'type': 'function',
                'function': {
                    'name': tu.get('name', ''),
                    'arguments': json.dumps(tu.get('input', {})),
                },
            })

    content = '\n'.join(text_parts) if text_parts else None
    stop_reason = response.get('stopReason', 'end_turn')
    finish_reason = _STOP_REASON_TO_FINISH_REASON.get(stop_reason, 'stop')
    if tool_calls and finish_reason == 'stop':
        finish_reason = 'tool_calls'

    usage_raw = response.get('usage', {})
    prompt_tokens = usage_raw.get('inputTokens', 0)
    completion_tokens = usage_raw.get('outputTokens', 0)

    message: Dict[str, Any] = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls

    return {
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
        'model': response.get('modelId', ''),
        'object': 'chat.completion',
    }


# ---------------------------------------------------------------------------
# Streaming: Bedrock Converse → SSE text/event-stream
# ---------------------------------------------------------------------------

async def call_converse_stream(
    region: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
) -> AsyncIterator[str]:
    """Async generator yielding SSE lines from a Bedrock converse_stream call.

    Falls back to non-streaming converse() when InvokeModelWithResponseStream
    is denied by IAM (AccessDeniedException on the streaming action).
    """
    client = get_bedrock_runtime_client(region)
    kwargs = build_converse_kwargs(model, messages, tools, max_tokens, temperature)

    loop = asyncio.get_event_loop()

    # Try streaming first; fall back on access-denied
    try:
        response = await loop.run_in_executor(None, lambda: client.converse_stream(**kwargs))
    except Exception as exc:
        exc_str = str(exc).lower()
        if 'invokemodelwithresponsestream' in exc_str and (
            'accessdenied' in exc_str or 'not authorized' in exc_str
        ):
            log.info(
                'bedrock: converse_stream denied by IAM (region=%s, model=%s) — falling back to converse()',
                region, model,
            )
            raw = await loop.run_in_executor(None, lambda: client.converse(**kwargs))
            normalized = normalize_converse_response(raw)
            chunk = json.dumps(normalized)
            yield f'data: {chunk}\n\n'
            yield 'data: [DONE]\n\n'
            return
        raise

    # Walk the stream and emit SSE chunks
    current_tool: Optional[Dict[str, Any]] = None
    has_tool_use = False

    for event in response.get('stream', []):
        if 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})
            if 'toolUse' in start:
                has_tool_use = True
                current_tool = {
                    'id': start['toolUse'].get('toolUseId', ''),
                    'name': start['toolUse'].get('name', ''),
                    'input_json': '',
                }

        elif 'contentBlockDelta' in event:
            delta = event['contentBlockDelta'].get('delta', {})
            if 'text' in delta and not has_tool_use:
                chunk = {
                    'choices': [{'delta': {'content': delta['text']}, 'finish_reason': None}],
                    'object': 'chat.completion.chunk',
                }
                yield f'data: {json.dumps(chunk)}\n\n'
            elif 'toolUse' in delta and current_tool is not None:
                current_tool['input_json'] += delta['toolUse'].get('input', '')

        elif 'contentBlockStop' in event:
            if current_tool is not None:
                input_dict = _parse_tool_args(current_tool['input_json'])
                tool_call_chunk = {
                    'choices': [{
                        'delta': {
                            'tool_calls': [{
                                'index': 0,
                                'id': current_tool['id'],
                                'type': 'function',
                                'function': {
                                    'name': current_tool['name'],
                                    'arguments': json.dumps(input_dict),
                                },
                            }]
                        },
                        'finish_reason': None,
                    }],
                    'object': 'chat.completion.chunk',
                }
                yield f'data: {json.dumps(tool_call_chunk)}\n\n'
                current_tool = None

        elif 'messageStop' in event:
            stop_reason = event['messageStop'].get('stopReason', 'end_turn')
            finish = _STOP_REASON_TO_FINISH_REASON.get(stop_reason, 'stop')
            if has_tool_use and finish == 'stop':
                finish = 'tool_calls'
            chunk = {
                'choices': [{'delta': {}, 'finish_reason': finish}],
                'object': 'chat.completion.chunk',
            }
            yield f'data: {json.dumps(chunk)}\n\n'

    yield 'data: [DONE]\n\n'
