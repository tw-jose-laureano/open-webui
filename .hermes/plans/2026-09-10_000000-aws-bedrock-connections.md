# AWS Bedrock Connections Feature — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add AWS Bedrock as a first-class connection provider in Open WebUI, supporting
IAM credential chain auto-detection (instance roles, ECS task roles, etc.) AND explicit
env-var credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN), with
model discovery via the Bedrock control-plane API and request routing via boto3
`bedrock-runtime.converse_stream`.

**Architecture:**
Open WebUI already has a provider/auth_type extension point used for Azure OpenAI
(`provider: 'azure'`, `auth_type: 'microsoft_entra_id'`). We follow the same pattern:
add `provider: 'bedrock'` and `auth_type: 'aws_iam'` to the connection config schema,
add a new `BedrockClient` that wraps boto3 Converse API and translates to/from the
OpenAI message format the rest of the stack already speaks, and gate it at the two
critical path intersections: `get_headers_and_cookies` (auth) and
`generate_chat_completion` (request routing).

**Tech Stack:**
- Backend: Python 3.11, boto3 (already in pyproject.toml), botocore, FastAPI async
- Format conversion: lifted directly from hermes-agent `agent/bedrock_adapter.py`
- Frontend: SvelteKit 5 / Svelte 5, existing `AddConnectionModal.svelte` extended
- Tests: pytest-asyncio (already in dev deps)

---

## Background: How the Existing Auth Extension Point Works

Open WebUI connections are stored as entries in `openai.api_configs` (a dict keyed by
URL or index). Each entry is an `api_config` dict with these fields today:

```
auth_type: 'bearer' | 'none' | 'session' | 'system_oauth' | 'azure_ad' | 'microsoft_entra_id'
provider:  '' | 'azure' | 'llama.cpp' | 'lmstudio' | 'litellm'
```

The two functions that read these are:

1. `get_headers_and_cookies()` in `backend/open_webui/routers/openai.py:154`
   — builds HTTP headers/auth for upstream requests. We add a branch here.

2. `generate_chat_completion()` at line 1465 (and the Azure/Anthropic divergence
   block around line 1569) — rewrites the URL and payload per provider. We add a
   Bedrock branch that routes to boto3 instead of aiohttp.

3. `get_models()` and `get_all_models()` — fetch model lists. We add a Bedrock
   discovery branch that calls `bedrock.list_foundation_models` /
   `bedrock.list_inference_profiles`.

4. `verify_connection()` at line 1066 — validates a connection config. We add a
   Bedrock verification path.

---

## Credential Detection Logic (from hermes-agent)

hermes-agent's `agent/bedrock_adapter.py` defines the full priority chain:

```
Priority 1: AWS_BEARER_TOKEN_BEDROCK        (Bedrock-native API key)
Priority 2: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY  (explicit static key)
Priority 3: AWS_PROFILE                     (named profile / SSO)
Priority 4: AWS_CONTAINER_CREDENTIALS_RELATIVE_URI  (ECS / CodeBuild task role)
Priority 5: AWS_WEB_IDENTITY_TOKEN_FILE     (EKS IRSA)
Fallback:   boto3 credential chain (IMDS — EC2 instance role, Lambda, etc.)
```

For Open WebUI the relevant cases are:
- **Explicit keys** (admin pastes keys in the UI → stored server-side or as env vars)
- **IAM auto-detection** (Docker on EC2/ECS/EKS — boto3 resolves the instance/task role
  automatically when no explicit keys are present)

The `auth_type: 'aws_iam'` value covers both: the backend always delegates credential
resolution to boto3's default chain, which handles all the cases above in order.

---

## Files Changed

### Backend
- `backend/open_webui/routers/openai.py` — credential branch, chat routing, model
  discovery, connection verification
- `backend/open_webui/utils/bedrock.py` — new module: all Bedrock-specific logic
  (credential detection, model discovery, message format conversion, converse call)

### Frontend
- `src/lib/components/AddConnectionModal.svelte` — add `aws_iam` auth option and
  `bedrock` provider option, region input, optional key fields

### Tests
- `backend/tests/routers/test_bedrock_connection.py` — new

---

## Task 1: Create `backend/open_webui/utils/bedrock.py`

**Objective:** Single-module home for all Bedrock-specific logic, adapted from
hermes-agent's `agent/bedrock_adapter.py`. No other file changes.

**Files:**
- Create: `backend/open_webui/utils/bedrock.py`

**Step 1: Write the module**

The module must expose these public symbols:

```python
# Credential / region helpers
def has_aws_credentials(env=None) -> bool: ...
def resolve_aws_region(env=None) -> str: ...

# boto3 client (per-region singleton cache, cleared on each call to reset_client_cache)
def get_bedrock_runtime_client(region: str): ...
def get_bedrock_control_client(region: str): ...
def reset_client_cache() -> None: ...

# Model discovery
def discover_bedrock_models(region: str) -> list[dict]: ...

# OpenAI → Converse format conversion
def convert_messages_to_converse(messages: list[dict]) -> tuple[list|None, list[dict]]: ...
def convert_tools_to_converse(tools: list[dict]) -> list[dict]: ...

# Response conversion
def normalize_converse_response(response: dict) -> dict:
    """Returns OpenAI-shaped dict: {choices:[{message:{role,content,tool_calls},finish_reason}], usage:{...}}"""

# High-level call (streaming chunks yielded)
async def call_converse_stream(
    region: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> AsyncIterator[str]: ...   # yields SSE data: lines
```

The implementation is a cleaned-up port of hermes-agent's `agent/bedrock_adapter.py`.
Key pieces to copy:

- `_AWS_AUTH_ENV_CHAIN` tuple and `has_aws_credentials()` — detect whether any
  credential source is available before attempting boto3 import.

- `resolve_aws_region()` — reads `AWS_REGION` → `AWS_DEFAULT_REGION` → botocore
  config variable → `"us-east-1"`.

- `_cached_client()` — per-region client dict, lazy boto3 import so the module loads
  even if boto3 is not installed (fails gracefully on first use).

- `discover_bedrock_models()` — calls `list_foundation_models` and
  `list_inference_profiles`, filters to active streaming TEXT-output models, returns
  `[{"id": model_id, "name": model_name, "provider": provider_name}]`.
  Cache results 1 hour per region (use a simple dict + timestamp, same as
  hermes-agent's `_discovery_cache`).

- `convert_messages_to_converse()` — translates OpenAI message list to
  `(system_blocks_or_None, converse_messages)`. Important edge cases from hermes-agent:
    - tool messages become `toolResult` user blocks
    - assistant messages with tool_calls get `toolUse` blocks
    - empty text → `"(empty)"` placeholder (Bedrock rejects whitespace-only text)
    - strict user/assistant alternation: merge same-role consecutive turns, prepend
      placeholder user turn if first turn is assistant, append placeholder if last
      turn is assistant

- `convert_tools_to_converse()` — maps `{"function": {name, description, parameters}}`
  to `{"toolSpec": {name, description, inputSchema: {"json": parameters}}}`.

- `normalize_converse_response()` — maps Bedrock response back to the OpenAI
  `{choices, usage}` dict shape. Return a plain `dict` (not SimpleNamespace) to
  make JSON serialisation trivial.

- `call_converse_stream()` — async generator. Calls `boto3_client.converse_stream()`
  in a `ThreadPoolExecutor` (boto3 is sync), then walks the event stream yielding
  SSE-formatted lines (`data: {"choices":[{"delta":{"content":"..."}}]}\n\n`).
  Match the format Open WebUI's frontend already expects from OpenAI streaming
  responses (same event loop as the main FastAPI app — use `asyncio.to_thread` or
  `loop.run_in_executor`).

  Complete function skeleton:

  ```python
  async def call_converse_stream(region, model, messages, tools=None, max_tokens=4096, temperature=None):
      import asyncio, json
      client = get_bedrock_runtime_client(region)
      kwargs = _build_converse_kwargs(model, messages, tools, max_tokens, temperature)
      loop = asyncio.get_event_loop()
      response = await loop.run_in_executor(None, lambda: client.converse_stream(**kwargs))
      for event in response.get('stream', []):
          if 'contentBlockDelta' in event:
              delta = event['contentBlockDelta'].get('delta', {})
              if 'text' in delta:
                  chunk = {'choices': [{'delta': {'content': delta['text']}, 'finish_reason': None}]}
                  yield f'data: {json.dumps(chunk)}\n\n'
          elif 'messageStop' in event:
              stop_reason = event['messageStop'].get('stopReason', 'end_turn')
              finish = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length'}.get(stop_reason, 'stop')
              chunk = {'choices': [{'delta': {}, 'finish_reason': finish}]}
              yield f'data: {json.dumps(chunk)}\n\n'
      yield 'data: [DONE]\n\n'
  ```

**Step 2: Verify the module imports cleanly without boto3**

```bash
cd backend
python -c "from open_webui.utils.bedrock import has_aws_credentials; print('ok')"
```

Expected: `ok` (boto3 import is lazy — only happens when a client is created).

**Step 3: Commit**

```bash
git add backend/open_webui/utils/bedrock.py
git commit -m "feat(bedrock): add utils/bedrock.py — credential detection, format conversion, converse streaming"
```

---

## Task 2: Credential Detection in `get_headers_and_cookies`

**Objective:** When `auth_type == 'aws_iam'` the function returns empty headers
(boto3 signs at the socket level, not HTTP headers). This makes the existing aiohttp
path a no-op for Bedrock — the actual request goes through boto3 in Task 3.

**Files:**
- Modify: `backend/open_webui/routers/openai.py` around line 184 (the `auth_type`
  if/elif chain inside `get_headers_and_cookies`)

**Step 1: Add the branch**

In `get_headers_and_cookies()`, after the existing
`elif auth_type in ('azure_ad', 'microsoft_entra_id'):` block, add:

```python
elif auth_type == 'aws_iam':
    # AWS Bedrock: credentials resolved by boto3's default chain at call time.
    # No Authorization header is needed — boto3 signs each request via SigV4.
    token = None
```

**Step 2: Verify no regressions on existing auth types**

```bash
cd backend
python -m pytest tests/ -k "openai" -x -q 2>/dev/null || echo "no existing openai tests to run"
```

**Step 3: Commit**

```bash
git add backend/open_webui/routers/openai.py
git commit -m "feat(bedrock): skip Authorization header for auth_type=aws_iam"
```

---

## Task 3: Route Chat Completions to Bedrock Converse

**Objective:** In `generate_chat_completion`, detect `provider == 'bedrock'` and
route to `call_converse_stream` instead of forwarding to an aiohttp upstream.

**Files:**
- Modify: `backend/open_webui/routers/openai.py` around line 1569 (the
  `if api_config.get('azure') or api_config.get('provider') == 'azure':` block)

**Step 1: Add the import at the top of the file**

After the existing `from open_webui.utils.anthropic import ...` import, add:

```python
from open_webui.utils.bedrock import (
    call_converse_stream as bedrock_call_converse_stream,
    has_aws_credentials as bedrock_has_credentials,
    resolve_aws_region as bedrock_resolve_region,
)
```

**Step 2: Add the Bedrock branch in `generate_chat_completion`**

Find the block starting with:

```python
    if api_config.get('azure') or api_config.get('provider') == 'azure':
```

Before that block (or right after `headers, cookies = await get_headers_and_cookies(...)`),
add:

```python
    if api_config.get('provider') == 'bedrock':
        region = api_config.get('bedrock_region') or bedrock_resolve_region()
        model = payload.get('model', '')
        messages = payload.get('messages', [])
        tools = payload.get('tools', None)
        max_tokens = payload.get('max_tokens', 4096)
        temperature = payload.get('temperature', None)

        if payload.get('stream', False):
            return StreamingResponse(
                bedrock_call_converse_stream(
                    region=region,
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                media_type='text/event-stream',
            )
        else:
            # Non-streaming: collect the full response
            from open_webui.utils.bedrock import normalize_converse_response, get_bedrock_runtime_client
            from open_webui.utils.bedrock import convert_messages_to_converse, convert_tools_to_converse, _build_converse_kwargs
            import asyncio
            client = get_bedrock_runtime_client(region)
            kwargs = _build_converse_kwargs(model, messages, tools, max_tokens, temperature)
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, lambda: client.converse(**kwargs))
            return JSONResponse(content=normalize_converse_response(raw))
```

Note: `_build_converse_kwargs` is an internal helper in `utils/bedrock.py` that
assembles the full boto3 kwargs dict (model, messages, inferenceConfig, toolConfig,
system). Make it a module-level function (not private) so it can be imported here.

**Step 3: Commit**

```bash
git add backend/open_webui/routers/openai.py
git commit -m "feat(bedrock): route provider=bedrock chat completions to boto3 converse_stream"
```

---

## Task 4: Bedrock Model Discovery in `get_all_models`

**Objective:** When a connection has `provider == 'bedrock'`, fetch models from the
Bedrock control-plane API instead of calling `/models` on a URL.

**Files:**
- Modify: `backend/open_webui/routers/openai.py` — the `get_models_request()` helper
  or the per-connection branch inside `get_all_models()` around line 704.

**Step 1: Add import**

Add to the bedrock import block from Task 3:

```python
from open_webui.utils.bedrock import discover_bedrock_models as bedrock_discover_models
```

**Step 2: Add Bedrock branch in `get_all_models`**

Find the loop over API connections inside `get_all_models()` where
`request_tasks.append(get_models_request(...))` is called (around line 704).

Add a short-circuit before that append:

```python
            if api_config.get('provider') == 'bedrock':
                # Bedrock model discovery via control-plane API — no URL needed.
                region = api_config.get('bedrock_region') or bedrock_resolve_region()
                bedrock_models = bedrock_discover_models(region)
                prefix_id = api_config.get('prefix_id', None)
                for m in bedrock_models:
                    model_id = f'{prefix_id}.{m["id"]}' if prefix_id else m['id']
                    models_list.append({
                        'id': model_id,
                        'name': m.get('name', m['id']),
                        'object': 'model',
                        'urlIdx': idx,
                        'connection_type': 'external',
                    })
                continue  # skip the normal HTTP model-list request for this connection
```

(The exact variable names — `models_list`, `idx` — must match what the surrounding
loop uses; read lines 679–760 carefully before inserting.)

**Step 3: Verify models endpoint returns Bedrock models when provider=bedrock**

Manual test (after wiring up a valid connection in the UI or config):

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/models | jq '.data[] | select(.id | startswith("anthropic"))'
```

Expected: Claude model entries present.

**Step 4: Commit**

```bash
git add backend/open_webui/routers/openai.py
git commit -m "feat(bedrock): model discovery via control-plane API for provider=bedrock connections"
```

---

## Task 5: Connection Verification for Bedrock

**Objective:** The "Verify connection" button in the UI should work for Bedrock. It
currently calls `verify_connection()` which tries an HTTP GET to `/models`. For Bedrock
we substitute a quick `bedrock.list_foundation_models` call.

**Files:**
- Modify: `backend/open_webui/routers/openai.py` around line 1074
  (`verify_connection()`)

**Step 1: Add the Bedrock branch in `verify_connection`**

Find:

```python
    if api_config.get('azure') or api_config.get('provider') == 'azure':
```

Before that block, add:

```python
    if api_config.get('provider') == 'bedrock':
        from open_webui.utils.bedrock import discover_bedrock_models, resolve_aws_region
        try:
            region = api_config.get('bedrock_region') or resolve_aws_region()
            models = discover_bedrock_models(region)
            return {
                'status': True,
                'details': f'Connected to AWS Bedrock ({region}). Found {len(models)} models.',
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Bedrock connection failed: {e}')
```

**Step 2: Commit**

```bash
git add backend/open_webui/routers/openai.py
git commit -m "feat(bedrock): verify_connection branch for provider=bedrock"
```

---

## Task 6: Frontend — Add Bedrock Provider Option to `AddConnectionModal.svelte`

**Objective:** Users can select "AWS Bedrock" as a provider, pick a region, and
optionally supply explicit credentials. When IAM auto-detection is selected they
see a note explaining that no key is needed.

**Files:**
- Modify: `src/lib/components/AddConnectionModal.svelte`

**Step 1: Add `bedrock` to the provider select** (around line 608)

Existing:
```svelte
<option value="azure">{$i18n.t('Azure OpenAI')}</option>
<option value="llama.cpp">{$i18n.t('llama.cpp')}</option>
```

Add:
```svelte
<option value="bedrock">{$i18n.t('AWS Bedrock')}</option>
```

**Step 2: Add `aws_iam` to the auth_type select** (around line 422)

Existing options: `none`, `bearer`, `session`, `system_oauth`, `microsoft_entra_id`.

Add (inside the `{#if !ollama}` block, alongside `microsoft_entra_id`):
```svelte
{#if !direct}
    <option value="aws_iam">{$i18n.t('AWS IAM')}</option>
{/if}
```

**Step 3: Add the `aws_iam` description branch in the auth_type conditional** (around line 449)

Existing last branch:
```svelte
{:else if ['azure_ad', 'microsoft_entra_id'].includes(auth_type)}
    <div class={`text-xs self-center translate-y-[1px] text-gray-500`}>
        {$i18n.t('Uses DefaultAzureCredential to authenticate')}
    </div>
```

Add after it:
```svelte
{:else if auth_type === 'aws_iam'}
    <div class={`text-xs self-center translate-y-[1px] text-gray-500`}>
        {$i18n.t('Uses AWS credential chain (instance role, env vars, profile)')}
    </div>
```

**Step 4: Add Bedrock region input — shown when `provider === 'bedrock'`**

Find the `{#if azure}` block (around line 617) that shows the API Version input.
After the closing `{/if}` of that block, add a parallel block:

```svelte
{#if provider === 'bedrock'}
    <div class="flex gap-2 mt-2">
        <div class="flex flex-col w-full">
            <label for="bedrock-region-input" class={`mb-0.5 text-xs text-gray-500`}>
                {$i18n.t('AWS Region')}
            </label>
            <div class="flex-1">
                <input
                    id="bedrock-region-input"
                    class={`w-full text-sm ${inputClass}`}
                    type="text"
                    bind:value={bedrockRegion}
                    placeholder={$i18n.t('us-east-1')}
                    autocomplete="off"
                />
            </div>
        </div>
    </div>
{/if}
```

**Step 5: Declare the `bedrockRegion` state variable** (around line 48, with the other
`let` declarations)

```svelte
let bedrockRegion = '';
```

**Step 6: Include `bedrock_region` in the config object emitted by `onSubmit`**

Find the `onSubmit` handler where the `config` object is assembled (the object that
contains `auth_type`, `provider`, `azure`, etc.), and add:

```svelte
...(provider === 'bedrock' ? { bedrock_region: bedrockRegion } : {}),
```

**Step 7: Populate `bedrockRegion` when editing an existing connection** (in the
`onMount` / edit initialisation block around line 247):

```svelte
bedrockRegion = connection.config?.bedrock_region ?? '';
```

**Step 8: For Bedrock + aws_iam, make the URL field optional**

When `provider === 'bedrock'` the URL field is not used (boto3 builds the endpoint
from region). Prefill it with a placeholder and make it read-only or collapse it.
The simplest approach: if `provider === 'bedrock'` and `url` is empty, set
`url = 'bedrock'` before submit so the config key is stable.

In the submit handler, add at the top:

```svelte
if (provider === 'bedrock' && !url.trim()) {
    url = 'bedrock';
}
```

**Step 9: Auto-set `auth_type` when user picks `provider = 'bedrock'`**

Add a reactive statement:

```svelte
$: if (provider === 'bedrock' && auth_type === 'bearer') {
    auth_type = 'aws_iam';
}
```

**Step 10: Build and lint**

```bash
npm run check
npm run lint:frontend
```

Expected: no new errors.

**Step 11: Commit**

```bash
git add src/lib/components/AddConnectionModal.svelte
git commit -m "feat(bedrock): add AWS Bedrock provider option and AWS IAM auth type to AddConnectionModal"
```

---

## Task 7: Write Backend Tests

**Objective:** Verify the critical Bedrock backend paths without a live AWS account.

**Files:**
- Create: `backend/tests/routers/test_bedrock_connection.py`

**Step 1: Write tests**

```python
"""Tests for AWS Bedrock connection support in Open WebUI.

All boto3 calls are mocked — no AWS credentials required.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ---- utils/bedrock.py tests ----

class TestHasAwsCredentials:
    def test_true_when_key_id_and_secret_set(self, monkeypatch):
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'secret')
        from open_webui.utils.bedrock import has_aws_credentials
        assert has_aws_credentials() is True

    def test_false_when_no_env_and_boto3_unavailable(self, monkeypatch):
        for var in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_PROFILE',
                    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', 'AWS_WEB_IDENTITY_TOKEN_FILE',
                    'AWS_BEARER_TOKEN_BEDROCK'):
            monkeypatch.delenv(var, raising=False)
        with patch('botocore.session.get_session') as mock_session:
            mock_session.return_value.get_credentials.return_value = None
            from open_webui.utils import bedrock as b
            # Force re-evaluation
            assert b.has_aws_credentials(env={}) is False

    def test_true_via_container_credentials_uri(self, monkeypatch):
        monkeypatch.setenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', '/v2/credentials/xyz')
        from open_webui.utils.bedrock import has_aws_credentials
        assert has_aws_credentials() is True


class TestResolveAwsRegion:
    def test_prefers_aws_region(self, monkeypatch):
        monkeypatch.setenv('AWS_REGION', 'eu-west-1')
        from open_webui.utils.bedrock import resolve_aws_region
        assert resolve_aws_region() == 'eu-west-1'

    def test_falls_back_to_default_region(self, monkeypatch):
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'ap-southeast-1')
        from open_webui.utils.bedrock import resolve_aws_region
        assert resolve_aws_region() == 'ap-southeast-1'

    def test_final_fallback_us_east_1(self, monkeypatch):
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
        with patch('botocore.session.get_session') as mock_session:
            mock_session.return_value.get_config_variable.return_value = ''
            from open_webui.utils.bedrock import resolve_aws_region
            assert resolve_aws_region() == 'us-east-1'


class TestConvertMessagesToConverse:
    def test_basic_user_message(self):
        from open_webui.utils.bedrock import convert_messages_to_converse
        system, msgs = convert_messages_to_converse([
            {'role': 'user', 'content': 'Hello'}
        ])
        assert system is None
        assert msgs == [{'role': 'user', 'content': [{'text': 'Hello'}]}]

    def test_system_message_extracted(self):
        from open_webui.utils.bedrock import convert_messages_to_converse
        system, msgs = convert_messages_to_converse([
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hi'},
        ])
        assert system == [{'text': 'You are helpful.'}]
        assert msgs[0]['role'] == 'user'

    def test_tool_result_becomes_user_block(self):
        from open_webui.utils.bedrock import convert_messages_to_converse
        _, msgs = convert_messages_to_converse([
            {'role': 'user', 'content': 'call tool'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'tc1', 'function': {'name': 'fn', 'arguments': '{}'}}
            ]},
            {'role': 'tool', 'tool_call_id': 'tc1', 'content': 'result'},
        ])
        # Last message must be user (tool result wrapped)
        assert msgs[-1]['role'] == 'user'
        assert 'toolResult' in msgs[-1]['content'][0]

    def test_empty_text_replaced_with_placeholder(self):
        from open_webui.utils.bedrock import convert_messages_to_converse
        _, msgs = convert_messages_to_converse([
            {'role': 'user', 'content': '   '}
        ])
        assert msgs[0]['content'][0]['text'] == '(empty)'


class TestNormalizeConverseResponse:
    def test_basic_text_response(self):
        from open_webui.utils.bedrock import normalize_converse_response
        raw = {
            'output': {'message': {'content': [{'text': 'Hello!'}]}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 10, 'outputTokens': 5},
        }
        result = normalize_converse_response(raw)
        assert result['choices'][0]['message']['content'] == 'Hello!'
        assert result['choices'][0]['finish_reason'] == 'stop'
        assert result['usage']['prompt_tokens'] == 10
        assert result['usage']['completion_tokens'] == 5

    def test_tool_use_response(self):
        from open_webui.utils.bedrock import normalize_converse_response
        raw = {
            'output': {'message': {'content': [
                {'toolUse': {'toolUseId': 'tc1', 'name': 'search', 'input': {'q': 'hi'}}}
            ]}},
            'stopReason': 'tool_use',
            'usage': {'inputTokens': 20, 'outputTokens': 8},
        }
        result = normalize_converse_response(raw)
        assert result['choices'][0]['finish_reason'] == 'tool_calls'
        assert result['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'search'


class TestDiscoverBedrockModels:
    def test_returns_models_list(self):
        from open_webui.utils.bedrock import discover_bedrock_models, reset_client_cache
        reset_client_cache()
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
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = {'inferenceProfileSummaries': []}
        with patch('open_webui.utils.bedrock.get_bedrock_control_client', return_value=mock_client):
            models = discover_bedrock_models('us-east-1')
        assert len(models) == 1
        assert models[0]['id'] == 'anthropic.claude-3-haiku-20240307-v1:0'
```

**Step 2: Run tests**

```bash
cd backend
python -m pytest tests/routers/test_bedrock_connection.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add backend/tests/routers/test_bedrock_connection.py
git commit -m "test(bedrock): unit tests for utils/bedrock.py — credentials, format conversion, discovery"
```

---

## Task 8: Docker / Deployment Notes (README update)

**Objective:** Document how to run Open WebUI with Bedrock on EC2/ECS and with explicit
env vars.

**Files:**
- Modify: `TROUBLESHOOTING.md` (or `docs/bedrock.md` — new file)

**Content to add:**

```markdown
## AWS Bedrock

### With an IAM Instance Role (EC2 / ECS / EKS)

Attach an IAM role to the instance/task with at minimum:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:ListFoundationModels`
- `bedrock:ListInferenceProfiles`

Run the container with no extra env vars. Open WebUI detects the instance metadata
service automatically when you create a connection with Provider = "AWS Bedrock" and
Auth = "AWS IAM".

### With Explicit Credentials

Pass credentials as environment variables:

```
docker run -e AWS_ACCESS_KEY_ID=... \
           -e AWS_SECRET_ACCESS_KEY=... \
           -e AWS_DEFAULT_REGION=us-east-1 \
           ghcr.io/open-webui/open-webui:latest
```

Or use `AWS_PROFILE` with a mounted `~/.aws/credentials` file.

### Connection Setup

1. Admin panel → Settings → Connections → Add OpenAI-compatible connection
2. URL: leave blank (will be set to "bedrock" automatically)
3. Auth: AWS IAM
4. Provider: AWS Bedrock
5. Region: your AWS region (e.g. `us-east-1`)
6. Click "Verify" — should show the discovered model count
7. Save
```

**Step 1: Commit**

```bash
git add docs/bedrock.md   # or TROUBLESHOOTING.md
git commit -m "docs(bedrock): add AWS Bedrock setup guide"
```

---

## End-to-End Validation Checklist

Before declaring done, verify the following manually (or with an integration test):

- [ ] Connection with `provider=bedrock`, `auth_type=aws_iam`, a valid region, and
  instance/env-var credentials → "Verify" button returns success + model count
- [ ] Model list in Open WebUI shows Bedrock models (Claude, Nova, etc.)
- [ ] Chat with a Bedrock model returns streaming responses
- [ ] Chat with tools (function calling) works — tool call round-trips correctly
- [ ] Setting explicit `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars on the
  container is picked up without any additional config
- [ ] On an EC2 instance with an IAM role and no env vars set, the connection still
  works (IMDS auto-detection)
- [ ] Removing/disabling the connection removes Bedrock models from the list
- [ ] `npm run check` and `npm run lint` pass
- [ ] `pytest backend/tests/` passes

---

## Open Questions / Risks

1. **Non-streaming fallback** — boto3 `converse_stream` requires
   `bedrock:InvokeModelWithResponseStream`. If the IAM role only has `bedrock:InvokeModel`,
   the streaming call will fail. The hermes-agent code handles this by catching the
   `AccessDeniedException` and falling back to `converse()`. Consider doing the same
   in `call_converse_stream()`.

2. **Tool calling support depth** — the current plan handles the basic tool-call round-
   trip. Bedrock-specific features (Guardrails, cachePoint markers, reasoning/thinking
   blocks for Claude) are out of scope for v1 and can be added incrementally.

3. **Cross-region inference profiles** — model IDs like `us.anthropic.claude-sonnet-4-6`
   are regional inference profiles. Discovery returns them; the frontend should show
   them. No special handling needed — they go in `payload['model']` as-is.

4. **Config storage of credentials** — AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
   should NOT be stored in the `api_configs` dict (it persists to the database). They
   should only be consumed from the process environment. The UI should make this clear.
   Consider a warning in the modal if the user tries to paste a key into the API Key
   field with `provider=bedrock`.

5. **boto3 availability** — `boto3` is already listed in `pyproject.toml` as a
   direct dependency (`boto3==1.42.62`), so it is always available in the standard
   Docker image. No optional-dep handling needed, unlike in hermes-agent.
