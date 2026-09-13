"""Tests for AWSSigV4Auth — botocore fully stubbed, no real credentials needed."""
import asyncio
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import httpx
import pytest


def _load_module_with_stubbed_botocore():
    """Load aws_sigv4.py with a minimal botocore stub injected into sys.modules."""
    # Build minimal botocore stubs
    botocore_stub = types.ModuleType("botocore")
    botocore_auth = types.ModuleType("botocore.auth")
    botocore_awsrequest = types.ModuleType("botocore.awsrequest")
    botocore_session = types.ModuleType("botocore.session")

    # Fake frozen credentials
    frozen = MagicMock()
    frozen.access_key = "AKIATEST"
    frozen.secret_key = "testsecret"
    frozen.token = None

    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen

    session_instance = MagicMock()
    session_instance.get_credentials.return_value = creds
    botocore_session.Session = MagicMock(return_value=session_instance)

    # Fake SigV4Auth that stamps a deterministic Authorization header
    class FakeSigV4Auth:
        def __init__(self, credentials, service, region):
            self.service = service
            self.region = region

        def add_auth(self, aws_req):
            aws_req.headers["Authorization"] = (
                f"AWS4-HMAC-SHA256 Service={self.service},Region={self.region}"
            )

    # Fake AWSRequest
    class FakeAWSRequest:
        def __init__(self, method, url, data, headers):
            self.method = method
            self.url = url
            self.body = data
            self.headers = dict(headers)

        def prepare(self):
            m = MagicMock()
            m.headers = self.headers
            return m

    botocore_auth.SigV4Auth = FakeSigV4Auth
    botocore_awsrequest.AWSRequest = FakeAWSRequest

    # Inject stubs — must set attributes on the top-level module so that
    # `import botocore; botocore.session` resolves correctly inside _lazy_botocore.
    botocore_stub.auth = botocore_auth
    botocore_stub.awsrequest = botocore_awsrequest
    botocore_stub.session = botocore_session

    sys.modules["botocore"] = botocore_stub
    sys.modules["botocore.auth"] = botocore_auth
    sys.modules["botocore.awsrequest"] = botocore_awsrequest
    sys.modules["botocore.session"] = botocore_session

    path = os.path.join(
        os.path.dirname(__file__), "../open_webui/utils/aws_sigv4.py"
    )
    spec = importlib.util.spec_from_file_location("aws_sigv4_test_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module_with_stubbed_botocore()
AWSSigV4Auth = _mod.AWSSigV4Auth


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_is_httpx_auth_subclass():
    """AWSSigV4Auth must be a proper httpx.Auth subclass."""
    assert issubclass(AWSSigV4Auth, httpx.Auth)


def test_sync_auth_flow_adds_authorization_header():
    auth = AWSSigV4Auth(region="us-east-1", service="bedrock")
    req = httpx.Request("GET", "https://example.amazonaws.com/mcp")
    signed = next(auth.auth_flow(req))
    assert "Authorization" in signed.headers
    assert "AWS4-HMAC-SHA256" in signed.headers["Authorization"]


def test_sync_auth_flow_embeds_region_and_service():
    auth = AWSSigV4Auth(region="eu-west-1", service="execute-api")
    req = httpx.Request("POST", "https://api.example.com/v1", content=b'{"x":1}')
    signed = next(auth.auth_flow(req))
    assert "eu-west-1" in signed.headers["Authorization"]
    assert "execute-api" in signed.headers["Authorization"]


def test_default_service_is_bedrock():
    auth = AWSSigV4Auth(region="us-east-1")
    assert auth.service == "bedrock"


def test_async_auth_flow_adds_authorization_header():
    auth = AWSSigV4Auth(region="us-east-1", service="bedrock")
    req = httpx.Request("GET", "https://example.amazonaws.com/mcp")

    async def run():
        gen = auth.async_auth_flow(req)
        signed = await gen.__anext__()
        return signed

    signed = asyncio.get_event_loop().run_until_complete(run())
    assert "Authorization" in signed.headers
    assert "AWS4-HMAC-SHA256" in signed.headers["Authorization"]
