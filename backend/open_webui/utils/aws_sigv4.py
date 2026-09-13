"""
AWS SigV4 httpx.Auth implementation for Open WebUI MCP connections.

Signs each httpx request using botocore's SigV4Auth signer.
Credentials are resolved at call time via the boto3 default chain:
  env vars -> ~/.aws/credentials profile -> ECS/EKS/EC2 instance role.

The credential fetch runs in a thread executor so IMDS-backed credentials
(RefreshableCredentials) never block the asyncio event loop.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator

import httpx

log = logging.getLogger(__name__)


def _lazy_botocore():
    try:
        import botocore.auth
        import botocore.awsrequest
        import botocore.session

        return botocore
    except ImportError as exc:
        raise ImportError(
            'botocore is required for AWS IAM authentication. '
            'It ships with boto3; ensure boto3 is installed.'
        ) from exc


class AWSSigV4Auth(httpx.Auth):
    """httpx.Auth subclass that signs requests with AWS SigV4.

    Usage::

        auth = AWSSigV4Auth(region="us-east-1", service="bedrock")
        client = httpx.AsyncClient(auth=auth)

    Credential resolution order (boto3 default chain):
      1. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
      2. AWS_PROFILE or default profile in ~/.aws/credentials
      3. ECS task role (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI)
      4. EKS IRSA (AWS_WEB_IDENTITY_TOKEN_FILE)
      5. EC2 / EKS node IMDS instance role
    """

    def __init__(self, region: str, service: str = 'bedrock') -> None:
        self.region = region
        self.service = service

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_frozen_credentials(self):
        """Synchronous credential fetch — safe to call in a thread executor.

        For static (env-var / file) credentials this is effectively
        instantaneous.  For RefreshableCredentials (EC2/ECS/EKS instance
        roles) this may make a network call to IMDS/STS when the token is
        near expiry, which is why callers should run it in an executor.
        """
        botocore = _lazy_botocore()
        session = botocore.session.Session()
        return session.get_credentials().get_frozen_credentials()

    def _sign_request(self, request: httpx.Request, frozen_creds) -> httpx.Request:
        """Return a new httpx.Request with SigV4 Authorization headers applied."""
        botocore = _lazy_botocore()

        aws_request = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )

        signer = botocore.auth.SigV4Auth(frozen_creds, self.service, self.region)
        signer.add_auth(aws_request)

        prepared = aws_request.prepare()
        merged_headers = dict(request.headers)
        merged_headers.update(dict(prepared.headers))

        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=merged_headers,
            content=request.content,
        )

    # ------------------------------------------------------------------
    # httpx.Auth protocol
    # ------------------------------------------------------------------

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Async auth flow used by httpx.AsyncClient.

        Fetches credentials in a thread executor to avoid blocking the
        event loop when IMDS/STS token refresh is needed.
        """
        loop = asyncio.get_event_loop()
        frozen_creds = await loop.run_in_executor(None, self._get_frozen_credentials)
        yield self._sign_request(request, frozen_creds)

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Sync auth flow fallback used by httpx.Client (not AsyncClient).

        Fetches credentials directly — blocking is acceptable in sync
        contexts.
        """
        frozen_creds = self._get_frozen_credentials()
        yield self._sign_request(request, frozen_creds)
