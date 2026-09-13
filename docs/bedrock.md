# AWS Bedrock

Open WebUI supports AWS Bedrock as a first-class connection provider. It uses boto3's
Converse API under the hood, so it works with all Bedrock model families (Anthropic
Claude, Amazon Nova, Meta Llama, DeepSeek, Mistral, and more) and both authentication
approaches: IAM instance/task roles and explicit environment variable credentials.

## Prerequisites

An IAM identity (role or user) with these permissions:

- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:ListFoundationModels`
- `bedrock:ListInferenceProfiles`

The quickest IAM policy to attach is `AmazonBedrockFullAccess`.

## Authentication

### Option 1 — IAM Instance Role (EC2, ECS, EKS, Lambda)

Attach an IAM role with Bedrock permissions to your instance or task. No environment
variables or API keys are needed. Open WebUI detects the role automatically via the
instance metadata service (IMDS).

### Option 2 — Explicit Environment Variables

Pass credentials to the container (or set them in the host environment):

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...          # only needed for temporary credentials (STS)
AWS_DEFAULT_REGION=us-east-1   # or set in the connection's AWS Region field
```

Docker example:

```bash
docker run \
  -e AWS_ACCESS_KEY_ID=AKIA... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -p 3000:8080 \
  ghcr.io/open-webui/open-webui:latest
```

### Option 3 — AWS Profile

Mount your `~/.aws` credentials directory and set `AWS_PROFILE`:

```bash
docker run \
  -v ~/.aws:/root/.aws:ro \
  -e AWS_PROFILE=my-profile \
  ghcr.io/open-webui/open-webui:latest
```

## Connection Setup (UI)

1. Open Admin Panel -> Settings -> Connections.
2. Click **+** to add a new OpenAI-compatible connection.
3. Set the fields:
   - **URL**: leave blank (it will be set to `bedrock` automatically)
   - **Auth**: `AWS IAM`
   - **Provider**: `AWS Bedrock`
   - **AWS Region**: your target region (e.g. `us-east-1`)
4. Click **Verify** — you should see a success message with the model count.
5. Click **Save**.

Models from your Bedrock region (foundation models and cross-region inference profiles)
will appear in the model selector automatically.

## Credential Resolution Priority

When the connection receives a request, boto3 resolves credentials in this order:

1. `AWS_BEARER_TOKEN_BEDROCK` environment variable (Bedrock API-key auth)
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (static key pair)
3. `AWS_PROFILE` (named profile — SSO, assume-role, etc.)
4. `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` (ECS task role)
5. `AWS_WEB_IDENTITY_TOKEN_FILE` (EKS IRSA)
6. Instance metadata service (EC2 / EKS node role)

## Notes

- AWS credentials are **never stored** in Open WebUI's database. They are consumed
  from the process environment at request time.
- The model list is cached per-region for one hour. To refresh it, disable and
  re-enable the connection.
- Cross-region inference profiles (e.g. `us.anthropic.claude-sonnet-4-6`) are
  listed alongside foundation models and can be selected directly.
- If the IAM role has `bedrock:InvokeModel` but not
  `bedrock:InvokeModelWithResponseStream`, streaming is denied. Open WebUI falls
  back to non-streaming `converse()` automatically.
