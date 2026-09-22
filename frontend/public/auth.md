# CRAG MultiHop auth.md

## Overview

CRAG MultiHop provides authentication and registration protocols for AI agents to interact with corrective retrieval-augmented generation, multi-hop question answering, and document evaluation APIs.

- **Audience**: AI Agents, Code Assistants, and Automated Systems
- **Resource Identifier**: `https://crag.nevatal.tech`
- **Authorization Server**: `https://crag.nevatal.tech`
- **Registration Endpoint**: `https://crag.nevatal.tech/api/agent/register`
- **Token Endpoint**: `https://crag.nevatal.tech/api/auth/token`
- **Supported Methods**: Bearer Token (`Authorization: Bearer <token>`)
- **Supported Scopes**: `read`, `write`, `crag`, `mcp`

## Agent Registration Flow

Agents can register or provision an API token:

```http
POST /api/agent/register HTTP/1.1
Host: crag.nevatal.tech
Content-Type: application/json

{
  "client_name": "MyCragAgent",
  "client_uri": "https://example.com/agent"
}
```

Response:
```json
{
  "access_token": "crag_agent_token_sample",
  "token_type": "Bearer",
  "expires_in": 2592000,
  "scope": "read write crag mcp"
}
```

## Discovery Endpoints

- OAuth Authorization Server Metadata: [/.well-known/oauth-authorization-server](https://crag.nevatal.tech/.well-known/oauth-authorization-server)
- OpenID Connect Configuration: [/.well-known/openid-configuration](https://crag.nevatal.tech/.well-known/openid-configuration)
- OAuth Protected Resource Metadata: [/.well-known/oauth-protected-resource](https://crag.nevatal.tech/.well-known/oauth-protected-resource)
- API Catalog: [/.well-known/api-catalog](https://crag.nevatal.tech/.well-known/api-catalog)
- MCP Server Card: [/.well-known/mcp/server-card.json](https://crag.nevatal.tech/.well-known/mcp/server-card.json)
- A2A Agent Card: [/.well-known/agent-card.json](https://crag.nevatal.tech/.well-known/agent-card.json)
- Agent Skills Discovery: [/.well-known/agent-skills/index.json](https://crag.nevatal.tech/.well-known/agent-skills/index.json)
