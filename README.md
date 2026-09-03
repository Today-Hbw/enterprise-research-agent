# Enterprise Research Agent

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, traceable, and extensible enterprise research agent. It brings LLMs, enterprise knowledge, the public web, read-only databases, restricted computation, browser automation, and MCP tools into one runtime while preserving the plan, sources, execution trace, usage, and estimated cost of every research run.

The project runs in deterministic offline mode by default. No API key or external service is required to explore the complete agent workflow and user interface. Real capabilities can be enabled individually through OpenAI, Doubao, Qdrant, RAG Platform, Brave Search, PostgreSQL, Playwright, Redis, or MCP without rewriting the agent core.

## Highlights

- **Works out of the box:** the offline planner and stub tools support development, demos, and tests without external accounts.
- **Traceable by design:** synchronous and SSE APIs expose plans, sources, tool traces, latency, token usage, and cost estimates.
- **Controlled knowledge retrieval:** ingest text or URLs with deterministic chunking, access filtering, anchored citations, hybrid retrieval, and reranking.
- **Explicit tool boundaries:** every tool has a JSON Schema, permission level, timeout, concurrency control, repeat-call protection, and error boundary.
- **Opt-in integrations:** external capabilities stay disabled until enabled through server configuration, host allowlists, or read-only credentials.
- **Built-in evaluation and UI:** run offline Agent/Retrieval evaluations and inspect conversations, plans, sources, traces, and metrics in a responsive three-panel interface.

## Capability Matrix

| Capability | Default | Optional live backend |
|---|---|---|
| LLM | Deterministic offline planner | OpenAI or Doubao Responses API |
| Knowledge store | In-memory store | Qdrant or RAG Platform HTTP API |
| Retrieval | Semantic search | RRF hybrid search and token-overlap reranking |
| Web | Fixed-response stubs | Brave Search and safe HTTP Fetch |
| Data analysis | SQL and Python stubs | Read-only PostgreSQL and restricted expression evaluation |
| Extended tools | Browser and MCP stubs | Playwright and server-configured HTTPS MCP tools |
| State and events | Process memory | PostgreSQL and Redis |

## Architecture

```mermaid
flowchart LR
    Client[Web UI / API] --> API[FastAPI + SSE]
    API --> Runtime[Agent Runtime]
    Runtime --> LLM[Deterministic / OpenAI / Doubao]
    Runtime --> Registry[Tool Registry]
    Registry --> Tools[Knowledge / Web / SQL / Python / Browser / MCP]
    Tools --> Backends[Memory / Qdrant / RAG Platform / PostgreSQL]
    Runtime --> Result[Plan + Sources + Trace + Metrics]
```

The runtime enforces step limits, run and tool timeouts, concurrency limits, and repeated-call protection. Tool permission ceilings, token budgets, and cost budgets are controlled by the server. The high-level plan updates as work progresses without exposing hidden model reasoning.

## Quick Start

Python 3.12 or 3.13 is recommended.

```bash
git clone https://github.com/Today-Hbw/enterprise-research-agent.git
cd enterprise-research-agent
python -m venv .venv
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

- Web UI: <http://localhost:8000>
- OpenAPI documentation: <http://localhost:8000/docs>
- Health endpoint: <http://localhost:8000/api/health>

The application now runs entirely offline. Submit a research query to observe plan updates, parallel tool calls, source citations, and the execution trace.

### Docker Compose

```bash
docker compose up --build
```

Compose starts the agent and Qdrant on localhost. Its fallback credentials are intended only for local development. Set strong, random values for `KNOWLEDGE_ADMIN_TOKEN` and `QDRANT_API_KEY` before exposing the stack.

## Enable Live Capabilities

Copy the example configuration first. Never commit a `.env` file containing secrets.

```bash
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Common integration settings:

```dotenv
# OpenAI; use LLM_PROVIDER=doubao and the matching DOUBAO_* settings for Doubao
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini

# Qdrant
KNOWLEDGE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# RAG Platform requires hybrid retrieval and a knowledge_base_id
# KNOWLEDGE_BACKEND=rag-platform
# KNOWLEDGE_RANKING=hybrid
# RAG_PLATFORM_BASE_URL=http://localhost:8001
# RAG_PLATFORM_API_KEY=your_api_key

# Brave Search
WEB_SEARCH_BACKEND=brave
BRAVE_SEARCH_API_KEY=your_brave_subscription_token

# Read-only PostgreSQL
SQL_BACKEND=postgres
POSTGRES_DSN=postgresql://research_readonly:password@localhost:5432/research
POSTGRES_ALLOWED_SCHEMAS=public,analytics

# Optional restricted computation and browser automation
PYTHON_BACKEND=isolated
BROWSER_BACKEND=playwright
BROWSER_ALLOWED_HOSTS=app.example.com,.approved.example
```

Selecting a live LLM provider without its API key causes startup to fail instead of silently falling back to deterministic mode. The PostgreSQL role must independently have read-only access to the allowed schemas.

### Controlled Knowledge Ingestion

Knowledge writes are disabled until `KNOWLEDGE_ADMIN_TOKEN` is configured:

```bash
curl -X POST http://localhost:8000/api/knowledge/documents \
  -H "Content-Type: application/json" \
  -H "X-Knowledge-Admin-Token: replace-with-a-long-random-secret" \
  -d '{"title":"Supplier policy","content":"Quarterly review is required.","knowledge_base_id":"policy","allowed_principal_ids":["demo-user"]}'
```

URL ingestion also requires `HTTP_FETCH_BACKEND=safe` and an `HTTP_ALLOWED_HOSTS` allowlist. Supported content includes UTF-8 text, Markdown, HTML, CSV, JSON/JSON-LD, and PDFs with extractable text. Encrypted PDFs, scanned PDFs without a text layer, and Office documents are not currently supported.

See [`.env.example`](.env.example) and the [deployment guide](docs/部署配置.md) for more configuration options.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service health and active modes |
| `GET` | `/api/tools` | Tool catalog, schemas, and permissions |
| `POST` | `/api/knowledge/documents` | Ingest a text document |
| `POST` | `/api/knowledge/import-url` | Safely download, parse, and ingest a URL |
| `POST` | `/api/chat` | Run a synchronous research request |
| `POST` | `/api/chat/stream` | Stream a research request over SSE |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Read one conversation |
| `GET` | `/api/runs` | Read run summaries and aggregate metrics |
| `GET` | `/api/runs/{id}` | Read one run and its complete trace |
| `GET` | `/api/runs/{id}/events` | Replay run events |

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Research the market and analyze it against procurement data"}'
```

## Tests and Evaluation

```bash
pytest
ruff check .
python -m app.evaluation --dataset evals/demo.json --output output/evaluation/demo-report.json
```

The evaluation runner exercises the real Agent Runtime and knowledge retrieval path, writes a JSON report, and exits with status `1` when a configured threshold is missed. It covers tool routing, citations, Recall@K, MRR, Hit Rate, call counts, and latency. It is a structural regression suite, not a claim of production answer quality.

## Security Boundaries

- The default configuration does not access the network, databases, or a browser.
- Knowledge access filters come from the server-side `AccessContext`; identity headers should be trusted only behind an authenticated gateway that rewrites them.
- HTTP Fetch and Browser use host allowlists and reject URL credentials, IP literals, and non-public addresses.
- SQL accepts one read-only statement and applies schema allowlists, timeouts, and row limits.
- The isolated Python backend evaluates only AST-allowlisted expressions; it is not a general-purpose code sandbox.
- Authentication, tenant provisioning, production rate limiting, and durable audit logs are not built in and must be added before public deployment.

Read the [security design](docs/安全设计.md) for the complete threat model and deployment requirements.

## Documentation

- [Project overview](docs/项目总览.md)
- [Architecture](docs/架构详解.md)
- [Tool system](docs/工具系统.md)
- [API reference](docs/API接口.md)
- [Security design](docs/安全设计.md)
- [Deployment and configuration](docs/部署配置.md)

## Contributing

The project launched on **August 27, 2026** and has already reached **82 clones** from **50 unique cloners** in 14 days *(data as of September 3, 2026)*.

![Clone stats in last 14 days](docs/assets/clones-stats.png)

Issues, pull requests, and discussions are all welcome — keep the default offline mode operational, add tests for behavior changes, and run `pytest` and `ruff check .` before submitting. Never commit `.env` files, API keys, database credentials, customer data, or internal documents.

If you find this project useful, a ⭐ Star is greatly appreciated!

## License

This project is open source under the [MIT License](LICENSE). You may use, copy, modify, merge, publish, and distribute it as long as the original copyright and license notice are retained.
