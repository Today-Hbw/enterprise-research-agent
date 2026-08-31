# Enterprise Research Agent

A lightweight, traceable research agent based on the [PRD](https://app.notion.com/p/3c897d4ad24781f698eefce7533cc444). The current release includes controlled knowledge ingestion and retrieval while keeping the remaining external capabilities deterministic and offline.

> Demo boundary: Knowledge Search is real by default. Web Search and HTTP Fetch can be explicitly enabled with server-side configuration; Python, Browser, and MCP remain fixed placeholder tools. SQL can be explicitly enabled with a read-only PostgreSQL configuration. The local deterministic embedding is intended for development, not production semantic quality.

## What works

- FastAPI chat API and POST-based SSE streaming
- Lightweight agent loop with max steps, total timeout, tool timeout, and repeated-call protection
- Parallel independent tool calls with a concurrency limit
- Unified tool registry, JSON schemas, permissions, results, and error boundaries
- Replaceable LLM provider interface: deterministic offline planner or OpenAI Responses API
- Real text ingestion, deterministic chunking, authorization filtering, and anchored citations
- Optional Qdrant vector storage; SQL is available as an explicitly configured read-only PostgreSQL tool, while three execution tools remain placeholders
- In-memory conversations and run records
- Source citations, explainable execution trace, token/cost metrics, configurable run budgets, and latency/call counts
- Tenant/principal-scoped Run Dashboard API and responsive three-panel UI for conversations, chat, sources, trace, and run metrics
- Explicit medium-screen Inspector drawer controls and built-in safe Markdown rendering for assistant answers
- Offline Agent/Retrieval Evaluation with versionable JSON datasets, threshold gates, and machine-readable reports
- Docker Compose and automated runtime/API/contract tests

## Architecture

```mermaid
flowchart LR
    UI[Web UI or API Client] --> API[FastAPI]
    API --> RT[Agent Runtime]
    RT --> LLM[LLM Provider Interface]
    RT --> REG[Tool Registry]
    REG --> KNOW[Knowledge Search]
    KNOW --> QDRANT[Memory or Qdrant]
    REG --> SQL[Optional Read-only PostgreSQL]
    REG --> STUB[6 Deterministic Stub Tools]
    RT --> STORE[In-memory Store]
    STORE --> OUT[SSE + Answer + Sources + Trace]
```

The deterministic provider still exercises the real runtime behavior. Discovery tools run in parallel, schema discovery gates read-only SQL, SQL results gate Python analysis, and browser is only selected for explicit interactive intent.

## LLM provider modes

The default remains `LLM_PROVIDER=deterministic`, so the application runs without a key and every
external tool remains a fixed-response demo stub.

To enable the OpenAI provider, set these environment variables locally (never commit the key):

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
# Optional: OPENAI_BASE_URL=https://api.openai.com/v1
```

The provider uses the [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
with the registry's JSON schemas as function tools. It preserves the model-issued `call_id`, returns
tool results as `function_call_output` items with `previous_response_id`, permits parallel tool calls,
and records input/output token totals in each run. This follows OpenAI's
[function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).

The application fails fast if `LLM_PROVIDER=openai` is selected without `OPENAI_API_KEY`; it does not
silently fall back to deterministic answers.

### Run budgets and cost estimates

Token and cost controls are server-owned and optional:

```bash
RUN_TOKEN_BUDGET=50000
RUN_COST_BUDGET_USD=1.00
LLM_INPUT_COST_PER_MILLION_TOKENS=0.25
LLM_OUTPUT_COST_PER_MILLION_TOKENS=2.00
```

Pricing is intentionally operator-supplied because it varies by provider, model, contract, and time.
The input/output rates must be configured together; every Run then records an estimated USD cost from
reported input/output usage. The runtime checks configured ceilings before tool execution and before
each subsequent model
call, marks the persisted Run as budget-exhausted, and stops further work. A provider response can
cross the remaining ceiling because the exact input usage is only known after that response returns;
if it already contains the final answer, the answer is retained and the over-budget state is visible.

## Run locally

Python 3.12 or 3.13 is recommended. Python 3.14 may not yet be supported by every pinned dependency.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The default `KNOWLEDGE_BACKEND=memory` needs no external service. To use Qdrant, configure:

```bash
KNOWLEDGE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
KNOWLEDGE_ADMIN_TOKEN=a_long_random_admin_token
```

Open <http://localhost:8000>. API docs are available at <http://localhost:8000/docs>.

### Frontend behavior

The desktop layout keeps the three-column Conversation / Chat / Inspector workspace. At viewport
widths from 701px through 980px, Inspector becomes an explicit drawer: use the header button to
open it and the close button, backdrop, or Escape key to dismiss it. The compact mobile layout keeps
Inspector hidden so Chat retains the available width.

Assistant answers render a built-in Markdown subset covering headings, paragraphs, ordered and
unordered lists, emphasis, inline and fenced code, blockquotes, dividers, and links. The renderer
does not execute raw HTML or load a third-party client script; only HTTP(S) Markdown links become
clickable. User messages and error text remain plain text.

Docker Compose starts the agent and Qdrant with named-volume persistence. Its fallback secrets are for localhost development only; set both variables before exposing or sharing the stack:

```bash
KNOWLEDGE_ADMIN_TOKEN=replace-me QDRANT_API_KEY=replace-me docker compose up --build
```

### Controlled knowledge ingestion

Ingestion is disabled unless `KNOWLEDGE_ADMIN_TOKEN` is configured:


```bash
curl -X POST http://localhost:8000/api/knowledge/documents \
  -H "Content-Type: application/json" \
  -H "X-Knowledge-Admin-Token: $KNOWLEDGE_ADMIN_TOKEN" \
  -H "X-Tenant-Id: demo" \
  -H "X-Principal-Ids: demo-user" \
  -d '{"title":"Supplier policy","content":"Quarterly review is required.","knowledge_base_id":"policy","allowed_principal_ids":["demo-user"]}'
```

`X-Tenant-Id` and `X-Principal-Ids` are ignored by default. Set `KNOWLEDGE_TRUST_ACCESS_HEADERS=true` only behind an authenticated gateway that removes caller-supplied versions and injects verified identity. This project does not yet implement authentication or a tenant directory.

Knowledge citations include `document_id`, `chunk_id`, `knowledge_base_id`, `char_start`, `char_end`, score, and the exact snippet. Character spans index the original submitted text.

### URL download, parse, and RAG import

URL import is available only when both controlled knowledge ingestion and the safe HTTP backend are
enabled. It reuses the HTTP host allowlist, public-IP DNS validation, redirect checks, response-size
limit, and HTTPS-only default before parsing and indexing the document:

```bash
curl -X POST http://localhost:8000/api/knowledge/import-url \
  -H "Content-Type: application/json" \
  -H "X-Knowledge-Admin-Token: $KNOWLEDGE_ADMIN_TOKEN" \
  -H "X-Tenant-Id: demo" \
  -H "X-Principal-Ids: demo-user" \
  -d '{"url":"https://docs.example.com/policy.pdf","knowledge_base_id":"policy"}'
```

Supported response types are UTF-8 plain text and Markdown, HTML, CSV, JSON/JSON-LD, and PDF.
`HTTP_FETCH_MAX_BYTES` bounds the downloaded body and `KNOWLEDGE_IMPORT_MAX_PDF_PAGES` bounds PDF
work. Parsed text is also capped at the knowledge document limit. Encrypted PDFs, image-only/scanned
PDFs without extractable text, ambiguous content types, and non-UTF-8 text are rejected; OCR and
office document formats are future extensions.

The import response includes the final redirected URL, detected content type, sanitized filename,
content SHA-256, document ID, and chunk count. The document ID is derived from tenant, source,
content, metadata, and access settings, so repeating an identical import safely upserts the same
document/chunk IDs. Changed content or permissions produce a different document ID.

Set `KNOWLEDGE_RANKING=hybrid` with the in-memory backend to fuse semantic and title/content keyword rankings using Reciprocal Rank Fusion. `KNOWLEDGE_HYBRID_RRF_K` tunes the fusion constant (default `60`). Set `KNOWLEDGE_RERANKER=token_overlap` to re-rank the configured candidate pool deterministically by query overlap in title and content; it is disabled by default. Metadata can be attached at ingestion and filtered only through server-configured KNOWLEDGE_METADATA_FILTER_KEYS; empty configuration disables metadata filters. Qdrant remains explicitly semantic-only until its sparse-vector/text-index path is configured.

To start the local stack with development defaults:

```bash
docker compose up --build
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health and demo-mode status |
| `GET` | `/api/tools` | Tool catalog and JSON schemas |
| `POST` | `/api/knowledge/documents` | Admin-token protected text ingestion |
| `POST` | `/api/knowledge/import-url` | Admin-token protected safe URL download, parse, and RAG ingestion |
| `POST` | `/api/chat` | Run synchronously and return a complete run |
| `POST` | `/api/chat/stream` | Stream ordered SSE events |
| `GET` | `/api/conversations` | List in-memory conversations |
| `GET` | `/api/conversations/{id}` | Read one conversation |
| `GET` | `/api/runs?limit=20` | Aggregate scoped run/token/cost metrics and recent Run summaries |
| `GET` | `/api/runs/{id}` | Read a traceable run |

Example:
### Isolated Python calculation

`PYTHON_BACKEND=stub` remains the default. To enable deterministic calculations, set:

```bash
PYTHON_BACKEND=isolated
PYTHON_WORKER_TIMEOUT_SECONDS=5
PYTHON_WORKER_MAX_OUTPUT_BYTES=65536
```

The enabled tool launches a separate `python -I -S` process in a fresh temporary working directory. It accepts only a bounded arithmetic/comparison expression and explicitly supplied scalar or numeric-sequence variables; imports, attribute access, calls, assignments, files, and networking are rejected by an AST allowlist. The parent enforces input/output caps and terminates the worker at the configured timeout. The expression-only contract intentionally bounds CPU and memory work at the application layer; OS-level cgroup/Job Object resource caps are a deployment hardening follow-up.
### Read-only PostgreSQL schema retrieval and SQL

SQL stays in `stub` mode unless configured. Enabling it replaces both `schema_search` and `execute_sql` with a PostgreSQL-backed implementation:

```bash
SQL_BACKEND=postgres
POSTGRES_DSN=postgresql://research_readonly:replace-me@localhost:5432/research
POSTGRES_ALLOWED_SCHEMAS=public,analytics
POSTGRES_QUERY_TIMEOUT_MS=5000
POSTGRES_MAX_ROWS=500
```

The configured database role must independently be read-only and limited to the approved schemas. Before execution, SQLGlot parses exactly one AST and rejects mutations, DDL, `SELECT INTO`, locking clauses, unapproved schemas, and unqualified tables when `public` is not permitted. Each query runs in a PostgreSQL read-only transaction with a local statement timeout, receives an enforced row cap, and creates an in-process audit event containing only a statement hash, tenant ID, row count, and truncation flag. This release intentionally does not persist audit events; durable audit storage remains part of the persistence milestone.
### Public Web Search and safe HTTP Fetch

Both capabilities remain `stub` by default, so local development and offline tests do not make network requests. To enable Brave Search, obtain a server-side subscription token and set:

```bash
WEB_SEARCH_BACKEND=brave
BRAVE_SEARCH_API_KEY=your_brave_subscription_token
```

The provider calls Brave's Web Search endpoint with the token in `X-Subscription-Token`; the tool schema, trace, API response, and citations never contain that secret. Brave documents this endpoint and header in its [Web Search API reference](https://api-dashboard.search.brave.com/api-reference/web/search/get) and [authentication guide](https://api-dashboard.search.brave.com/documentation/guides/authentication).

HTTP Fetch is separately enabled and requires an explicit allowlist. Use only domains controlled or approved by your organization:

```bash
HTTP_FETCH_BACKEND=safe
HTTP_ALLOWED_HOSTS=www.example.com,.approved.example
HTTP_FETCH_MAX_BYTES=1000000
HTTP_FETCH_MAX_REDIRECTS=3
```

The fetcher permits HTTPS by default, rejects URL credentials and IP literals, requires every resolved address to be public, validates every redirect target again, accepts only HTML/plain text/JSON, and stops reading after the configured byte limit. `HTTP_FETCH_ALLOW_HTTP=true` is only for narrowly controlled development endpoints and should not be used in production. This is a defense-in-depth application control; production deployment should additionally enforce outbound egress rules and DNS protections at the network layer.

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"调研市场并结合采购数据做分析"}'
```

Events are sequenced and include `run_started`, `agent_decision`, `tool_started`, `tool_completed`, `assistant_delta`, and `run_completed` (or `run_failed`). The trace intentionally records decision summaries, not hidden chain-of-thought.

## Agent and Retrieval Evaluation

The repository includes a deterministic offline evaluation runner and a strict example dataset:

```bash
python -m app.evaluation \
  --dataset evals/demo.json \
  --output output/evaluation/demo-report.json
```

The command emits the same JSON report to stdout and returns exit code `1` when any configured
threshold is missed. Use `--no-fail-on-threshold` for exploratory runs that should always return
success.

Retrieval cases index the dataset's documents into an isolated in-memory backend, run the configured
semantic/hybrid/rerank path, and report macro Recall@K, Mean Reciprocal Rank, Hit Rate, and
case-level ranked document IDs. Agent cases execute the real `AgentRuntime` with the deterministic
provider, real Knowledge Search, and the existing fixed tools; they check:

- completed Run status and expected/forbidden tool routing;
- source count and required source types;
- presence of numbered citations and optional required answer terms;
- LLM/tool call and latency ceilings;
- aggregate pass, completion, tool-recall, and citation rates.

The dataset also defines regression thresholds, so the JSON file can be reviewed and versioned with
code changes. This first slice intentionally does not use an LLM-as-a-judge, live network services,
or a production embedding model. Citation evaluation checks traceability presence, not factual
faithfulness; human or model-based answer-quality scoring can be added later without changing the
current dataset/report boundary.

## Test

```bash
pytest
ruff check .
python -m app.evaluation --dataset evals/demo.json
```

## Safety and current limitations

- Knowledge filters are generated server-side from an `AccessContext`, never from LLM tool arguments.
- Access headers are disabled by default and are only a trusted-gateway integration seam, not authentication.
- Ingestion requires a separate admin token and is disabled when the token is absent.
- The deterministic local embedding supports offline tests but is not a substitute for a production embedding model.
- Qdrant should use API-key/TLS controls and private networking outside local development.
- SQL remains stubbed by default; the PostgreSQL backend requires explicit configuration, AST validation, a read-only transaction, timeout, schema allowlist, row cap, and non-sensitive audit metadata.
- Python is stubbed by default. The optional isolated mode supports only bounded expressions, not arbitrary Python packages or scripts; OS-level resource caps remain a deployment hardening follow-up.
- HTTP and browser tools are stubbed by default. The opt-in safe HTTP backend makes only allowlisted,
  public-address, size-bounded requests; browser automation remains a stub.
- Tool execution has a server-side `TOOL_MAX_PERMISSION` ceiling (`high` by default); calls above it are rejected and retained in the run trace. High-risk browser behavior still has no approval workflow.
- Run token/cost budgets are enforced between provider responses; they are not billing guarantees, and configured cost rates must be kept current by the operator.
- The built-in evaluation runner is an offline structural regression suite; it does not claim production retrieval quality or semantic answer correctness.
- The OpenAI API key is read only from configuration and is never included in API responses, traces, or logs.
- State is process-local and disappears on restart.
- Authentication, tenant provisioning, durable audit logs, and production rate limiting are not implemented.

## Replacement path

Each real integration should implement the existing `BaseTool` contract and be registered without changing the agent runtime or API event protocol. Recommended order:

1. Real LLM provider and native tool calling.
2. Production embedding, payload indexes, corpus-scale reranking, and retrieval evaluation.
3. Extend the bounded URL importer with OCR and office document parsers.
4. Schema retrieval, SQL AST validation, read-only PostgreSQL, timeout, and row limit.
5. Isolated Python worker and Playwright browser worker.
6. MCP discovery/invocation and permissions.
7. PostgreSQL/Redis persistence, auth, budget controls, and evaluation.
