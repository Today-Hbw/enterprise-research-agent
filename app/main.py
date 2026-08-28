import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.provider import build_provider
from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.knowledge import (
    DeterministicEmbedder,
    IndexedKnowledgeDocument,
    InMemoryKnowledgeBackend,
    KnowledgeDocumentInput,
    KnowledgeService,
    QdrantKnowledgeBackend,
)
from app.models import (
    AccessContext,
    ChatRequest,
    ChatResponse,
    Conversation,
    RunRecord,
    StreamEvent,
    ToolSpec,
)
from app.store import store
from app.tools.browser import PlaywrightBrowserTool
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.mcp import McpInvokeTool, load_mcp_catalog
from app.tools.python_worker import IsolatedPythonTool
from app.tools.sql import ExecuteSqlTool, PostgresBackend, SchemaSearchTool
from app.tools.stubs import build_tool_registry
from app.tools.web import BraveSearchBackend, HttpFetchTool, SafeHttpFetcher, WebSearchTool

settings = get_settings()
embedder = DeterministicEmbedder(settings.knowledge_embedding_dimensions)
if settings.knowledge_backend == "qdrant":
    knowledge_backend = QdrantKnowledgeBackend(
        base_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimensions=settings.knowledge_embedding_dimensions,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )
else:
    knowledge_backend = InMemoryKnowledgeBackend()
knowledge_service = KnowledgeService(
    backend=knowledge_backend,
    embedder=embedder,
    chunk_size=settings.knowledge_chunk_size,
    chunk_overlap=settings.knowledge_chunk_overlap,
)
web_resources: list[BraveSearchBackend | SafeHttpFetcher] = []
web_search_tool: WebSearchTool | None = None
http_fetch_tool: HttpFetchTool | None = None
sql_backend: PostgresBackend | None = None
schema_search_tool: SchemaSearchTool | None = None
execute_sql_tool: ExecuteSqlTool | None = None
python_execute_tool: IsolatedPythonTool | None = None
browser_tool: PlaywrightBrowserTool | None = None
mcp_invoke_tool = McpInvokeTool(
    load_mcp_catalog(settings.mcp_servers_json), settings.tool_timeout_seconds
)
if settings.browser_backend == "playwright":
    browser_tool = PlaywrightBrowserTool(
        set(settings.browser_allowed_hosts.split(",")), settings.browser_timeout_seconds
    )
if settings.python_backend == "isolated":
    python_execute_tool = IsolatedPythonTool(
        timeout_seconds=settings.python_worker_timeout_seconds,
        max_output_bytes=settings.python_worker_max_output_bytes,
    )
if settings.sql_backend == "postgres":
    if settings.postgres_dsn is None:
        raise RuntimeError("POSTGRES_DSN is required when SQL_BACKEND=postgres")
    sql_backend = PostgresBackend(
        dsn=settings.postgres_dsn.get_secret_value(),
        allowed_schemas=frozenset(
            item.strip() for item in settings.postgres_allowed_schemas.split(",") if item.strip()
        ),
        query_timeout_ms=settings.postgres_query_timeout_ms,
        max_rows=settings.postgres_max_rows,
    )
    schema_search_tool = SchemaSearchTool(sql_backend, settings.tool_timeout_seconds)
    execute_sql_tool = ExecuteSqlTool(sql_backend, settings.tool_timeout_seconds)
if settings.web_search_backend == "brave":
    brave_api_key = (
        settings.brave_search_api_key.get_secret_value() if settings.brave_search_api_key else ""
    )
    brave_backend = BraveSearchBackend(
        api_key=brave_api_key,
        base_url=settings.brave_search_base_url,
        timeout_seconds=settings.web_search_timeout_seconds,
    )
    web_resources.append(brave_backend)
    web_search_tool = WebSearchTool(
        backend=brave_backend, timeout_seconds=settings.tool_timeout_seconds
    )
if settings.http_fetch_backend == "safe":
    safe_fetcher = SafeHttpFetcher(
        allowed_hosts=set(settings.http_allowed_hosts.split(",")),
        max_bytes=settings.http_fetch_max_bytes,
        max_redirects=settings.http_fetch_max_redirects,
        timeout_seconds=settings.http_fetch_timeout_seconds,
        allow_http=settings.http_fetch_allow_http,
    )
    web_resources.append(safe_fetcher)
    http_fetch_tool = HttpFetchTool(
        fetcher=safe_fetcher, timeout_seconds=settings.tool_timeout_seconds
    )
registry = build_tool_registry(
    settings.tool_timeout_seconds,
    knowledge_tool=KnowledgeSearchTool(
        service=knowledge_service, timeout_seconds=settings.tool_timeout_seconds
    ),
    web_search_tool=web_search_tool,
    http_fetch_tool=http_fetch_tool,
    schema_search_tool=schema_search_tool,
    execute_sql_tool=execute_sql_tool,
    python_execute_tool=python_execute_tool,
    browser_tool=browser_tool,
    mcp_invoke_tool=mcp_invoke_tool,
)
runtime = AgentRuntime(
    settings=settings,
    provider=build_provider(settings),
    registry=registry,
    store=store,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await runtime.provider.aclose()
        await knowledge_service.aclose()

        for resource in web_resources:
            await resource.aclose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Traceable research agent with controlled knowledge retrieval and stub tools.",
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "provider": runtime.provider.name,
        "demo_mode": runtime.provider.is_demo,
        "knowledge_backend": settings.knowledge_backend,
        "web_search_backend": settings.web_search_backend,
        "http_fetch_backend": settings.http_fetch_backend,
        "sql_backend": settings.sql_backend,
        "python_backend": settings.python_backend,
        "browser_backend": settings.browser_backend,
    }


@app.get("/api/tools", response_model=list[ToolSpec])
async def list_tools() -> list[ToolSpec]:
    return registry.specs()


def access_context_from_headers(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> AccessContext:
    if not settings.knowledge_trust_access_headers:
        return AccessContext(
            tenant_id=settings.knowledge_default_tenant,
            principal_ids={settings.knowledge_default_principal},
        )
    tenant_id = (x_tenant_id or settings.knowledge_default_tenant).strip()
    if not tenant_id or len(tenant_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid X-Tenant-Id")
    principal_ids = {item.strip() for item in (x_principal_ids or "").split(",") if item.strip()}
    if any(len(item) > 128 for item in principal_ids):
        raise HTTPException(status_code=400, detail="Invalid X-Principal-Ids")
    if not principal_ids:
        principal_ids.add(settings.knowledge_default_principal)
    return AccessContext(tenant_id=tenant_id, principal_ids=principal_ids)


def require_knowledge_admin(x_knowledge_admin_token: str | None) -> None:
    configured = settings.knowledge_admin_token
    secret = configured.get_secret_value() if configured else ""
    if not secret:
        raise HTTPException(status_code=503, detail="Knowledge ingestion is disabled")
    supplied = x_knowledge_admin_token or ""
    if not hmac.compare_digest(secret, supplied):
        raise HTTPException(status_code=403, detail="Invalid knowledge admin token")


@app.post(
    "/api/knowledge/documents",
    response_model=IndexedKnowledgeDocument,
    status_code=201,
)
async def create_knowledge_document(
    document: KnowledgeDocumentInput,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
    x_knowledge_admin_token: str | None = Header(default=None, alias="X-Knowledge-Admin-Token"),
) -> IndexedKnowledgeDocument:
    require_knowledge_admin(x_knowledge_admin_token)
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    if not document.public and not document.allowed_principal_ids:
        document = document.model_copy(
            update={"allowed_principal_ids": sorted(access_context.principal_ids)}
        )
    return await knowledge_service.index_document(
        tenant_id=access_context.tenant_id, document=document
    )


@app.get("/api/conversations", response_model=list[Conversation])
async def list_conversations(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> list[Conversation]:
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    return await store.list_conversations(
        tenant_id=access_context.tenant_id,
        principal_ids=access_context.principal_ids,
    )


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> Conversation:
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    conversation = await store.get_conversation(
        conversation_id,
        tenant_id=access_context.tenant_id,
        principal_ids=access_context.principal_ids,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/api/runs/{run_id}", response_model=RunRecord)
async def get_run(
    run_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> RunRecord:
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    run = await store.get_run(
        run_id,
        tenant_id=access_context.tenant_id,
        principal_ids=access_context.principal_ids,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> ChatResponse:
    final_event: StreamEvent | None = None
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    try:
        async for item in runtime.stream(
            query=request.query.strip(),
            conversation_id=request.conversation_id,
            access_context=access_context,
        ):
            final_event = item
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if final_event is None:
        raise HTTPException(status_code=500, detail="Agent produced no events")
    run = await store.get_run(
        final_event.run_id,
        tenant_id=access_context.tenant_id,
        principal_ids=access_context.principal_ids,
    )
    if run is None:
        raise HTTPException(status_code=500, detail="Run was not persisted")
    return ChatResponse(conversation_id=run.conversation_id, run=run)


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_principal_ids: str | None = Header(default=None, alias="X-Principal-Ids"),
) -> StreamingResponse:
    access_context = access_context_from_headers(x_tenant_id, x_principal_ids)
    if request.conversation_id:
        existing = await store.get_conversation(
            request.conversation_id,
            tenant_id=access_context.tenant_id,
            principal_ids=access_context.principal_ids,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    async def generate() -> AsyncIterator[str]:
        async for item in runtime.stream(
            query=request.query.strip(),
            conversation_id=request.conversation_id,
            access_context=access_context,
        ):
            payload = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            yield f"event: {item.event}\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
