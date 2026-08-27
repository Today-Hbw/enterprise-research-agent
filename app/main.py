import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.provider import build_provider
from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, Conversation, RunRecord, StreamEvent, ToolSpec
from app.store import store
from app.tools.stubs import build_stub_registry

settings = get_settings()
registry = build_stub_registry(settings.tool_timeout_seconds)
runtime = AgentRuntime(
    settings=settings,
    provider=build_provider(settings),
    registry=registry,
    store=store,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await runtime.provider.aclose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Traceable research-agent skeleton. External tools return fixed demo content.",
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
    }


@app.get("/api/tools", response_model=list[ToolSpec])
async def list_tools() -> list[ToolSpec]:
    return registry.specs()


@app.get("/api/conversations", response_model=list[Conversation])
async def list_conversations() -> list[Conversation]:
    return await store.list_conversations()


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str) -> Conversation:
    conversation = await store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/api/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: str) -> RunRecord:
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    final_event: StreamEvent | None = None
    try:
        async for item in runtime.stream(
            query=request.query.strip(), conversation_id=request.conversation_id
        ):
            final_event = item
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if final_event is None:
        raise HTTPException(status_code=500, detail="Agent produced no events")
    run = await store.get_run(final_event.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Run was not persisted")
    return ChatResponse(conversation_id=run.conversation_id, run=run)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if request.conversation_id:
        existing = await store.get_conversation(request.conversation_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    async def generate() -> AsyncIterator[str]:
        async for item in runtime.stream(
            query=request.query.strip(), conversation_id=request.conversation_id
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
