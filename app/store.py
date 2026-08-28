import asyncio

from app.models import Conversation, Message, RunRecord, RunStatus, utc_now


class InMemoryStore:
    """Demo-only state store. Replace with PostgreSQL/Redis for durable deployments."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_conversation(
        self,
        conversation_id: str | None,
        first_query: str,
        tenant_id: str = "demo",
        principal_ids: set[str] | None = None,
    ) -> Conversation:
        effective_principals = principal_ids or {"demo-user"}
        async with self._lock:
            if conversation_id:
                conversation = self._conversations.get(conversation_id)
                if (
                    conversation is None
                    or conversation.tenant_id != tenant_id
                    or not conversation.principal_ids.intersection(effective_principals)
                ):
                    raise KeyError(f"Conversation not found: {conversation_id}")
                return conversation

            title = first_query.strip().replace("\n", " ")[:60] or "New conversation"
            conversation = Conversation(
                title=title,
                tenant_id=tenant_id,
                principal_ids=effective_principals,
            )
            self._conversations[conversation.conversation_id] = conversation
            return conversation

    async def add_message(self, conversation_id: str, message: Message) -> None:
        async with self._lock:
            conversation = self._conversations[conversation_id]
            conversation.messages.append(message)
            conversation.updated_at = utc_now()

    async def save_run(self, run: RunRecord) -> None:
        async with self._lock:
            conversation = self._conversations[run.conversation_id]
            if conversation.tenant_id != run.tenant_id:
                raise ValueError("Run tenant does not match its conversation tenant")
            self._runs[run.run_id] = run
            if run.run_id not in conversation.run_ids:
                conversation.run_ids.append(run.run_id)
            conversation.updated_at = utc_now()

    async def get_run(
        self,
        run_id: str,
        tenant_id: str | None = None,
        principal_ids: set[str] | None = None,
    ) -> RunRecord | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or (tenant_id is not None and run.tenant_id != tenant_id):
                return None
            if principal_ids is not None and not run.principal_ids.intersection(principal_ids):
                return None
            return run

    async def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str | None = None,
        principal_ids: set[str] | None = None,
    ) -> Conversation | None:
        async with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None or (
                tenant_id is not None and conversation.tenant_id != tenant_id
            ):
                return None
            if principal_ids is not None and not conversation.principal_ids.intersection(
                principal_ids
            ):
                return None
            return conversation

    async def list_conversations(
        self, tenant_id: str | None = None, principal_ids: set[str] | None = None
    ) -> list[Conversation]:
        async with self._lock:
            return sorted(
                (
                    item
                    for item in self._conversations.values()
                    if (tenant_id is None or item.tenant_id == tenant_id)
                    and (
                        principal_ids is None
                        or bool(item.principal_ids.intersection(principal_ids))
                    )
                ),
                key=lambda item: item.updated_at,
                reverse=True,
            )

    async def fail_running_run(self, run_id: str, error: str) -> RunRecord | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run and run.status == RunStatus.RUNNING:
                run.status = RunStatus.FAILED
                run.error = error
                run.completed_at = utc_now()
            return run


store = InMemoryStore()
