import asyncio

from app.models import Conversation, Message, RunRecord, RunStatus, utc_now


class InMemoryStore:
    """Demo-only state store. Replace with PostgreSQL/Redis for durable deployments."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_conversation(
        self, conversation_id: str | None, first_query: str
    ) -> Conversation:
        async with self._lock:
            if conversation_id:
                conversation = self._conversations.get(conversation_id)
                if conversation is None:
                    raise KeyError(f"Conversation not found: {conversation_id}")
                return conversation

            title = first_query.strip().replace("\n", " ")[:60] or "New conversation"
            conversation = Conversation(title=title)
            self._conversations[conversation.conversation_id] = conversation
            return conversation

    async def add_message(self, conversation_id: str, message: Message) -> None:
        async with self._lock:
            conversation = self._conversations[conversation_id]
            conversation.messages.append(message)
            conversation.updated_at = utc_now()

    async def save_run(self, run: RunRecord) -> None:
        async with self._lock:
            self._runs[run.run_id] = run
            conversation = self._conversations[run.conversation_id]
            if run.run_id not in conversation.run_ids:
                conversation.run_ids.append(run.run_id)
            conversation.updated_at = utc_now()

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        async with self._lock:
            return self._conversations.get(conversation_id)

    async def list_conversations(self) -> list[Conversation]:
        async with self._lock:
            return sorted(
                self._conversations.values(), key=lambda item: item.updated_at, reverse=True
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
