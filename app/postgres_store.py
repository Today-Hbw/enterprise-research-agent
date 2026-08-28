from __future__ import annotations

from typing import Any

import psycopg

from app.models import Conversation, Message, RunRecord, RunStatus, utc_now


class PostgresStore:
    """Durable JSONB state store; database access remains tenant/principal scoped."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def initialize(self) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS agent_conversations (id text PRIMARY KEY, tenant_id text NOT NULL, principals text[] NOT NULL, updated_at timestamptz NOT NULL, payload jsonb NOT NULL)"
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS agent_conversations_scope ON agent_conversations (tenant_id, updated_at DESC)"
            )
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS agent_runs (id text PRIMARY KEY, conversation_id text NOT NULL REFERENCES agent_conversations(id), tenant_id text NOT NULL, principals text[] NOT NULL, payload jsonb NOT NULL)"
            )
            await connection.commit()

    async def get_or_create_conversation(
        self,
        conversation_id: str | None,
        first_query: str,
        tenant_id: str = "demo",
        principal_ids: set[str] | None = None,
    ) -> Conversation:
        principals = principal_ids or {"demo-user"}
        if conversation_id:
            item = await self.get_conversation(conversation_id, tenant_id, principals)
            if item is None:
                raise KeyError(f"Conversation not found: {conversation_id}")
            return item
        item = Conversation(
            title=first_query.strip().replace("\n", " ")[:60] or "New conversation",
            tenant_id=tenant_id,
            principal_ids=principals,
        )
        await self._save_conversation(item)
        return item

    async def add_message(self, conversation_id: str, message: Message) -> None:
        item = await self.get_conversation(conversation_id)
        if item is None:
            raise KeyError(conversation_id)
        item.messages.append(message)
        item.updated_at = utc_now()
        await self._save_conversation(item)

    async def save_run(self, run: RunRecord) -> None:
        item = await self.get_conversation(run.conversation_id)
        if item is None or item.tenant_id != run.tenant_id:
            raise ValueError("Run tenant does not match its conversation tenant")
        if run.run_id not in item.run_ids:
            item.run_ids.append(run.run_id)
        item.updated_at = utc_now()
        await self._save_conversation(item)
        await self._execute(
            "INSERT INTO agent_runs (id, conversation_id, tenant_id, principals, payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT (id) DO UPDATE SET payload=EXCLUDED.payload, principals=EXCLUDED.principals",
            (
                run.run_id,
                run.conversation_id,
                run.tenant_id,
                list(run.principal_ids),
                run.model_dump_json(),
            ),
        )

    async def get_run(
        self, run_id: str, tenant_id: str | None = None, principal_ids: set[str] | None = None
    ) -> RunRecord | None:
        rows = await self._fetch("SELECT payload FROM agent_runs WHERE id=%s", (run_id,))
        if not rows:
            return None
        item = RunRecord.model_validate(rows[0][0])
        return (
            item
            if self._allowed(item.tenant_id, item.principal_ids, tenant_id, principal_ids)
            else None
        )

    async def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str | None = None,
        principal_ids: set[str] | None = None,
    ) -> Conversation | None:
        rows = await self._fetch(
            "SELECT payload FROM agent_conversations WHERE id=%s", (conversation_id,)
        )
        if not rows:
            return None
        item = Conversation.model_validate(rows[0][0])
        return (
            item
            if self._allowed(item.tenant_id, item.principal_ids, tenant_id, principal_ids)
            else None
        )

    async def list_conversations(
        self, tenant_id: str | None = None, principal_ids: set[str] | None = None
    ) -> list[Conversation]:
        rows = await self._fetch(
            "SELECT payload FROM agent_conversations ORDER BY updated_at DESC", ()
        )
        return [
            item
            for (payload,) in rows
            if self._allowed(
                (item := Conversation.model_validate(payload)).tenant_id,
                item.principal_ids,
                tenant_id,
                principal_ids,
            )
        ]

    async def fail_running_run(self, run_id: str, error: str) -> RunRecord | None:
        item = await self.get_run(run_id)
        if item and item.status == RunStatus.RUNNING:
            item.status, item.error, item.completed_at = RunStatus.FAILED, error, utc_now()
            await self.save_run(item)
        return item

    async def _save_conversation(self, item: Conversation) -> None:
        await self._execute(
            "INSERT INTO agent_conversations (id, tenant_id, principals, updated_at, payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT (id) DO UPDATE SET updated_at=EXCLUDED.updated_at, principals=EXCLUDED.principals, payload=EXCLUDED.payload",
            (
                item.conversation_id,
                item.tenant_id,
                list(item.principal_ids),
                item.updated_at,
                item.model_dump_json(),
            ),
        )

    async def _execute(self, query: str, params: tuple[Any, ...]) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(query, params)
            await connection.commit()

    async def _fetch(self, query: str, params: tuple[Any, ...]) -> list[Any]:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchall()

    @staticmethod
    def _allowed(
        actual_tenant: str,
        actual_principals: set[str],
        tenant_id: str | None,
        principal_ids: set[str] | None,
    ) -> bool:
        return (tenant_id is None or actual_tenant == tenant_id) and (
            principal_ids is None or bool(actual_principals.intersection(principal_ids))
        )
