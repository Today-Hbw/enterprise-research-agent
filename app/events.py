from __future__ import annotations

from redis.asyncio import Redis

from app.models import StreamEvent


class RedisEventBuffer:
    def __init__(self, url: str, ttl_seconds: int) -> None:
        self.client = Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    async def append(self, event: StreamEvent) -> None:
        key = f"agent:run-events:{event.run_id}"
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, event.model_dump_json())
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()

    async def after(self, run_id: str, sequence: int) -> list[StreamEvent]:
        raw = await self.client.lrange(f"agent:run-events:{run_id}", 0, -1)
        return [
            event
            for item in raw
            if (event := StreamEvent.model_validate_json(item)).sequence > sequence
        ]

    async def aclose(self) -> None:
        await self.client.aclose()
