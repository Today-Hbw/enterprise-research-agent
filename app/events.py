from __future__ import annotations

import logging

from redis.asyncio import Redis

from app.models import StreamEvent

logger = logging.getLogger(__name__)


class RedisEventBuffer:
    def __init__(self, url: str, ttl_seconds: int) -> None:
        logger.info("Connecting to Redis event buffer: ttl=%ds", ttl_seconds)
        self.client = Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    async def append(self, event: StreamEvent) -> None:
        key = f"agent:run-events:{event.run_id}"
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, event.model_dump_json())
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()
        logger.debug(
            "Appended event %s (seq=%d) to Redis for run %s",
            event.event,
            event.sequence,
            event.run_id,
        )

    async def after(self, run_id: str, sequence: int) -> list[StreamEvent]:
        raw = await self.client.lrange(f"agent:run-events:{run_id}", 0, -1)
        events = [
            event
            for item in raw
            if (event := StreamEvent.model_validate_json(item)).sequence > sequence
        ]
        logger.debug(
            "Replayed %d event(s) from Redis for run %s (after seq=%d)",
            len(events),
            run_id,
            sequence,
        )
        return events

    async def aclose(self) -> None:
        logger.info("Closing Redis event buffer connection")
        await self.client.aclose()
