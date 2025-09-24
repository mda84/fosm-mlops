"""Streaming simulator for fiber-optic sensor data."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class StreamConfig:
    source_path: Path
    interval: float = 0.5
    batch_size: int = 1024


class StreamSimulator:
    """Simulate Kafka-like streaming using asyncio queues."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def producer(self) -> None:
        df = pd.read_parquet(self.config.source_path)
        for start in range(0, len(df), self.config.batch_size):
            chunk = df.iloc[start : start + self.config.batch_size]
            payload = chunk.to_json(orient="records")
            await self._queue.put(payload)
            await asyncio.sleep(self.config.interval)
        await self._queue.put("__STREAM_END__")

    async def consumer(self) -> AsyncIterator[pd.DataFrame]:
        while True:
            payload = await self._queue.get()
            if payload == "__STREAM_END__":
                break
            yield pd.read_json(payload)

    async def run(self, handler: Callable[[pd.DataFrame], None]) -> None:
        async def _consume() -> None:
            async for batch in self.consumer():
                handler(batch)

        await asyncio.gather(self.producer(), _consume())


async def stream_to_disk(
    config: StreamConfig,
    output_dir: Path,
    metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sink = output_dir / f"stream_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jsonl"

    def handler(batch: pd.DataFrame) -> None:
        batch.to_json(sink, orient="records", lines=True, mode="a")

    simulator = StreamSimulator(config)
    await simulator.run(handler)
    if metadata:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
