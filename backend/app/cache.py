import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass

from .models import AnalysisResult


@dataclass
class CacheItem:
    expires_at: float
    result: AnalysisResult


class ResultCache:
    def __init__(self, ttl_seconds: int, max_items: int = 1_000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> AnalysisResult | None:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= time.monotonic():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return item.result.model_copy(update={"cache_hit": True})

    async def set(self, key: str, result: AnalysisResult) -> None:
        async with self._lock:
            self._items[key] = CacheItem(
                expires_at=time.monotonic() + self.ttl_seconds,
                result=result.model_copy(update={"cache_hit": False}),
            )
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

