"""Configuration commune : cache en mémoire, vidé avant chaque test."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from puls_events_rag.core.cache import MEMORY_URL, cache, setup_cache


@pytest.fixture(autouse=True)
def _empty_cache() -> Iterator[None]:
    """Chaque test démarre avec un cache mémoire vide (jamais Valkey)."""
    setup_cache(MEMORY_URL, enabled=True)
    asyncio.run(cache.clear())
    yield
