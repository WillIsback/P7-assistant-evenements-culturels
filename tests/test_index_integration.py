"""Intégration : vraie collecte Open Agenda (une page) et vrais embeddings.

Exclu par défaut ; ``uv run pytest -m integration --no-cov`` avec ``.env`` renseigné.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from puls_events_rag.core.index import EventFilter, RegionalIndex
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient
from puls_events_rag.core.providers.openai_api import cached_embeddings

pytestmark = pytest.mark.integration


def test_index_reel_sur_une_page(tmp_path: Path) -> None:
    if not (os.getenv("OPENAGENDA_API_KEY") and os.getenv("OPENROUTER_API_KEY")):
        pytest.skip("secrets Open Agenda / OpenRouter absents")

    async def run() -> tuple[RegionalIndex, int]:
        async with OpenAgendaClient() as client:
            page = await anext(client.iter_events(page_size=50))
        embeddings = cached_embeddings()
        index = await RegionalIndex.build(page, embeddings, model=embeddings.model)
        index.save(tmp_path / "index")
        loaded = RegionalIndex.load(tmp_path / "index", embeddings, model=embeddings.model)
        result = await loaded.search("atelier", k=3, where=EventFilter())
        return loaded, len(result.events)

    # Un seul event loop : LangChain garde le client HTTP asynchrone lié au loop.
    index, found = asyncio.run(run())

    assert index.manifest.event_count > 0
    assert index.manifest.dimension == 1024
    assert index.facets().event_count == index.manifest.event_count
    assert found > 0
