"""Tests de performance sans réseau (#12) : index de taille réaliste, embeddings factices.

Seuils volontairement larges (machines de CI partagées) ; les mesures réelles sont
consignées dans docs/evaluation/performance.md (``scripts/perf.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from puls_events_rag.core.index import EventFilter, RegionalIndex
from tests.test_index import AIX, _event

EVENTS = 5_000  # ≈ taille de l'agenda régional (4 294 événements mesurés)
DIMENSION = 1024  # mistral-embed
SEARCH_BUDGET_S = 0.5
LOAD_BUDGET_S = 3.0


class HashEmbeddings(Embeddings):
    """Vecteurs pseudo-aléatoires déterministes de dimension réelle, sans réseau."""

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        return [((seed[i % 32] + i) % 97) / 97.0 for i in range(DIMENSION)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture(scope="module")
def saved_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    communes = ("Marseille", "Aix-en-Provence", "Aubagne", "Martigues")
    events = [
        _event(uid, f"Événement {uid}", commune=communes[uid % 4], lat=AIX.lat, lng=AIX.lng)
        for uid in range(EVENTS)
    ]
    index = asyncio.run(RegionalIndex.build(events, HashEmbeddings(), model="hash"))
    directory = tmp_path_factory.mktemp("perf") / "index"
    index.save(directory)
    return directory


def test_chargement_de_l_index(saved_index: Path) -> None:
    started = time.perf_counter()
    index = RegionalIndex.load(saved_index, HashEmbeddings(), model="hash")
    elapsed = time.perf_counter() - started

    assert index.manifest.event_count == EVENTS
    assert elapsed < LOAD_BUDGET_S, f"chargement {elapsed:.2f} s"


def test_recherche_filtree_exhaustive(saved_index: Path) -> None:
    index = RegionalIndex.load(saved_index, HashEmbeddings(), model="hash")
    now = datetime(2026, 11, 8, tzinfo=UTC)
    where = EventFilter(communes=("Aix-en-Provence",), since=now, until=now + timedelta(days=15))

    async def run() -> float:
        await index.search("atelier", k=5, where=where)  # préchauffage
        started = time.perf_counter()
        for _ in range(5):
            await index.search("atelier", k=5, where=where)
        return (time.perf_counter() - started) / 5

    elapsed = asyncio.run(run())

    assert elapsed < SEARCH_BUDGET_S, f"recherche {elapsed * 1000:.0f} ms"
