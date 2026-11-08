"""Tests d'intégration : vrais appels au LLM et à l'API Open Agenda.

Exclus par défaut ; à lancer avec ``uv run pytest -m integration --no-cov``.
Ils nécessitent les secrets de ``.env`` (LLM + ``OPENAGENDA_API_KEY``).

Un même event loop est partagé par les tests du module : LangChain met en
cache le client HTTP asynchrone par défaut, qui reste lié au loop qui l'a créé.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any

import pytest

from puls_events_rag.core.event_search import find_events
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient
from puls_events_rag.core.providers.openai_api import LangOpenAI

pytestmark = pytest.mark.integration

QUESTION = "Quel événement a eu lieu au Jas de Bouffan ?"


@pytest.fixture(scope="module")
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Un seul event loop pour tout le module (client HTTP LangChain partagé)."""
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    try:
        yield event_loop
    finally:
        event_loop.close()
        asyncio.set_event_loop(None)


def _has_llm_credentials() -> bool:
    profile = (os.getenv("LLM_PROFILE") or "local").lower()
    if profile == "ci":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("VLLM_BASE_URL") or os.getenv("LLM_URL"))


def _has_credentials() -> bool:
    return _has_llm_credentials() and bool(os.getenv("OPENAGENDA_API_KEY"))


def test_extrait_des_mots_cles_reels(loop: asyncio.AbstractEventLoop) -> None:
    if not _has_llm_credentials():
        pytest.skip("secrets LLM absents")

    keywords = loop.run_until_complete(LangOpenAI().get_keywords(QUESTION))

    assert keywords, "le LLM doit extraire au moins un mot-clé"
    assert all(isinstance(keyword, str) and keyword.strip() for keyword in keywords)
    assert len(keywords) <= 8
    assert any("jas de bouffan" in keyword.casefold() for keyword in keywords)


def test_recherche_reelle_de_bout_en_bout(loop: asyncio.AbstractEventLoop) -> None:
    if not _has_credentials():
        pytest.skip("secrets LLM et OPENAGENDA_API_KEY requis")

    async def run() -> list[dict[str, Any]]:
        async with OpenAgendaClient() as client:
            return await find_events(QUESTION, llm=LangOpenAI(), client=client, size=5)

    events = loop.run_until_complete(run())

    assert events, "Open Agenda doit renvoyer au moins un événement"
    assert len(events) <= 5
    assert all("uid" in event for event in events)


def test_retrieval_reel_sur_la_question_entiere(loop: asyncio.AbstractEventLoop) -> None:
    if not (_has_credentials() and os.getenv("OPENROUTER_API_KEY")):
        pytest.skip("secrets LLM, OPENAGENDA_API_KEY et OPENROUTER_API_KEY requis")

    from puls_events_rag.core.semantic_search import retrieve_events

    async def run() -> list[Any]:
        llm = LangOpenAI()
        async with OpenAgendaClient() as client:
            return await retrieve_events(
                QUESTION, llm=llm, embeddings=llm.embeddings, client=client, k=3, size=20
            )

    results = loop.run_until_complete(run())

    assert results, "le retrieval doit renvoyer au moins un événement"
    assert len(results) <= 3
    assert len({result.uid for result in results}) == len(results)
