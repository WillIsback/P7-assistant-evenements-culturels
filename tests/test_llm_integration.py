"""Tests d'intégration : vrais appels au LLM.

Exclus par défaut ; à lancer avec ``uv run pytest -m integration --no-cov``.
Ils nécessitent les secrets LLM de ``.env``.

Un même event loop est partagé par les tests du module : LangChain met en
cache le client HTTP asynchrone par défaut, qui reste lié au loop qui l'a créé.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import pytest

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


def test_extrait_des_mots_cles_reels(loop: asyncio.AbstractEventLoop) -> None:
    if not _has_llm_credentials():
        pytest.skip("secrets LLM absents")

    keywords = loop.run_until_complete(LangOpenAI().get_keywords(QUESTION))

    assert keywords, "le LLM doit extraire au moins un mot-clé"
    assert all(isinstance(keyword, str) and keyword.strip() for keyword in keywords)
    assert len(keywords) <= 8
    assert any("jas de bouffan" in keyword.casefold() for keyword in keywords)
