"""Tests unitaires de l'extraction de mots-clés (aucun appel réseau)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from puls_events_rag.core.providers.openai_api import (
    KeywordExtractionError,
    LangOpenAI,
)


def _llm(response: str) -> LangOpenAI:
    return LangOpenAI(FakeListChatModel(responses=[response]))


def test_extrait_nettoie_et_deduplique_les_mots_cles() -> None:
    llm = _llm('{"keywords": ["Jas de Bouffan", " jas de bouffan ", "Aix-en-Provence"]}')

    keywords = asyncio.run(llm.get_keywords("Quel evenement a eu lieu au Jas de Bouffan"))

    assert keywords == ["Jas de Bouffan", "Aix-en-Provence"]


def test_accepte_un_json_entoure_de_balises_markdown() -> None:
    llm = _llm('```json\n{"keywords": ["exposition", "gratuit"]}\n```')

    assert asyncio.run(llm.get_keywords("exposition gratuite")) == ["exposition", "gratuit"]


def test_rejette_une_reponse_hors_contrat() -> None:
    llm = _llm("Voici les mots-clés : concert")

    with pytest.raises(KeywordExtractionError):
        asyncio.run(llm.get_keywords("concert"))


def test_question_vide_sans_appel_llm() -> None:
    assert asyncio.run(_llm("inutilisé").get_keywords("   ")) == []


def test_build_embeddings_utilise_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    from puls_events_rag.core.providers.openai_api import build_embeddings

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    embeddings = build_embeddings()

    assert embeddings.model == "mistralai/mistral-embed-2312"
    assert embeddings.openai_api_base == "https://openrouter.ai/api/v1"
    assert embeddings.check_embedding_ctx_length is False


def test_build_embeddings_exige_la_cle(monkeypatch: pytest.MonkeyPatch) -> None:
    from puls_events_rag.core.providers.openai_api import build_embeddings

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_embeddings()
