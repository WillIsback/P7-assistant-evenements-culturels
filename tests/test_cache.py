"""Tests du cache (backend mémoire, aucun appel réseau)."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import httpx
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from puls_events_rag.core.cache import CachedEmbeddings, make_key
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient
from puls_events_rag.core.providers.openai_api import KeywordExtractionError, LangOpenAI


def test_make_key_stable_et_sensible_aux_parametres() -> None:
    assert make_key("ns", {"b": 1, "a": 2}) == make_key("ns", {"a": 2, "b": 1})
    assert make_key("ns", "modele-a", "q") != make_key("ns", "modele-b", "q")
    assert make_key("ns", "q").startswith("puls:ns:")


class _CountingEmbeddings(Embeddings):
    """Faux modèle qui enregistre les textes réellement envoyés."""

    def __init__(self) -> None:
        self.fake = DeterministicFakeEmbedding(size=8)
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.fake.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append([text])
        return self.embed_query(text)


def test_embeddings_ne_calcule_que_les_textes_nouveaux() -> None:
    underlying = _CountingEmbeddings()
    embeddings = CachedEmbeddings(underlying, model="fake")

    async def run() -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
        first = await embeddings.aembed_documents(["a", "b"])
        second = await embeddings.aembed_documents(["b", "c", "a"])
        q1 = await embeddings.aembed_query("question")
        q2 = await embeddings.aembed_query("question")
        return first, second, q1, q2

    first, second, q1, q2 = asyncio.run(run())

    assert underlying.calls == [["a", "b"], ["c"], ["question"]]
    assert second == [first[1], underlying.embed_query("c"), first[0]]
    assert q1 == q2


def test_embeddings_cle_depend_du_modele() -> None:
    underlying = _CountingEmbeddings()

    async def run() -> None:
        await CachedEmbeddings(underlying, model="m1").aembed_documents(["a"])
        await CachedEmbeddings(underlying, model="m2").aembed_documents(["a"])

    asyncio.run(run())

    assert underlying.calls == [["a"], ["a"]]


def test_mots_cles_mis_en_cache_par_question_normalisee() -> None:
    llm = LangOpenAI(FakeListChatModel(responses=['{"keywords": ["concert"]}']))

    async def run() -> tuple[list[str], list[str]]:
        # Une seule réponse disponible : le 2e appel doit venir du cache.
        return await llm.get_keywords("Un concert ?"), await llm.get_keywords("  un  CONCERT ? ")

    assert asyncio.run(run()) == (["concert"], ["concert"])


def test_reponse_hors_contrat_jamais_mise_en_cache() -> None:
    llm = LangOpenAI(FakeListChatModel(responses=["pas du json", '{"keywords": ["expo"]}']))

    async def run() -> list[str]:
        with contextlib.suppress(KeywordExtractionError):
            await llm.get_keywords("Une expo")
        return await llm.get_keywords("Une expo")

    assert asyncio.run(run()) == ["expo"]


def _counting_client(requests: list[httpx.Request]) -> OpenAgendaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"events": [{"uid": len(requests)}]})

    return OpenAgendaClient(
        api_key="cle-test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def test_open_agenda_requete_identique_servie_par_le_cache() -> None:
    requests: list[httpx.Request] = []
    client = _counting_client(requests)

    async def run() -> list[list[dict[str, object]]]:
        return [
            await client.get_events(search="concert"),
            await client.get_events(search="concert"),  # date par défaut stable : cache
            await client.get_events(search="expo"),
        ]

    first, second, third = asyncio.run(run())

    assert len(requests) == 2
    assert first == second == [{"uid": 1}]
    assert third == [{"uid": 2}]


def test_open_agenda_date_par_defaut_arrondie_au_jour() -> None:
    requests: list[httpx.Request] = []
    asyncio.run(_counting_client(requests).get_events(search="concert"))

    since = datetime.fromisoformat(requests[0].url.params["timings[gte]"])
    assert (since.hour, since.minute, since.second, since.microsecond) == (0, 0, 0, 0)
    assert since.tzinfo == UTC
