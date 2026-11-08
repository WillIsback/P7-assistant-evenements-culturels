"""Tests de la chaîne RAG (#9) : LLM, parseur et index simulés, aucun réseau."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.rag.chain import CLARIFY, NO_RESULT, RagChain
from puls_events_rag.rag.query import LLMQueryParser, SearchQuery, fallback_query
from tests.test_index import AIX, EMBEDDINGS, MODEL, _event

TODAY = date(2026, 11, 8)


class _Parser:
    def __init__(self, query: SearchQuery) -> None:
        self.query = query

    async def __call__(self, question: str, *, today: date) -> SearchQuery:
        return self.query


class _RecordingLLM(FakeListChatModel):
    """Modèle factice qui garde les messages reçus (contexte injecté)."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _call(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> str:
        self.received.append(messages)
        return super()._call(messages, *args, **kwargs)


def _index() -> RegionalIndex:
    events = [
        _event(
            1,
            "Atelier immersion",
            commune="Aix-en-Provence",
            lat=AIX.lat,
            lng=AIX.lng,
            begin="2026-11-12T09:00:00+01:00",
            end="2026-11-12T12:00:00+01:00",
        ),
        _event(2, "Concert", commune="Marseille"),
        _event(3, "Atelier", commune="Venelles", lat=43.58, lng=5.44),
    ]
    return asyncio.run(RegionalIndex.build(events, EMBEDDINGS, model=MODEL))


INDEX = _index()


def _ask(query: SearchQuery, question: str = "question", **kwargs: Any) -> Any:
    llm = _RecordingLLM(responses=["Voici l'atelier « Atelier immersion »."])
    chain = RagChain(index=INDEX, llm=llm, parser=_Parser(query), max_distance=0.5, **kwargs)
    return asyncio.run(chain.ask(question, today=TODAY)), llm


def test_reponse_et_sources_avec_contexte_injecte() -> None:
    answer, llm = _ask(SearchQuery(query="atelier immersion", communes=["Aix"]))

    assert answer.answer == "Voici l'atelier « Atelier immersion »."
    assert [source.uid for source in answer.sources] == ["1"]
    assert answer.sources[0].commune == "Aix-en-Provence"
    system = llm.received[0][0].content
    assert "Titre : Atelier immersion" in system
    assert "2026-11-08" in system
    assert answer.latency_ms >= 0


def test_aucun_resultat_sans_appel_llm() -> None:
    answer, llm = _ask(SearchQuery(query="football", communes=["Aix-en-Provence"]))

    assert answer.answer == NO_RESULT
    assert answer.sources == []
    assert llm.received == []


def test_elargissement_annonce_au_llm() -> None:
    answer, llm = _ask(
        SearchQuery(
            query="atelier",
            communes=["Aix-en-Provence"],
            since=date(2026, 11, 9),
            until=date(2026, 11, 11),
        )
    )

    assert answer.radius_km == 10.0
    assert [source.uid for source in answer.sources] == ["3"]
    assert (
        "Aucun événement pertinent à Aix-en-Provence : commence par le dire"
        in llm.received[0][0].content
    )
    assert answer.sources[0].distance_km is not None


def test_filtre_de_periode_et_communes_du_referentiel() -> None:
    query = SearchQuery(
        query="atelier",
        communes=["aix", "Lyon"],
        since=date(2026, 11, 8),
        until=date(2026, 11, 23),
    )

    where = query.to_filter()

    assert where.communes == ("Aix-en-Provence",)
    assert where.since == datetime(2026, 11, 8, tzinfo=UTC)
    assert where.until is not None and where.until.date() == date(2026, 11, 23)


def test_repli_sans_llm() -> None:
    query = fallback_query("Des ateliers à Aix ?")

    assert query.query == "Des ateliers à Aix ?"
    assert query.communes == ["Aix-en-Provence"]
    assert query.since is None


class _BrokenLLM(FakeListChatModel):
    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return FakeListChatModel(responses=["pas du JSON"])


def test_parseur_llm_hors_contrat_se_replie() -> None:
    parser = LLMQueryParser(_BrokenLLM(responses=[]))

    query = asyncio.run(parser("Un concert à Marseille", today=TODAY))

    assert query == fallback_query("Un concert à Marseille")


def test_sans_theme_pas_de_seuil() -> None:
    answer, _ = _ask(SearchQuery(query="", communes=["Aix-en-Provence"]))

    assert [source.uid for source in answer.sources] == ["1"]


def test_lieu_hors_metropole_signale() -> None:
    _, llm = _ask(SearchQuery(query="atelier immersion", communes=["Lyon"]))

    assert "Lyon est hors de la métropole" in llm.received[0][0].content


def test_sans_theme_ni_filtre_demande_de_preciser() -> None:
    answer, llm = _ask(SearchQuery(query=""), question="Quel temps fera-t-il demain ?")

    assert answer.answer == CLARIFY
    assert answer.sources == []
    assert llm.received == []


def test_aucun_resultat_hors_metropole() -> None:
    answer, _ = _ask(SearchQuery(query="football", communes=["Lyon"]))

    assert answer.answer.startswith("Lyon est hors de la métropole")
