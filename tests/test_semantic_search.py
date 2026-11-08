"""Tests unitaires du traitement RAG des événements (aucun appel réseau)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from puls_events_rag.core.providers.open_agenda import OpenAgendaClient
from puls_events_rag.core.semantic_search import (
    CHUNK_SIZE,
    EventIndex,
    event_to_documents,
    events_to_documents,
    retrieve_events,
)

EMBEDDINGS = DeterministicFakeEmbedding(size=32)


def _event(uid: int = 43496199, **overrides: Any) -> dict[str, Any]:
    """Événement au format réel de l'API Open Agenda v2 (``detailed=1``)."""
    event: dict[str, Any] = {
        "uid": uid,
        "title": {"fr": "Kouss•Ftartchi", "en": "Kouss•Ftartchi"},
        "description": {"fr": "Cuisine et concert au menu !", "en": "Cuisine and concert!"},
        "longDescription": {
            "fr": "Cuisine et concert au menu !  \nLe projet tunisien **Ftartchi** est invité."
        },
        "keywords": {"fr": ["Concert", "Gratuit"]},
        "dateRange": {"fr": "Vendredi 28 août, 14h30"},
        "firstTiming": {"begin": "2026-08-28T14:30:00+02:00", "end": "2026-08-28T16:30:00+02:00"},
        "lastTiming": {"begin": "2026-08-28T14:30:00+02:00", "end": "2026-08-28T16:30:00+02:00"},
        "location": {
            "name": "La Cômerie",
            "address": "176 rue Breteuil, 13006 Marseille",
            "city": "Marseille",
        },
        "age": {"min": 6, "max": None},
        "conditions": {},
        "canonicalUrl": "https://openagenda.com/saison-mediterranee-2026/events/43496199",
    }
    event.update(overrides)
    return event


def test_event_to_documents_construit_texte_et_metadonnees() -> None:
    [doc] = event_to_documents(_event())

    assert doc.page_content.startswith("Titre : Kouss•Ftartchi\nDates : Vendredi 28 août")
    assert "Lieu : La Cômerie, 176 rue Breteuil, 13006 Marseille" in doc.page_content
    assert "Mots-clés : Concert, Gratuit" in doc.page_content
    assert "Public : à partir de 6 ans" in doc.page_content
    assert "Conditions" not in doc.page_content  # champ vide omis
    assert "**" not in doc.page_content  # Markdown retiré
    assert doc.page_content.count("Cuisine et concert au menu !") == 1  # résumé non dupliqué
    assert doc.metadata["uid"] == 43496199
    assert doc.metadata["city"] == "Marseille"
    assert doc.metadata["begin"] == "2026-08-28T14:30:00+02:00"
    assert doc.metadata["url"].endswith("/43496199")


def test_repli_sur_une_autre_langue_et_evenement_minimal() -> None:
    [doc] = event_to_documents({"uid": 1, "title": {"en": "Jazz night"}})

    assert doc.page_content == "Titre : Jazz night"
    assert doc.metadata["location"] == ""


def test_ignore_les_evenements_sans_titre_ou_uid() -> None:
    assert event_to_documents({"uid": 1, "title": {}}) == []
    assert event_to_documents({"title": {"fr": "Sans uid"}}) == []


def test_longue_description_decoupee_avec_en_tete_repete() -> None:
    long_text = " ".join(f"phrase{i}" for i in range(CHUNK_SIZE))
    docs = event_to_documents(_event(longDescription={"fr": long_text}, description={}))

    assert len(docs) > 1
    assert all(doc.page_content.startswith("Titre : Kouss•Ftartchi") for doc in docs)
    assert [doc.metadata["chunk"] for doc in docs] == list(range(len(docs)))


def test_events_to_documents_deduplique_par_uid() -> None:
    docs = events_to_documents([_event(1), _event(1), _event(2), {"uid": 3}])

    assert [doc.metadata["uid"] for doc in docs] == [1, 2]


def test_index_renvoie_des_evenements_distincts_tries() -> None:
    events = [_event(uid, title={"fr": f"Événement {uid}"}) for uid in range(1, 6)]
    events[0]["longDescription"] = {"fr": " ".join(f"mot{i}" for i in range(CHUNK_SIZE))}

    index = asyncio.run(EventIndex.from_events(events, EMBEDDINGS))
    target = events_to_documents([events[2]])[0].page_content
    results = asyncio.run(index.search(target, k=3))

    assert index.event_count == 5
    assert results[0].uid == 3
    assert results[0].score == pytest.approx(0.0, abs=1e-5)
    assert len({r.uid for r in results}) == len(results) == 3
    assert [r.score for r in results] == sorted(r.score for r in results)


def test_index_vide_ne_calcule_aucun_embedding() -> None:
    index = asyncio.run(EventIndex.from_events([], EMBEDDINGS))

    assert index.event_count == 0
    assert asyncio.run(index.search("concert")) == []


class _FakeExtractor:
    async def get_keywords(self, query: str) -> list[str]:
        return ["concert"]


def test_retrieve_events_de_bout_en_bout_sans_reseau() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"total": 2, "events": [_event(1), _event(2)]})

    client = OpenAgendaClient(
        api_key="cle-test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    results = asyncio.run(
        retrieve_events(
            "Un concert à Marseille ?",
            llm=_FakeExtractor(),
            embeddings=EMBEDDINGS,
            client=client,
            k=1,
        )
    )

    assert requests[0].url.params["detailed"] == "1"
    assert requests[0].url.params["search"] == "concert"
    assert len(results) == 1
    assert results[0].metadata["title"] == "Kouss•Ftartchi"
