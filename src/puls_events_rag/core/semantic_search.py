"""Recherche sémantique :
    événements Open Agenda -> documents -> index Faiss -> retrieval.

Le flux complet est ::

    question --(mots-clés LLM)--> Open Agenda --(events_to_documents)--> Faiss
             \\______________________ retrieval sur la question entière ___/

L'index est construit **à la volée** pour chaque liste d'événements chargée
(une :class:`EventIndex` par client / session) : il n'y a pas d'index global
persisté, donc rien à reconstruire quand l'agenda évolue.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Self

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from puls_events_rag.core.documents import events_to_documents
from puls_events_rag.core.event_search import KeywordExtractor, find_events
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedEvent:
    """Un événement retrouvé, avec le meilleur passage et sa distance
    (L2, plus bas = mieux)."""

    uid: Any
    score: float
    document: Document

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


class EventIndex:
    """Index Faiss en mémoire d'une liste d'événements (un par client / session)."""

    def __init__(self, store: FAISS | None, event_count: int = 0) -> None:
        self._store = store
        self.event_count = event_count

    @classmethod
    async def from_events(cls, events: Sequence[Mapping[str, Any]], embeddings: Embeddings) -> Self:
        """Construit l'index à la volée ; une liste vide donne un index vide
        (sans appel)."""
        documents = events_to_documents(events)
        if not documents:
            logger.info("Aucun événement indexable : index vide")
            return cls(None)
        store = await FAISS.afrom_documents(documents, embeddings)
        event_count = len({doc.metadata["uid"] for doc in documents})
        logger.info("Index Faiss : %d événement(s), %d passage(s)", event_count, len(documents))
        return cls(store, event_count)

    async def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedEvent]:
        """Renvoie au plus ``k`` événements distincts, du plus au moins pertinent.

        Plusieurs passages d'un même événement peuvent remonter : on en demande
        davantage à Faiss puis on garde le meilleur passage par ``uid``.
        """
        if self._store is None or not query.strip() or k < 1:
            return []
        hits = await self._store.asimilarity_search_with_score(query, k=k * 4)
        best: dict[Any, RetrievedEvent] = {}
        for document, score in hits:
            uid = document.metadata["uid"]
            if uid not in best:
                best[uid] = RetrievedEvent(uid=uid, score=float(score), document=document)
            if len(best) == k:
                break
        return list(best.values())


async def retrieve_events(
    question: str,
    *,
    llm: KeywordExtractor,
    embeddings: Embeddings,
    client: OpenAgendaClient,
    k: int = DEFAULT_TOP_K,
    since: date | datetime | None = None,
    until: date | datetime | None = None,
    size: int = 50,
) -> list[RetrievedEvent]:
    """Pipeline complet : mots-clés -> Open Agenda -> index à la volée -> retrieval.

    Open Agenda sert de pré-filtre lexical (``size`` candidats, description
    longue incluse) ; Faiss reclasse ensuite ces candidats sur la **question
    entière**, qui porte plus de sens que les seuls mots-clés.
    """
    events = await find_events(
        question,
        llm=llm,
        client=client,
        since=since,
        until=until,
        size=size,
        detailed=True,
    )
    index = await EventIndex.from_events(events, embeddings)
    return await index.search(question, k=k)
