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
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Self

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from puls_events_rag.core.event_search import KeywordExtractor, find_events
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient

logger = logging.getLogger(__name__)

LANGUAGE = "fr"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
_MARKDOWN = re.compile(r"[*_`#>\\]+")


def _localized(value: Any, language: str = LANGUAGE) -> str:
    """Renvoie la traduction ``language`` d'un champ multilingue Open Agenda.

    Repli sur la première traduction non vide ; accepte aussi une chaîne brute.
    """
    if isinstance(value, str):
        return _clean(value)
    if not isinstance(value, Mapping) or not value:
        return ""
    preferred = value.get(language)
    if isinstance(preferred, str) and preferred.strip():
        return _clean(preferred)
    for text in value.values():
        if isinstance(text, str) and text.strip():
            return _clean(text)
    return ""


def _localized_list(value: Any, language: str = LANGUAGE) -> list[str]:
    """Mots-clés multilingues : ``{"fr": ["Concert", ...]}`` -> ``["Concert", ...]``."""
    if not isinstance(value, Mapping):
        return []
    items: Any = value.get(language) or next((v for v in value.values() if v), [])
    if not isinstance(items, list):
        return []
    return [_clean(item) for item in items if isinstance(item, str) and item.strip()]


def _clean(text: str) -> str:
    """Retire l'emphase Markdown et normalise les espaces
    (sauts de paragraphe conservés)."""
    text = _MARKDOWN.sub("", text)
    paragraphs = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(p for p in paragraphs if p)


def _location(event: Mapping[str, Any]) -> tuple[str, str]:
    """Renvoie (libellé complet du lieu, ville)."""
    location = event.get("location")
    if not isinstance(location, Mapping):
        return "", ""
    name = location.get("name") or ""
    address = location.get("address") or ""
    city = location.get("city") or ""
    parts = [str(p) for p in (name, address) if p]
    if city and not any(str(city) in p for p in parts):
        parts.append(str(city))
    return ", ".join(parts), str(city)


def _timing(event: Mapping[str, Any], key: str, bound: str) -> str | None:
    timing = event.get(key)
    if isinstance(timing, Mapping) and isinstance(timing.get(bound), str):
        return str(timing[bound])
    return None


def _age(event: Mapping[str, Any]) -> str:
    age = event.get("age")
    if not isinstance(age, Mapping):
        return ""
    low, high = age.get("min"), age.get("max")
    if low is not None and high is not None:
        return f"de {low} à {high} ans"
    if low is not None:
        return f"à partir de {low} ans"
    if high is not None:
        return f"jusqu'à {high} ans"
    return ""


def event_to_documents(event: Mapping[str, Any]) -> list[Document]:
    """Transforme un événement Open Agenda en documents prêts à indexer.

    Chaque document commence par un en-tête (titre, dates, lieu, mots-clés)
    suivi d'un morceau de la description : un fragment de longue description
    reste ainsi rattaché à son événement lors du retrieval. Un événement sans
    titre ni identifiant est ignoré.
    """
    uid = event.get("uid")
    title = _localized(event.get("title"))
    if uid is None or not title:
        return []

    location, city = _location(event)
    keywords = _localized_list(event.get("keywords"))
    header_lines = [
        f"Titre : {title}",
        f"Dates : {_localized(event.get('dateRange'))}",
        f"Lieu : {location}",
        f"Mots-clés : {', '.join(keywords)}",
        f"Public : {_age(event)}",
        f"Conditions : {_localized(event.get('conditions'))}",
    ]
    header = "\n".join(line for line in header_lines if not line.endswith(": "))

    summary = _localized(event.get("description"))
    long_description = _localized(event.get("longDescription"))
    body = "\n".join(text for text in (summary, long_description) if text)
    if summary and long_description.startswith(summary):
        body = long_description  # La longue description reprend souvent le résumé.

    metadata: dict[str, Any] = {
        "uid": uid,
        "title": title,
        "url": event.get("canonicalUrl"),
        "location": location,
        "city": city,
        "date_range": _localized(event.get("dateRange")),
        "begin": _timing(event, "firstTiming", "begin"),
        "end": _timing(event, "lastTiming", "end"),
        "keywords": keywords,
        "distance_km": event.get("distance_km"),
    }

    chunks = _SPLITTER.split_text(body) if body else [""]
    return [
        Document(
            page_content=f"{header}\n\n{chunk}".strip(),
            metadata={**metadata, "chunk": index},
        )
        for index, chunk in enumerate(chunks)
    ]


def events_to_documents(events: Iterable[Mapping[str, Any]]) -> list[Document]:
    """Convertit la sortie de :func:`find_events` (doublons d'``uid`` ignorés)."""
    documents: list[Document] = []
    seen: set[Any] = set()
    for event in events:
        uid = event.get("uid")
        if uid in seen:
            continue
        converted = event_to_documents(event)
        if converted:
            seen.add(uid)
            documents.extend(converted)
    return documents


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
