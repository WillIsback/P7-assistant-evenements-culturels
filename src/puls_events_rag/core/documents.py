"""Événements Open Agenda -> documents LangChain prêts à indexer (#6).

Chaque événement devient un ou plusieurs passages : un en-tête (titre, dates,
lieu, mots-clés) suivi d'un morceau de description. Les métadonnées portent ce
qu'il faut pour **citer** l'événement et pour le **filtrer au retrieval** :
commune officielle (référentiel ``communes.json``), coordonnées, dates ISO.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from puls_events_rag.core.cities import resolve_commune
from puls_events_rag.core.geo import event_point

logger = logging.getLogger(__name__)

LANGUAGE = "fr"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
_MARKDOWN = re.compile(r"[*_`#>\\]+")
_HTML_TAG = re.compile(r"<[^>]+>")


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
    text = _MARKDOWN.sub("", _HTML_TAG.sub(" ", text))
    paragraphs = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(p for p in paragraphs if p)


def _location(event: Mapping[str, Any]) -> tuple[str, str]:
    """Renvoie (libellé complet du lieu, ville affichée)."""
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
    raw_location = event.get("location")
    admin = raw_location.get("adminLevel4") if isinstance(raw_location, Mapping) else None
    commune = resolve_commune(str(admin)) if admin else None
    commune = commune or (resolve_commune(city) if city else None)
    point = event_point(event)
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
        "commune": commune,
        "lat": point.lat if point else None,
        "lng": point.lng if point else None,
        "date_range": _localized(event.get("dateRange")),
        "begin": _timing(event, "firstTiming", "begin"),
        "end": _timing(event, "lastTiming", "end"),
        "keywords": keywords,
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
    """Convertit une liste d'événements (doublons d'``uid`` ignorés)."""
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
