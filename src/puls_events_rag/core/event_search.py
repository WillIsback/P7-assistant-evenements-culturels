"""Recherche ciblée : question utilisateur -> mots-clés -> Open Agenda."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Protocol

from puls_events_rag.core.cities import city_centers, match_cities, strip_cities
from puls_events_rag.core.geo import GeoPoint, bounding_box, distance_to_nearest_km
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient

logger = logging.getLogger(__name__)

# Rayons successifs (km) essayés quand la commune citée n'a aucun événement.
FALLBACK_RADII_KM = (5.0, 10.0, 20.0, 40.0)
DISTANCE_KEY = "distance_km"  # Champ ajouté aux événements : distance à la commune citée.


class KeywordExtractor(Protocol):
    """Ce qu'attend la recherche ciblée (implémenté par ``LangOpenAI``)."""

    async def get_keywords(self, query: str) -> list[str]: ...


async def find_events(
    question: str,
    *,
    llm: KeywordExtractor,
    client: OpenAgendaClient,
    since: date | datetime | None = None,
    until: date | datetime | None = None,
    size: int = 20,
    detailed: bool = False,
) -> list[dict[str, Any]]:
    """Extrait les mots-clés de ``question`` puis interroge Open Agenda.

    Les mots-clés sont passés au paramètre ``search`` (pertinence) plutôt qu'à
    ``keyword[]`` (ET strict) : un LLM produit du texte libre, pas la taxonomie
    normalisée d'Open Agenda. Une liste vide n'appelle pas l'API. ``detailed``
    ajoute la description longue, utile à l'indexation RAG.

    Les communes de la métropole citées dans la question (référentiel
    ``communes.json``, croisé par regex) deviennent un filtre ``city[]`` ; elles
    sont alors retirées du texte ``search``, déjà garanti par le filtre.

    Si ces communes n'ont **aucun** événement, on élargit à la zone : rectangle
    autour de leur centre, de rayon croissant (:data:`FALLBACK_RADII_KM`), puis
    tri par distance réelle. Chaque événement reçoit alors ``distance_km``.
    """
    keywords = await llm.get_keywords(question)
    if not keywords:
        logger.info("Aucun mot-clé extrait de %r : pas d'appel Open Agenda", question)
        return []

    cities = match_cities(question)
    search_terms = strip_cities(keywords, cities) if cities else keywords
    logger.info(
        "Recherche Open Agenda pour %r -> search=%s city=%s", question, search_terms, cities
    )
    common: dict[str, Any] = {
        "search": " ".join(search_terms),
        "since": since,
        "until": until,
        "size": size,
        "detailed": detailed,
    }
    events = await client.get_events(cities=cities, **common)
    centers = city_centers(cities)
    if events or not centers:
        return _with_distance(events, centers)

    for radius in FALLBACK_RADII_KM:
        events = await client.get_events(area=bounding_box(centers, radius), **common)
        if events:
            logger.info("Aucun événement à %s : %d trouvé(s) à %g km", cities, len(events), radius)
            return sorted(
                _with_distance(events, centers),
                key=lambda event: event.get(DISTANCE_KEY, float("inf")),
            )
    logger.info("Aucun événement à moins de %g km de %s", FALLBACK_RADII_KM[-1], cities)
    return []


def _with_distance(events: list[dict[str, Any]], centers: list[GeoPoint]) -> list[dict[str, Any]]:
    """Ajoute ``distance_km`` (arrondie à 0,1 km) aux événements géolocalisés."""
    if not centers:
        return events
    for event in events:
        distance = distance_to_nearest_km(event, centers)
        if distance is not None:
            event[DISTANCE_KEY] = round(distance, 1)
    return events
