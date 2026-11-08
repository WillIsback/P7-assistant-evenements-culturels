"""Outils LangChain de l'agent : recherche filtrée et vue d'ensemble de l'index régional.

Les outils n'appellent jamais Open Agenda : ils interrogent l'index courant (relu à
chaque appel, pour profiter d'une resynchronisation). Réponse au format
``content_and_artifact`` : un résumé compact pour le LLM, les sources structurées en
artefact (persistées avec le fil par le checkpointer).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from puls_events_rag.core.cities import resolve_commune
from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.rag.chain import to_source
from puls_events_rag.rag.query import SearchQuery

MAX_K = 10


class PeriodArgs(BaseModel):
    communes: list[str] = Field(
        default_factory=list, description="communes de la métropole (noms officiels)"
    )
    since: date | None = Field(default=None, description="début de période AAAA-MM-JJ")
    until: date | None = Field(default=None, description="fin de période AAAA-MM-JJ")


class SearchEventsArgs(PeriodArgs):
    query: str = Field(description="thème ou type d'événement recherché, sans ville ni date")
    k: int = Field(default=5, ge=1, le=MAX_K, description="nombre maximal d'événements")


def _unknown(communes: list[str]) -> list[str]:
    return [name for name in communes if resolve_commune(name) is None]


def _scope(communes: list[str]) -> str:
    unknown = _unknown(communes)
    if not unknown:
        return ""
    return f"Hors de la métropole Aix-Marseille-Provence (non couvert) : {', '.join(unknown)}.\n"


def build_tools(
    get_index: Callable[[], RegionalIndex], *, max_distance: float | None
) -> list[BaseTool]:
    """Outils liés à l'index courant renvoyé par ``get_index``."""

    async def search_events(
        query: str,
        communes: list[str] | None = None,
        since: date | None = None,
        until: date | None = None,
        k: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        communes = communes or []
        search = SearchQuery(query=query, communes=communes, since=since, until=until)
        result = await get_index().search(
            query, k=k, where=search.to_filter(), max_distance=max_distance
        )
        sources = [to_source(event) for event in result.events]
        header = _scope(communes)
        if not sources:
            return header + "Aucun événement pertinent, même aux alentours.", []
        if result.radius_km is not None:
            header += (
                f"Aucun événement pertinent dans {', '.join(communes)} : résultats élargis "
                f"à {result.radius_km:g} km.\n"
            )
        lines = [
            f"- {s.title} | {s.date_range or 'date inconnue'} | {s.location or ''}"
            + (f" | à {s.distance_km} km" if s.distance_km is not None else "")
            + (f" | {s.url}" if s.url else "")
            for s in sources
        ]
        content = header + f"{len(sources)} événement(s) :\n" + "\n".join(lines)
        return content, [source.model_dump() for source in sources]

    async def list_event_themes(
        communes: list[str] | None = None,
        since: date | None = None,
        until: date | None = None,
    ) -> str:
        communes = communes or []
        where = SearchQuery(query="-", communes=communes, since=since, until=until).to_filter()
        facets = get_index().facets(where)
        themes = ", ".join(f"{name} ({count})" for name, count in facets.top_keywords)
        return (
            _scope(communes)
            + f"{facets.event_count} événement(s) correspondent."
            + (f" Thèmes dominants : {themes}." if themes else "")
        )

    return [
        StructuredTool.from_function(
            coroutine=search_events,
            name="search_events",
            description=(
                "Cherche des événements par thème dans l'agenda de la métropole "
                "Aix-Marseille-Provence, filtrés par communes et période. Élargit "
                "automatiquement aux alentours si rien ne correspond dans les communes."
            ),
            args_schema=SearchEventsArgs,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            coroutine=list_event_themes,
            name="list_event_themes",
            description=(
                "Compte les événements d'une zone et d'une période et donne leurs thèmes "
                "dominants. À utiliser quand la demande ne précise pas encore de thème."
            ),
            args_schema=PeriodArgs,
        ),
    ]
