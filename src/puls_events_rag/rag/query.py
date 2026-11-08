"""Question en langage naturel → requête de recherche structurée (thème, communes, période).

Le LLM produit un :class:`SearchQuery` (sortie structurée LangChain) en connaissant
la date du jour, ce qui résout « ce week-end » ou « les 15 prochains jours ». Les
communes sont ramenées au référentiel ; en cas d'échec du LLM, repli déterministe :
la question entière comme requête et les communes repérées par regex.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from puls_events_rag.core.cities import match_cities, resolve_commune
from puls_events_rag.core.index import EventFilter

logger = logging.getLogger(__name__)

PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu transformes une question sur des événements de la métropole Aix-Marseille-"
            "Provence en requête de recherche. Nous sommes le {today} ({weekday}).\n"
            "- query : le thème ou le type d'événement recherché, sans la ville ni les dates "
            "(reprends la question si rien d'autre) ;\n"
            "- communes : les communes citées (noms officiels), liste vide si aucune ;\n"
            "- since / until : la période demandée en dates ISO (AAAA-MM-JJ), null si la "
            "question n'en précise pas. « les 15 prochains jours » : since = aujourd'hui, "
            "until = aujourd'hui + 15 jours.",
        ),
        ("human", "{question}"),
    ]
)

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


class SearchQuery(BaseModel):
    """Requête de recherche extraite d'une question."""

    query: str = Field(description="thème ou type d'événement recherché")
    communes: list[str] = Field(default_factory=list, description="communes citées")
    since: date | None = Field(default=None, description="début de la période (inclus)")
    until: date | None = Field(default=None, description="fin de la période (incluse)")

    def to_filter(self) -> EventFilter:
        """Filtre de l'index : communes du référentiel, période en UTC (jours entiers)."""
        communes = tuple(dict.fromkeys(c for c in map(resolve_commune, self.communes) if c))
        since = datetime.combine(self.since, time.min, tzinfo=UTC) if self.since else None
        until = datetime.combine(self.until, time.max, tzinfo=UTC) if self.until else None
        return EventFilter(communes=communes, since=since, until=until)


class QueryParser(Protocol):
    async def __call__(self, question: str, *, today: date) -> SearchQuery: ...


def fallback_query(question: str) -> SearchQuery:
    """Repli sans LLM : question entière, communes repérées par regex, pas de période."""
    return SearchQuery(query=question, communes=match_cities(question))


class LLMQueryParser:
    """Extraction par sortie structurée (``with_structured_output``)."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = PARSER_PROMPT | llm.with_structured_output(SearchQuery)

    async def __call__(self, question: str, *, today: date) -> SearchQuery:
        try:
            parsed = await self._chain.ainvoke(
                {
                    "question": question,
                    "today": today.isoformat(),
                    "weekday": WEEKDAYS[today.weekday()],
                }
            )
        except Exception as exc:  # LLM indisponible ou hors contrat : on dégrade
            logger.warning("Extraction de la requête impossible (%s) : repli", exc)
            return fallback_query(question)
        if not isinstance(parsed, SearchQuery) or not parsed.query.strip():
            return fallback_query(question)
        # Le LLM peut oublier une commune citée : on complète par la détection regex.
        communes = list(dict.fromkeys([*parsed.communes, *match_cities(question)]))
        return parsed.model_copy(update={"communes": communes})
