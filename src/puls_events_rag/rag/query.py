"""Question en langage naturel → requête de recherche structurée (thème, communes, période).

Le LLM produit un :class:`SearchQuery` (sortie structurée LangChain) en connaissant
la date du jour, ce qui résout « ce week-end » ou « les 15 prochains jours ». Les
communes sont ramenées au référentiel ; en cas d'échec du LLM, repli déterministe :
la question entière comme requête et les communes repérées par regex.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, time
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from puls_events_rag.core.cities import match_cities, normalize, resolve_commune
from puls_events_rag.core.index import EventFilter

logger = logging.getLogger(__name__)

PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu transformes une question sur des événements de la métropole Aix-Marseille-"
            "Provence en requête de recherche. Nous sommes le {today} ({weekday}).\n"
            "- query : une courte phrase EN FRANÇAIS décrivant ce que la personne cherche, "
            "sans la ville ni les dates (ex. « atelier pour refaire son CV », « forum de "
            "recrutement ») ; traduis si la question est dans une autre langue ; chaîne vide "
            "si la question ne précise aucun thème (« des événements à Aix ? », « qu'est-ce "
            "qu'il y a ? ») ;\n"
            "- communes : tous les lieux cités, même hors de la métropole, liste vide si aucun ;\n"
            "- about_events : faux si la question ne demande pas des événements, sorties ou "
            "activités (météo, culture générale, code, recette, abonnement…) ;\n"
            "- since / until : la période demandée en dates ISO (AAAA-MM-JJ), null si la "
            "question n'en précise pas. « les 15 prochains jours » : since = aujourd'hui, "
            "until = aujourd'hui + 15 jours.",
        ),
        ("human", "{question}"),
    ]
)

# Indices temporels : sans eux, une période renvoyée par le LLM est inventée.
_TEMPORAL = re.compile(
    r"\b(aujourd|demain|hier|soir|matin|week|semaine|mois|jours?|annee|ete|hiver|"
    r"printemps|automne|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|"
    r"octobre|novembre|decembre|prochaine?s?|noel|vacances|today|tomorrow|month|"
    r"weekend|\d{1,2}(er)?|\d{4})\b"
)

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


class SearchQuery(BaseModel):
    """Requête de recherche extraite d'une question."""

    query: str = Field(
        default="", description="thème recherché en une courte phrase, vide si aucun"
    )
    communes: list[str] = Field(default_factory=list, description="communes citées")
    about_events: bool = Field(
        default=True, description="vrai si la question demande des événements ou sorties"
    )
    since: date | None = Field(default=None, description="début de la période (inclus)")
    until: date | None = Field(default=None, description="fin de la période (incluse)")

    @property
    def has_theme(self) -> bool:
        return bool(self.query.strip())

    @property
    def outside_communes(self) -> list[str]:
        """Lieux cités hors du référentiel de la métropole (non couverts)."""
        return [name for name in self.communes if resolve_commune(name) is None]

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
        if not isinstance(parsed, SearchQuery):
            return fallback_query(question)
        return ground(parsed, question)


def ground(parsed: SearchQuery, question: str) -> SearchQuery:
    """Ne garde que ce que la question dit vraiment.

    Communes : seulement celles écrites dans la question (référentiel ou texte brut,
    pour signaler « Lyon »), complétées par la détection regex. Période : écartée si la
    question ne contient aucun indice temporel (le LLM en invente parfois une).
    """
    text = normalize(question)
    cited = match_cities(question)
    kept = [
        name
        for name in parsed.communes
        if resolve_commune(name) in cited or (normalize(name) and normalize(name) in text)
    ]
    communes = list(dict.fromkeys([*kept, *cited]))
    update: dict[str, object] = {"communes": communes}
    if not _TEMPORAL.search(text):
        update |= {"since": None, "until": None}
    return parsed.model_copy(update=update)
