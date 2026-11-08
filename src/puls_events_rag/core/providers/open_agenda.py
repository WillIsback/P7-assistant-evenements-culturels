"""Recherche d'événements via l'API Open Agenda v2."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime, timedelta
from types import TracebackType
from typing import Any, Self

import httpx
from dotenv import load_dotenv

from puls_events_rag.core.cache import TTL_OPEN_AGENDA, cache_get, cache_set, make_key
from puls_events_rag.core.geo import BoundingBox

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openagenda.com/v2"
DEFAULT_AGENDA_UID = "21769447"
DEFAULT_LOOKBACK_DAYS = 365
MAX_PAGE_SIZE = 300
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class OpenAgendaError(RuntimeError):
    """Erreur de configuration ou de réponse de l'API Open Agenda."""


def _env(name: str) -> str | None:
    """Lit une variable d'environnement non vide."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _to_iso(value: date | datetime) -> str:
    return value.isoformat()


class OpenAgendaClient:
    """Client de lecture des événements d'un agenda.

    Le client HTTP est injectable (``httpx.MockTransport`` en test) et l'objet
    s'utilise comme gestionnaire de contexte asynchrone ::

        async with OpenAgendaClient() as client:
            events = await client.get_events(search="Jas de Bouffan")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agenda_uid: str | int | None = None,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.agenda_uid = str(agenda_uid or _env("OPENAGENDA_AGENDA_UID") or DEFAULT_AGENDA_UID)
        self.events_url = f"{BASE_URL}/agendas/{self.agenda_uid}/events"
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            key = api_key or _env("OPENAGENDA_API_KEY")
            if key is None:
                raise OpenAgendaError("OPENAGENDA_API_KEY manquante")
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"key": key},
            )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Ferme le client HTTP si c'est nous qui l'avons créé."""
        if self._owns_client:
            await self._client.aclose()

    async def get_events(
        self,
        *,
        keywords: Sequence[str] = (),
        search: str | None = None,
        cities: Sequence[str] = (),
        area: BoundingBox | None = None,
        since: date | datetime | None = None,
        until: date | datetime | None = None,
        size: int = 20,
        detailed: bool = False,
    ) -> list[dict[str, Any]]:
        """Recherche des événements de l'agenda.

        ``keywords`` alimente le paramètre ``keyword[]`` : l'API combine les
        termes en **ET** logique, donc ils doivent correspondre à des facettes
        réellement indexées par Open Agenda.

        ``search`` fait une recherche syntaxique classée par pertinence : c'est
        le bon paramètre pour du texte libre (lieu, nom propre), et donc pour
        des mots-clés extraits par un LLM.

        ``cities`` alimente ``city[]`` (valeur exacte de ``location.adminLevel4``) ;
        plusieurs communes sont combinées en **OU**. ``area`` restreint à un
        rectangle géographique (``geo[northEast]`` / ``geo[southWest]``).

        Sans ``since`` ni ``until``, on limite aux 365 derniers jours (arrondi
        au jour, pour que la requête — et donc sa clé de cache — reste stable).
        Les réponses valides sont mises en cache :data:`TTL_OPEN_AGENDA` secondes.
        """
        if not 1 <= size <= MAX_PAGE_SIZE:
            raise ValueError(f"size doit être compris entre 1 et {MAX_PAGE_SIZE}")

        params: dict[str, str | list[str]] = {}
        cleaned_keywords = [" ".join(keyword.split()) for keyword in keywords]
        cleaned_keywords = [keyword for keyword in cleaned_keywords if keyword]
        if cleaned_keywords:
            params["keyword[]"] = cleaned_keywords

        if cleaned_cities := [city.strip() for city in cities if city.strip()]:
            params["city[]"] = cleaned_cities
        if area is not None:
            params.update(area.to_params())

        if search is not None and (text := search.strip()):
            params["search"] = text
            params["threshold"] = "auto"

        if since is None and until is None:
            today = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            since = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        if since is not None:
            params["timings[gte]"] = _to_iso(since)
        if until is not None:
            params["timings[lte]"] = _to_iso(until)

        params["size"] = str(size)
        if detailed:
            params["detailed"] = "1"

        logger.info(
            "Open Agenda : keywords=%s search=%r cities=%s area=%s since=%s",
            list(keywords),
            search,
            list(cities),
            area,
            since,
        )
        key = make_key("openagenda", self.events_url, params)
        cached = await cache_get(key)
        if isinstance(cached, list):
            logger.info("%d événement(s) Open Agenda en cache", len(cached))
            return cached

        try:
            response = await self._client.get(self.events_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenAgendaError(f"Échec de la requête Open Agenda : {exc}") from exc

        payload = self._parse_json(response)
        events = payload.get("events")
        if not isinstance(events, list):
            raise OpenAgendaError("Réponse Open Agenda inattendue : clé 'events' absente")

        logger.info("%d événement(s) renvoyé(s) sur %s", len(events), payload.get("total"))
        await cache_set(key, events, TTL_OPEN_AGENDA)
        return events

    async def iter_events(
        self,
        *,
        since: date | datetime | None = None,
        detailed: bool = True,
        page_size: int = MAX_PAGE_SIZE,
        max_retries: int = 3,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Parcourt **tout** l'agenda, page par page (collecte de l'index régional).

        Pagination par curseur : chaque réponse fournit ``after``, renvoyé tel
        quel (``after[]``) pour obtenir la page suivante ; la collecte s'arrête
        sur une page vide ou sans curseur. Sans ``since``, fenêtre des 365
        derniers jours. Les erreurs transitoires (429, 5xx, réseau) sont
        réessayées ``max_retries`` fois avec une attente exponentielle. Pas de
        cache : la fraîcheur est gérée par l'index lui-même.
        """
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size doit être compris entre 1 et {MAX_PAGE_SIZE}")
        if since is None:
            today = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            since = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        base: dict[str, str | list[str]] = {
            "timings[gte]": _to_iso(since),
            "size": str(page_size),
        }
        if detailed:
            base["detailed"] = "1"
        after: list[str] = []
        while True:
            params = {**base, "after[]": after} if after else dict(base)
            payload = await self._get_with_retry(params, max_retries)
            events = payload.get("events")
            if not isinstance(events, list):
                raise OpenAgendaError("Réponse Open Agenda inattendue : clé 'events' absente")
            if not events:
                return
            yield events
            cursor = payload.get("after")
            if not isinstance(cursor, list) or not cursor:
                return
            after = [str(value) for value in cursor]

    async def _get_with_retry(
        self, params: dict[str, str | list[str]], max_retries: int
    ) -> dict[str, Any]:
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.get(self.events_url, params=params)
            except httpx.TransportError as exc:
                error: Exception = exc
            else:
                if response.status_code not in RETRY_STATUSES:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as exc:
                        raise OpenAgendaError(f"Échec de la requête Open Agenda : {exc}") from exc
                    return self._parse_json(response)
                error = OpenAgendaError(f"Open Agenda indisponible (HTTP {response.status_code})")
            if attempt < max_retries:
                delay = 2.0**attempt
                logger.warning("Open Agenda : %s, nouvel essai dans %.0f s", error, delay)
                await asyncio.sleep(delay)
        raise OpenAgendaError(f"Échec de la collecte Open Agenda : {error}") from error

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise OpenAgendaError("Réponse Open Agenda invalide : JSON attendu") from exc
        if not isinstance(payload, dict):
            raise OpenAgendaError("Réponse Open Agenda inattendue : objet JSON attendu")
        return payload
