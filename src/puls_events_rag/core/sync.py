"""Synchronisation de l'index régional : collecte → documents → Faiss, fraîcheur 24 h (#7).

- :func:`sync_index` : la reconstruction complète (``scripts/build_index.py``,
  ``POST /rebuild``, tâche de fond) ;
- :class:`IndexManager` : l'index courant, partagé par l'API. Un index expiré
  continue de servir pendant qu'une **seule** resynchronisation tourne en fond ;
  le nouvel index le remplace d'un bloc quand il est prêt.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from langchain_core.embeddings import Embeddings

from puls_events_rag.core.collect import RAW_PATH, collect_events, read_jsonl, write_jsonl
from puls_events_rag.core.index import IndexLoadError, RegionalIndex
from puls_events_rag.core.providers.open_agenda import OpenAgendaClient

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/index")
DEFAULT_TTL_HOURS = 24.0


class IndexUnavailableError(RuntimeError):
    """Aucun index utilisable pour l'instant (une synchronisation est en cours)."""


class RefreshInProgressError(RuntimeError):
    """Une synchronisation est déjà en cours."""


@dataclass(frozen=True)
class IndexSettings:
    directory: Path
    ttl: timedelta

    @classmethod
    def from_env(cls) -> IndexSettings:
        directory = Path(os.getenv("INDEX_DIR") or DEFAULT_INDEX_DIR)
        hours = float(os.getenv("INDEX_TTL_HOURS") or DEFAULT_TTL_HOURS)
        return cls(directory=directory, ttl=timedelta(hours=hours))


async def sync_index(
    *,
    embeddings: Embeddings,
    model: str,
    directory: Path,
    client: OpenAgendaClient | None = None,
    raw_path: Path = RAW_PATH,
    skip_collect: bool = False,
) -> RegionalIndex:
    """Collecte l'agenda (ou relit ``raw_path``), construit l'index et le sauvegarde."""
    if skip_collect:
        events = read_jsonl(raw_path)
        logger.info("Collecte sautée : %d événement(s) relus depuis %s", len(events), raw_path)
    else:
        if client is None:
            async with OpenAgendaClient() as owned:
                events = await collect_events(owned)
        else:
            events = await collect_events(client)
        write_jsonl(events, raw_path)
    index = await RegionalIndex.build(events, embeddings, model=model)
    index.save(directory)
    return index


class IndexManager:
    """Détient l'index courant et décide quand le resynchroniser."""

    def __init__(
        self,
        settings: IndexSettings,
        *,
        loader: Callable[[Path], RegionalIndex],
        builder: Callable[[], Awaitable[RegionalIndex]],
    ) -> None:
        self.settings = settings
        self._loader = loader
        self._builder = builder
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.current: RegionalIndex | None = None

    def load(self) -> RegionalIndex | None:
        """Charge l'index sur disque s'il existe et est compatible."""
        try:
            self.current = self._loader(self.settings.directory)
        except IndexLoadError as exc:
            logger.warning("Index régional non chargé : %s", exc)
            return None
        return self.current

    @property
    def refreshing(self) -> bool:
        return self._lock.locked()

    def is_fresh(self) -> bool:
        return self.current is not None and self.current.manifest.is_fresh(self.settings.ttl)

    async def ensure_fresh(self) -> RegionalIndex:
        """Index à utiliser maintenant ; lance une synchro de fond s'il est expiré ou absent.

        Lève :class:`IndexUnavailableError` s'il n'existe encore aucun index.
        """
        if self.current is None:
            self.load()
        current = self.current
        if current is not None and current.manifest.is_fresh(self.settings.ttl):
            return current
        self.start_background_refresh()
        if self.current is None:
            raise IndexUnavailableError(
                "Index régional en cours de construction, réessayer plus tard"
            )
        return self.current

    def start_background_refresh(self) -> None:
        """Lance une synchronisation de fond, sauf si une est déjà en cours."""
        if self.refreshing or (self._task is not None and not self._task.done()):
            return
        self._task = asyncio.create_task(self._refresh_quietly())

    async def _refresh_quietly(self) -> None:
        try:
            await self.refresh()
        except RefreshInProgressError:
            pass
        except Exception:  # la synchro de fond ne doit jamais tuer l'API
            logger.exception("Échec de la resynchronisation de l'index régional")

    async def refresh(self) -> RegionalIndex:
        """Reconstruit l'index puis l'échange ; :class:`RefreshInProgressError` si déjà en cours."""
        if self._lock.locked():
            raise RefreshInProgressError("Une synchronisation de l'index est déjà en cours")
        async with self._lock:
            index = await self._builder()
            self.current = index
            logger.info("Index régional remplacé (%d événements)", index.manifest.event_count)
            return index

    async def wait(self) -> None:
        """Attend la fin de la synchronisation de fond éventuelle (tests, arrêt)."""
        if self._task is not None:
            await self._task

    async def stop(self) -> None:
        """Annule la synchronisation de fond éventuelle (arrêt de l'API)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
