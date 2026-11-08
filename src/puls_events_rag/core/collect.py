"""Collecte de tout l'agenda régional (alimentation de l'index, #5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from puls_events_rag.core.providers.open_agenda import DEFAULT_LOOKBACK_DAYS, OpenAgendaClient

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/events.jsonl")


def default_since(now: datetime | None = None) -> datetime:
    """Début de la fenêtre : il y a 365 jours, à minuit UTC."""
    today = (now or datetime.now(tz=UTC)).replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def _ended_before(event: dict[str, Any], since: datetime) -> bool:
    last = event.get("lastTiming")
    end = last.get("end") if isinstance(last, dict) else None
    if not isinstance(end, str):
        return False
    try:
        return datetime.fromisoformat(end) < since
    except ValueError:
        return False


async def collect_events(
    client: OpenAgendaClient, *, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Récupère tous les événements de la fenêtre, dédoublonnés par ``uid``.

    L'API filtre déjà sur ``timings[gte]`` ; on retire en plus tout événement
    dont la dernière date est antérieure à ``since`` (garantie « moins d'un an »).
    """
    since = since or default_since()
    seen: set[Any] = set()
    events: list[dict[str, Any]] = []
    async for page in client.iter_events(since=since, detailed=True):
        for event in page:
            uid = event.get("uid")
            if uid is None or uid in seen or _ended_before(event, since):
                continue
            seen.add(uid)
            events.append(event)
        logger.info("Collecte Open Agenda : %d événement(s)", len(events))
    return events


def write_jsonl(events: Iterable[dict[str, Any]], path: Path) -> None:
    """Écrit un événement par ligne (UTF-8), en créant le dossier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Relit un fichier écrit par :func:`write_jsonl`."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
