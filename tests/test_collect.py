"""Tests de la collecte de l'agenda régional."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from puls_events_rag.core.collect import collect_events, read_jsonl, write_jsonl

SINCE = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeClient:
    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.since: object = None

    async def iter_events(self, **kwargs: Any) -> AsyncIterator[list[dict[str, Any]]]:
        self.since = kwargs.get("since")
        for page in self.pages:
            yield page


def _event(uid: int, end: str = "2026-06-01T10:00:00+02:00") -> dict[str, Any]:
    return {"uid": uid, "title": {"fr": f"É{uid}"}, "lastTiming": {"end": end}}


def test_collecte_dedoublonne_par_uid() -> None:
    client = _FakeClient([[_event(1), _event(2)], [_event(2), _event(3)]])

    events = asyncio.run(collect_events(client, since=SINCE))  # type: ignore[arg-type]

    assert [event["uid"] for event in events] == [1, 2, 3]
    assert client.since == SINCE


def test_collecte_retire_les_evenements_de_plus_d_un_an() -> None:
    client = _FakeClient([[_event(1, end="2025-11-01T10:00:00+01:00"), _event(2), {"uid": 3}]])

    events = asyncio.run(collect_events(client, since=SINCE))  # type: ignore[arg-type]

    assert [event["uid"] for event in events] == [2, 3]


def test_jsonl_aller_retour(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "events.jsonl"
    events = [_event(1), {"uid": 2, "title": {"fr": "Fête à Aix"}}]

    write_jsonl(events, path)

    assert read_jsonl(path) == events
