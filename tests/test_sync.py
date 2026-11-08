"""Tests de la synchronisation de l'index régional (sans réseau)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from puls_events_rag.core.collect import write_jsonl
from puls_events_rag.core.index import IndexLoadError, RegionalIndex
from puls_events_rag.core.sync import (
    IndexManager,
    IndexSettings,
    IndexUnavailableError,
    RefreshInProgressError,
    sync_index,
)
from tests.test_index import EMBEDDINGS, MODEL, _event

TTL = timedelta(hours=24)


def _index(synced_at: datetime, uid: int = 1) -> RegionalIndex:
    return asyncio.run(
        RegionalIndex.build([_event(uid, "Atelier")], EMBEDDINGS, model=MODEL, now=synced_at)
    )


class _Builder:
    def __init__(self, result: RegionalIndex, gate: asyncio.Event | None = None) -> None:
        self.result = result
        self.gate = gate
        self.calls = 0

    async def __call__(self) -> RegionalIndex:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return self.result


def _manager(tmp_path: Path, existing: RegionalIndex | None, builder: _Builder) -> IndexManager:
    def loader(_: Path) -> RegionalIndex:
        if existing is None:
            raise IndexLoadError("absent")
        return existing

    return IndexManager(IndexSettings(tmp_path, TTL), loader=loader, builder=builder)


def test_index_frais_non_reconstruit(tmp_path: Path) -> None:
    fresh = _index(datetime.now(tz=UTC))
    builder = _Builder(fresh)
    manager = _manager(tmp_path, fresh, builder)

    async def run() -> RegionalIndex:
        return await manager.ensure_fresh()

    assert asyncio.run(run()) is fresh
    assert builder.calls == 0


def test_index_expire_servi_puis_remplace(tmp_path: Path) -> None:
    stale = _index(datetime.now(tz=UTC) - timedelta(hours=30))
    fresh = _index(datetime.now(tz=UTC), uid=2)

    async def run() -> tuple[RegionalIndex, RegionalIndex, int, RegionalIndex | None]:
        gate = asyncio.Event()
        builder = _Builder(fresh, gate)
        manager = _manager(tmp_path, stale, builder)
        first = await manager.ensure_fresh()
        second = await manager.ensure_fresh()  # pas de seconde synchro
        await asyncio.sleep(0)
        gate.set()
        await manager.wait()
        return first, second, builder.calls, manager.current

    first, second, calls, current = asyncio.run(run())

    assert first is stale and second is stale
    assert calls == 1
    assert current is fresh


def test_aucun_index_leve_indisponible(tmp_path: Path) -> None:
    built = _index(datetime.now(tz=UTC))

    async def run() -> int:
        builder = _Builder(built)
        manager = _manager(tmp_path, None, builder)
        with pytest.raises(IndexUnavailableError):
            await manager.ensure_fresh()
        await manager.wait()
        assert manager.current is not None
        return builder.calls

    assert asyncio.run(run()) == 1


def test_refresh_concurrent_refuse(tmp_path: Path) -> None:
    built = _index(datetime.now(tz=UTC))

    async def run() -> None:
        gate = asyncio.Event()
        manager = _manager(tmp_path, None, _Builder(built, gate))
        task = asyncio.create_task(manager.refresh())
        await asyncio.sleep(0)
        with pytest.raises(RefreshInProgressError):
            await manager.refresh()
        gate.set()
        await task

    asyncio.run(run())


def test_echec_de_synchro_de_fond_n_arrete_rien(tmp_path: Path) -> None:
    stale = _index(datetime.now(tz=UTC) - timedelta(hours=30))

    async def failing() -> RegionalIndex:
        raise RuntimeError("Open Agenda en panne")

    async def run() -> RegionalIndex | None:
        manager = IndexManager(
            IndexSettings(tmp_path, TTL), loader=lambda _: stale, builder=failing
        )
        await manager.ensure_fresh()
        await manager.wait()
        return manager.current

    assert asyncio.run(run()) is stale


def test_sync_index_depuis_donnees_brutes(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "events.jsonl"
    events: list[dict[str, Any]] = [_event(1, "Atelier"), _event(2, "Concert")]
    write_jsonl(events, raw)

    index = asyncio.run(
        sync_index(
            embeddings=EMBEDDINGS,
            model=MODEL,
            directory=tmp_path / "index",
            raw_path=raw,
            skip_collect=True,
        )
    )

    assert index.manifest.event_count == 2
    reloaded = RegionalIndex.load(tmp_path / "index", EMBEDDINGS, model=MODEL)
    assert reloaded.manifest == index.manifest


def test_reglages_par_environnement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_DIR", "data/autre-index")
    monkeypatch.setenv("INDEX_TTL_HOURS", "12")

    settings = IndexSettings.from_env()

    assert settings.directory == Path("data/autre-index")
    assert settings.ttl == timedelta(hours=12)
