"""Tests de l'API RAG (/ask, /health, /rebuild) avec services simulés, sans réseau."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from puls_events_rag.api.app import app
from puls_events_rag.api.services import Services
from puls_events_rag.core.index import IndexLoadError, RegionalIndex
from puls_events_rag.core.sync import IndexManager, IndexSettings
from puls_events_rag.rag.query import SearchQuery
from tests.test_index import AIX, EMBEDDINGS, MODEL, _event

INDEX = asyncio.run(
    RegionalIndex.build(
        [_event(1, "Atelier immersion", commune="Aix-en-Provence", lat=AIX.lat, lng=AIX.lng)],
        EMBEDDINGS,
        model=MODEL,
    )
)


class _Parser:
    async def __call__(self, question: str, *, today: date) -> SearchQuery:
        return SearchQuery(query="atelier immersion", communes=["Aix-en-Provence"])


def _services(tmp_path: Path, index: RegionalIndex | None) -> Services:
    async def builder() -> RegionalIndex:
        await asyncio.sleep(3600)  # reconstruction « longue » : reste en cours
        return INDEX

    def loader(_: Path) -> RegionalIndex:
        if index is None:
            raise IndexLoadError("absent")
        return index

    manager = IndexManager(
        IndexSettings(tmp_path, timedelta(hours=24)), loader=loader, builder=builder
    )
    manager.load()
    return Services(
        manager=manager,
        llm=FakeListChatModel(responses=["Un atelier à Aix-en-Provence."]),
        parser=_Parser(),
        llm_profile="ci",
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app.state.services = _services(tmp_path, INDEX)
    yield TestClient(app)
    del app.state.services


def test_ask_repond_avec_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Un atelier à Aix ?", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Un atelier à Aix-en-Provence."
    assert [source["title"] for source in body["sources"]] == ["Atelier immersion"]
    assert body["filters"]["communes"] == ["Aix-en-Provence"]
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"k": 3},
        {"question": "a", "k": 0},
    ],
)
def test_ask_valide_la_question(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_sans_index_renvoie_503(tmp_path: Path) -> None:
    app.state.services = _services(tmp_path, None)
    try:
        response = TestClient(app).post("/ask", json={"question": "Un concert ?"})
    finally:
        del app.state.services

    assert response.status_code == 503


def test_health_detaille_l_index(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["llm_profile"] == "ci"
    assert body["index"]["loaded"] is True
    assert body["index"]["event_count"] == 1
    assert body["index"]["fresh"] is True


def test_rebuild_desactive_sans_jeton_configure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REBUILD_TOKEN", raising=False)

    assert client.post("/rebuild").status_code == 404


def test_rebuild_authentification(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REBUILD_TOKEN", "secret-test")

    assert client.post("/rebuild").status_code == 401
    assert client.post("/rebuild", headers={"Authorization": "Bearer faux"}).status_code == 403
    assert (
        client.get("/rebuild", headers={"Authorization": "Bearer secret-test"}).status_code == 200
    )


class _RecordingManager(IndexManager):
    """Gestionnaire dont la synchro de fond est simulée (pas de tâche réelle)."""

    started = 0
    busy = False

    @property
    def refreshing(self) -> bool:
        return self.busy

    def start_background_refresh(self) -> None:
        self.started += 1
        self.busy = True


def test_rebuild_lance_puis_refuse_un_second(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REBUILD_TOKEN", "secret-test")
    headers = {"Authorization": "Bearer secret-test"}
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    manager = services.manager
    recording = _RecordingManager(
        manager.settings, loader=lambda _: INDEX, builder=lambda: asyncio.sleep(0, INDEX)
    )
    recording.load()
    services.manager = recording

    first = client.post("/rebuild", headers=headers)
    second = client.post("/rebuild", headers=headers)
    status = client.get("/rebuild", headers=headers).json()

    assert first.status_code == 202
    assert second.status_code == 409
    assert recording.started == 1
    assert status["refreshing"] is True
    assert status["event_count"] == 1


def test_metadata_decrit_la_couverture(client: TestClient) -> None:
    body = client.get("/metadata").json()

    assert body["zone"] == "Métropole Aix-Marseille-Provence"
    assert body["lookback_days"] == 365
    assert body["index"]["event_count"] == 1
    assert body["communes"] == [{"commune": "Aix-en-Provence", "events": 1}]
    assert body["llm_profile"] == "ci"
    assert 0 < body["max_distance"] < 1


def test_metadata_sans_index_renvoie_503(tmp_path: Path) -> None:
    app.state.services = _services(tmp_path, None)
    try:
        response = TestClient(app).get("/metadata")
    finally:
        del app.state.services

    assert response.status_code == 503
