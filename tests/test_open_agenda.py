"""Tests unitaires du client Open Agenda et de la recherche ciblée."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from puls_events_rag.core.providers.open_agenda import (
    OpenAgendaClient,
    OpenAgendaError,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    requests: list[httpx.Request],
) -> OpenAgendaClient:
    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return OpenAgendaClient(
        api_key="cle-test",
        agenda_uid=21769447,
        client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
    )


def test_construit_les_filtres_attendus() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 1, "events": [{"uid": 42}]}),
        requests,
    )

    async def run() -> list[dict[str, object]]:
        return await client.get_events(
            keywords=["exposition", "gratuit"],
            since=datetime(2026, 1, 1, tzinfo=UTC),
            size=5,
        )

    events = asyncio.run(run())

    assert events == [{"uid": 42}]
    assert requests[0].url.path == "/v2/agendas/21769447/events"
    params = requests[0].url.params
    assert params.get_list("keyword[]") == ["exposition", "gratuit"]
    assert params["timings[gte]"] == "2026-01-01T00:00:00+00:00"
    assert params["size"] == "5"


def test_recherche_plein_texte_et_lookback_par_defaut() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 0, "events": []}),
        requests,
    )

    asyncio.run(client.get_events(search="  Jas de Bouffan  "))

    params = requests[0].url.params
    assert params["search"] == "Jas de Bouffan"
    assert params["threshold"] == "auto"
    assert "timings[gte]" in params


def test_propage_les_erreurs_http() -> None:
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(401, json={"error": "cle invalide"}), requests)

    with pytest.raises(OpenAgendaError):
        asyncio.run(client.get_events(search="concert"))


def test_rejette_une_reponse_non_json() -> None:
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(200, text="<html>maintenance</html>"), requests)

    with pytest.raises(OpenAgendaError, match="JSON"):
        asyncio.run(client.get_events(search="concert"))


def test_refuse_une_taille_hors_limites() -> None:
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(200, json={}), requests)

    with pytest.raises(ValueError, match="size"):
        asyncio.run(client.get_events(size=0))


def test_cle_open_agenda_manquante(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAGENDA_API_KEY", raising=False)

    with pytest.raises(OpenAgendaError, match="OPENAGENDA_API_KEY"):
        OpenAgendaClient()


def _pages(client: OpenAgendaClient) -> list[list[dict[str, object]]]:
    async def run() -> list[list[dict[str, object]]]:
        return [page async for page in client.iter_events(since=datetime(2026, 1, 1, tzinfo=UTC))]

    return asyncio.run(run())


def test_pagination_suit_le_curseur() -> None:
    pages = [
        {"events": [{"uid": 1}, {"uid": 2}], "after": ["a", "1"]},
        {"events": [{"uid": 3}, {"uid": 4}], "after": ["b", "2"]},
        {"events": [{"uid": 5}]},
    ]
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(200, json=pages[len(requests) - 1]), requests)

    result = _pages(client)

    assert [len(page) for page in result] == [2, 2, 1]
    assert requests[0].url.params.get_list("after[]") == []
    assert requests[1].url.params.get_list("after[]") == ["a", "1"]
    assert requests[2].url.params.get_list("after[]") == ["b", "2"]
    assert requests[0].url.params["size"] == "300"
    assert requests[0].url.params["detailed"] == "1"


def test_pagination_reessaie_puis_reussit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("puls_events_rag.core.providers.open_agenda.asyncio.sleep", _no_sleep)
    requests: list[httpx.Request] = []
    responses = [httpx.Response(429), httpx.Response(200, json={"events": [{"uid": 1}]})]
    client = _client(lambda request: responses[len(requests) - 1], requests)

    assert _pages(client) == [[{"uid": 1}]]
    assert len(requests) == 2


def test_pagination_reessaie_puis_echoue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("puls_events_rag.core.providers.open_agenda.asyncio.sleep", _no_sleep)
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(503), requests)

    with pytest.raises(OpenAgendaError):
        _pages(client)
    assert len(requests) == 4


async def _no_sleep(_: float) -> None:
    return None
