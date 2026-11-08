"""Tests unitaires du client Open Agenda et de la recherche ciblée."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from puls_events_rag.core.cities import load_communes
from puls_events_rag.core.event_search import find_events
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


class _FakeExtractor:
    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    async def get_keywords(self, query: str) -> list[str]:
        return self.keywords


def test_find_events_utilise_le_parametre_search() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 0, "events": []}),
        requests,
    )

    events = asyncio.run(
        find_events(
            "Quel evenement a eu lieu au Jas de Bouffan",
            llm=_FakeExtractor(["Jas de Bouffan", "Aix-en-Provence"]),
            client=client,
        )
    )

    assert events == []
    assert requests[0].url.params["search"] == "Jas de Bouffan Aix-en-Provence"


def test_find_events_sans_mots_cles_n_appelle_pas_l_api() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 0, "events": []}),
        requests,
    )

    events = asyncio.run(find_events("question vide", llm=_FakeExtractor([]), client=client))

    assert events == []
    assert requests == []


def test_find_events_filtre_par_commune_citee() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 0, "events": []}),
        requests,
    )

    asyncio.run(
        find_events(
            "Un entretien pro au Jas de Bouffan à Aix ou à Aubagne",
            llm=_FakeExtractor(["entretien", "Jas de Bouffan", "Aix", "Aubagne"]),
            client=client,
        )
    )

    params = requests[0].url.params
    assert params.get_list("city[]") == ["Aix-en-Provence", "Aubagne"]
    assert params["search"] == "entretien Jas de Bouffan"


def test_find_events_sans_commune_ne_filtre_pas() -> None:
    requests: list[httpx.Request] = []
    client = _client(
        lambda request: httpx.Response(200, json={"total": 0, "events": []}),
        requests,
    )

    asyncio.run(
        find_events("Près du parc Jourdan", llm=_FakeExtractor(["parc Jourdan"]), client=client)
    )

    assert "city[]" not in requests[0].url.params


def _event_at(uid: int, lat: float, lng: float) -> dict[str, object]:
    return {"uid": uid, "location": {"latitude": lat, "longitude": lng}}


def test_find_events_elargit_a_la_zone_si_la_commune_est_vide() -> None:
    requests: list[httpx.Request] = []
    # Gignac-la-Nerthe : centre 43.387 N / 5.233 E.
    marseille, marignane = _event_at(1, 43.2964, 5.3700), _event_at(2, 43.4159, 5.2145)

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "city[]" in params:
            return httpx.Response(200, json={"total": 0, "events": []})
        lat_span = float(params["geo[northEast][lat]"]) - float(params["geo[southWest][lat]"])
        events = [marseille, marignane] if lat_span > 0.15 else []  # vide à 5 km
        return httpx.Response(200, json={"total": len(events), "events": events})

    events = asyncio.run(
        find_events(
            "Un concert à Gignac-la-Nerthe",
            llm=_FakeExtractor(["concert", "Gignac-la-Nerthe"]),
            client=_client(handler, requests),
        )
    )

    assert requests[0].url.params.get_list("city[]") == ["Gignac-la-Nerthe"]
    assert len(requests) == 3  # commune, 5 km (vide), 10 km
    assert all("city[]" not in r.url.params for r in requests[1:])
    assert requests[2].url.params["search"] == "concert"
    assert [event["uid"] for event in events] == [2, 1]  # Marignane avant Marseille
    assert events[0]["distance_km"] == pytest.approx(3.5, abs=0.3)
    assert events[1]["distance_km"] == pytest.approx(15.3, abs=0.5)


def test_find_events_n_elargit_pas_si_la_commune_a_des_resultats() -> None:
    requests: list[httpx.Request] = []
    aix = load_communes()["Aix-en-Provence"].center
    client = _client(
        lambda request: httpx.Response(200, json={"events": [_event_at(1, aix.lat, aix.lng)]}),
        requests,
    )

    events = asyncio.run(find_events("expo à Aix", llm=_FakeExtractor(["expo"]), client=client))

    assert len(requests) == 1
    assert events[0]["distance_km"] == 0.0


def test_find_events_rien_dans_la_zone() -> None:
    requests: list[httpx.Request] = []
    client = _client(lambda request: httpx.Response(200, json={"events": []}), requests)

    events = asyncio.run(find_events("à Trets", llm=_FakeExtractor(["Trets"]), client=client))

    assert events == []
    assert len(requests) == 5  # commune + 4 rayons
