"""Tests de l'index régional : persistance, fraîcheur, retrieval filtré (sans réseau)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from puls_events_rag.core.cities import load_communes
from puls_events_rag.core.index import EventFilter, IndexLoadError, RegionalIndex

VOCABULARY = ("atelier", "concert", "immersion", "exposition", "danse")
MODEL = "fake-embed"


class KeywordEmbeddings(Embeddings):
    """Vecteur = occurrences du vocabulaire (+ biais) : similarité prévisible."""

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in VOCABULARY] + [0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


EMBEDDINGS = KeywordEmbeddings()
AIX = load_communes()["Aix-en-Provence"].center


def _event(
    uid: int,
    title: str,
    *,
    commune: str | None = "Marseille",
    lat: float | None = 43.2965,
    lng: float | None = 5.3698,
    begin: str = "2026-11-10T09:00:00+01:00",
    end: str = "2026-11-10T12:00:00+01:00",
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    location: dict[str, Any] = {"name": "Lieu", "city": commune or "Ailleurs"}
    if commune:
        location["adminLevel4"] = commune
    if lat is not None:
        location["latitude"], location["longitude"] = lat, lng
    return {
        "uid": uid,
        "title": {"fr": title},
        "firstTiming": {"begin": begin, "end": end},
        "lastTiming": {"begin": begin, "end": end},
        "keywords": {"fr": keywords or []},
        "location": location,
        "updatedAt": "2026-11-01",
    }


def _build(events: list[dict[str, Any]], now: datetime | None = None) -> RegionalIndex:
    return asyncio.run(RegionalIndex.build(events, EMBEDDINGS, model=MODEL, now=now))


def test_sauvegarde_puis_chargement(tmp_path: Path) -> None:
    index = _build([_event(1, "Atelier"), _event(2, "Concert")])

    index.save(tmp_path / "index")
    loaded = RegionalIndex.load(tmp_path / "index", EMBEDDINGS, model=MODEL)

    assert loaded.manifest == index.manifest
    assert loaded.manifest.event_count == 2
    result = asyncio.run(loaded.search("concert", k=1))
    assert [event.uid for event in result.events] == [2]


def test_sauvegarde_remplace_l_index_existant(tmp_path: Path) -> None:
    _build([_event(1, "Atelier")]).save(tmp_path / "index")
    _build([_event(1, "Atelier"), _event(2, "Danse")]).save(tmp_path / "index")

    loaded = RegionalIndex.load(tmp_path / "index", EMBEDDINGS, model=MODEL)

    assert loaded.manifest.event_count == 2
    assert not (tmp_path / "index.tmp").exists()


def test_chargement_refuse_autre_modele(tmp_path: Path) -> None:
    _build([_event(1, "Atelier")]).save(tmp_path / "index")

    with pytest.raises(IndexLoadError, match=r"autre-modele"):
        RegionalIndex.load(tmp_path / "index", EMBEDDINGS, model="autre-modele")


def test_chargement_index_absent(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError):
        RegionalIndex.load(tmp_path / "absent", EMBEDDINGS, model=MODEL)


def test_aucun_evenement_indexable() -> None:
    with pytest.raises(ValueError):
        _build([{"uid": 1, "title": {}}])


def test_fraicheur_24_heures() -> None:
    synced = datetime(2026, 11, 8, 12, tzinfo=UTC)
    manifest = _build([_event(1, "Atelier")], now=synced).manifest
    ttl = timedelta(hours=24)

    assert manifest.is_fresh(ttl, now=synced + timedelta(hours=23))
    assert not manifest.is_fresh(ttl, now=synced + timedelta(hours=25))


def test_filtre_commune_exhaustif() -> None:
    # 30 événements marseillais très proches de la requête, un seul aixois moins proche :
    # le filtre doit le retrouver même s'il est loin dans le classement global.
    events = [_event(i, "Atelier atelier immersion") for i in range(1, 31)]
    events.append(_event(99, "Atelier", commune="Aix-en-Provence", lat=AIX.lat, lng=AIX.lng))
    index = _build(events)

    result = asyncio.run(
        index.search("atelier immersion", k=3, where=EventFilter(communes=("Aix-en-Provence",)))
    )

    assert [event.uid for event in result.events] == [99]
    assert result.radius_km is None


def test_filtre_periode_par_chevauchement() -> None:
    events = [
        _event(1, "Exposition", begin="2026-10-01T10:00:00+02:00", end="2026-12-31T18:00:00+01:00"),
        _event(2, "Exposition", begin="2026-12-15T10:00:00+01:00", end="2026-12-15T18:00:00+01:00"),
        _event(3, "Exposition", begin="2026-09-01T10:00:00+02:00", end="2026-09-02T18:00:00+02:00"),
    ]
    index = _build(events)
    where = EventFilter(
        since=datetime(2026, 11, 8, tzinfo=UTC), until=datetime(2026, 11, 23, tzinfo=UTC)
    )

    result = asyncio.run(index.search("exposition", k=5, where=where))

    assert [event.uid for event in result.events] == [1]


def test_search_elargit_par_rayon() -> None:
    # Rien à Aix ; un atelier à ~6 km (Venelles), un autre à Marseille (~25 km).
    events = [
        _event(1, "Atelier", commune="Venelles", lat=43.58, lng=5.44),
        _event(2, "Atelier", commune="Marseille", lat=43.2965, lng=5.3698),
    ]
    index = _build(events)

    result = asyncio.run(
        index.search("atelier", k=5, where=EventFilter(communes=("Aix-en-Provence",)))
    )

    assert [event.uid for event in result.events] == [1]
    assert result.radius_km == 10.0
    assert result.events[0].distance_km == pytest.approx(6.4, abs=0.5)


def test_sans_elargissement() -> None:
    index = _build([_event(1, "Atelier", commune="Venelles", lat=43.58, lng=5.44)])

    result = asyncio.run(
        index.search("atelier", where=EventFilter(communes=("Aix-en-Provence",)), expand=False)
    )

    assert result.events == []


def test_evenement_sans_coordonnees_ignore_par_filtre_geo() -> None:
    events = [
        _event(1, "Atelier", commune=None, lat=None),
        _event(2, "Atelier", commune="Venelles", lat=43.58, lng=5.44),
    ]
    index = _build(events)

    by_city = asyncio.run(
        index.search("atelier", k=5, where=EventFilter(communes=("Aix-en-Provence",)))
    )
    by_period = asyncio.run(index.search("atelier", k=5))

    assert [event.uid for event in by_city.events] == [2]
    assert {event.uid for event in by_period.events} == {1, 2}


def test_requete_vide() -> None:
    index = _build([_event(1, "Atelier")])

    assert asyncio.run(index.search("  ")).events == []


def test_facettes() -> None:
    events = [
        _event(1, "Atelier", commune="Aix-en-Provence", keywords=["Emploi", "Atelier"]),
        _event(2, "Atelier", commune="Aix-en-Provence", keywords=["Emploi"]),
        _event(3, "Concert", keywords=["Musique"]),
    ]
    index = _build(events)

    facets = index.facets(EventFilter(communes=("Aix-en-Provence",)))

    assert facets.event_count == 2
    assert facets.top_keywords[0] == ("Emploi", 2)


def test_seuil_de_pertinence_declenche_l_elargissement() -> None:
    # À Aix, seulement un concert (hors sujet) ; un atelier pertinent à ~6 km.
    events = [
        _event(1, "Concert", commune="Aix-en-Provence", lat=AIX.lat, lng=AIX.lng),
        _event(2, "Atelier", commune="Venelles", lat=43.58, lng=5.44),
    ]
    index = _build(events)
    where = EventFilter(communes=("Aix-en-Provence",))

    without = asyncio.run(index.search("atelier", k=5, where=where))
    strict = asyncio.run(index.search("atelier", k=5, where=where, max_distance=0.5))

    assert [event.uid for event in without.events] == [1]
    assert [event.uid for event in strict.events] == [2]
    assert strict.radius_km == 10.0
