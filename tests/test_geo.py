"""Tests des outils géographiques (aucun appel réseau)."""

from __future__ import annotations

import pytest

from puls_events_rag.core.geo import (
    GeoPoint,
    bounding_box,
    distance_to_nearest_km,
    event_point,
    haversine_km,
)

MARSEILLE = GeoPoint(43.2803, 5.3806)
AIX = GeoPoint(43.5283, 5.4497)


def test_haversine_marseille_aix() -> None:
    assert haversine_km(MARSEILLE, AIX) == pytest.approx(28.2, abs=0.5)
    assert haversine_km(AIX, AIX) == 0.0


def test_bounding_box_contient_le_cercle() -> None:
    box = bounding_box([MARSEILLE], 10)

    assert box.south < MARSEILLE.lat < box.north
    assert box.west < MARSEILLE.lng < box.east
    assert haversine_km(MARSEILLE, GeoPoint(box.north, MARSEILLE.lng)) == pytest.approx(10, 0.01)
    assert haversine_km(MARSEILLE, GeoPoint(MARSEILLE.lat, box.east)) == pytest.approx(10, 0.01)
    assert set(box.to_params()) == {
        "geo[northEast][lat]",
        "geo[northEast][lng]",
        "geo[southWest][lat]",
        "geo[southWest][lng]",
    }


def test_bounding_box_englobe_plusieurs_centres() -> None:
    box = bounding_box([MARSEILLE, AIX], 1)

    assert box.south < MARSEILLE.lat and box.north > AIX.lat


def test_bounding_box_sans_centre() -> None:
    with pytest.raises(ValueError):
        bounding_box([], 5)


def test_coordonnees_et_distance_d_un_evenement() -> None:
    event = {"location": {"latitude": 43.5283, "longitude": 5.4497}}

    assert event_point(event) == AIX
    assert distance_to_nearest_km(event, [MARSEILLE, AIX]) == 0.0
    assert event_point({"location": {"latitude": None}}) is None
    assert event_point({}) is None
    assert distance_to_nearest_km({}, [AIX]) is None
