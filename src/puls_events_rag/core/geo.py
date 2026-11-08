"""Outils géographiques : distance entre deux points et zone de recherche Open Agenda."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE_LAT = 111.32


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lng: float


@dataclass(frozen=True)
class BoundingBox:
    """Rectangle ``geo[northEast]`` / ``geo[southWest]`` de l'API Open Agenda."""

    south: float
    west: float
    north: float
    east: float

    def to_params(self) -> dict[str, str]:
        return {
            "geo[northEast][lat]": f"{self.north:.6f}",
            "geo[northEast][lng]": f"{self.east:.6f}",
            "geo[southWest][lat]": f"{self.south:.6f}",
            "geo[southWest][lng]": f"{self.west:.6f}",
        }


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Distance orthodromique en kilomètres."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    d_lat, d_lng = lat2 - lat1, math.radians(b.lng - a.lng)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def bounding_box(centers: Iterable[GeoPoint], radius_km: float) -> BoundingBox:
    """Plus petit rectangle contenant un cercle de ``radius_km`` autour de chaque centre."""
    points = list(centers)
    if not points:
        raise ValueError("Au moins un centre est requis")
    boxes = []
    for p in points:
        d_lat = radius_km / KM_PER_DEGREE_LAT
        d_lng = radius_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(p.lat)))
        boxes.append((p.lat - d_lat, p.lng - d_lng, p.lat + d_lat, p.lng + d_lng))
    return BoundingBox(
        south=min(b[0] for b in boxes),
        west=min(b[1] for b in boxes),
        north=max(b[2] for b in boxes),
        east=max(b[3] for b in boxes),
    )


def event_point(event: Mapping[str, Any]) -> GeoPoint | None:
    """Coordonnées ``location.latitude`` / ``location.longitude`` d'un événement."""
    location = event.get("location")
    if not isinstance(location, Mapping):
        return None
    lat, lng = location.get("latitude"), location.get("longitude")
    if isinstance(lat, int | float) and isinstance(lng, int | float):
        return GeoPoint(float(lat), float(lng))
    return None


def distance_to_nearest_km(event: Mapping[str, Any], centers: Iterable[GeoPoint]) -> float | None:
    """Distance de l'événement au centre le plus proche (``None`` sans coordonnées)."""
    point = event_point(event)
    if point is None:
        return None
    distances = [haversine_km(point, center) for center in centers]
    return min(distances) if distances else None
