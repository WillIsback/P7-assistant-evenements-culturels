"""Détection des communes citées dans une question, pour filtrer Open Agenda.

Le référentiel ``data/communes.json`` associe chaque commune de la métropole
(valeur exacte de ``location.adminLevel4``) à ses formes courtes usuelles, à son
code INSEE et à son centre géographique (``geo.api.gouv.fr``).
Question et référentiel sont normalisés de la même façon (casse, accents,
tirets, apostrophes) puis croisés par une seule expression régulière : « aix »,
« Aix en Provence » et « AIX-EN-PROVENCE » renvoient tous ``Aix-en-Provence``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

from puls_events_rag.core.geo import GeoPoint

_SEPARATORS = re.compile(r"[\s\-'\u2019]+")


def normalize(text: str) -> str:
    """Minuscules, sans accents, tirets et apostrophes remplacés par une espace."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SEPARATORS.sub(" ", without_accents).strip()


@dataclass(frozen=True)
class Commune:
    name: str
    aliases: tuple[str, ...]
    insee: str
    center: GeoPoint


@cache
def load_communes() -> dict[str, Commune]:
    """Charge le référentiel embarqué, indexé par nom officiel."""
    raw = files("puls_events_rag.data").joinpath("communes.json").read_text(encoding="utf-8")
    return {
        name: Commune(
            name=name,
            aliases=tuple(entry["aliases"]),
            insee=entry["insee"],
            center=GeoPoint(entry["lat"], entry["lng"]),
        )
        for name, entry in json.loads(raw)["communes"].items()
    }


def city_centers(cities: list[str]) -> list[GeoPoint]:
    """Centres géographiques des communes connues parmi ``cities``."""
    communes = load_communes()
    return [communes[city].center for city in cities if city in communes]


@cache
def _index() -> tuple[re.Pattern[str], dict[str, str]]:
    """Construit l'expression régulière et la table forme normalisée -> commune."""
    lookup: dict[str, str] = {}
    for name, commune in load_communes().items():
        for form in (name, *commune.aliases):
            lookup.setdefault(normalize(form), name)
    # Formes les plus longues d'abord : « aix en provence » avant « aix ».
    alternatives = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, alternatives)) + r")(?!\w)")
    return pattern, lookup


def match_cities(text: str) -> list[str]:
    """Renvoie les communes citées dans ``text``, dans l'ordre d'apparition, sans doublon."""
    pattern, lookup = _index()
    found = (lookup[match] for match in pattern.findall(normalize(text)))
    return list(dict.fromkeys(found))


def strip_cities(keywords: list[str], cities: list[str]) -> list[str]:
    """Retire les mots-clés qui ne font que nommer une commune déjà filtrée.

    « Aix » est inutile dans ``search`` quand ``city[]=Aix-en-Provence`` est
    posé ; « Jas de Bouffan » ou « mairie d'Aix » sont conservés.
    """
    _, lookup = _index()
    filtered = set(cities)
    return [keyword for keyword in keywords if lookup.get(normalize(keyword)) not in filtered]


def resolve_commune(name: str) -> str | None:
    """Nom officiel de la commune désignée par ``name`` (nom exact ou alias), sinon ``None``.

    « aix », « Aix en Provence » et « AIX-EN-PROVENCE » renvoient ``Aix-en-Provence`` ;
    une commune hors de la métropole renvoie ``None``.
    """
    _, lookup = _index()
    return lookup.get(normalize(name))
