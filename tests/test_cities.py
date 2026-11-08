"""Tests de la détection des communes dans une question (aucun appel réseau)."""

from __future__ import annotations

import pytest

from puls_events_rag.core.cities import load_communes, match_cities, normalize, strip_cities


def test_referentiel_complet_et_embarque() -> None:
    communes = load_communes()

    assert len(communes) == 92
    assert communes["Aix-en-Provence"].aliases == ("Aix",)
    assert communes["Marseille"].insee == "13055"
    # Tous les centres sont dans la métropole (Bouches-du-Rhône, Vaucluse, Var).
    assert all(43.1 < c.center.lat < 43.8 and 4.6 < c.center.lng < 5.8 for c in communes.values())


def test_normalize_casse_accents_et_separateurs() -> None:
    assert normalize("  Berre-l\u2019Étang ") == "berre l etang"
    assert normalize("SAINT-PAUL-LÈS-DURANCE") == "saint paul les durance"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Un concert à Marseille ce week-end ?", ["Marseille"]),
        ("expo à aix en provence", ["Aix-en-Provence"]),
        ("Un entretien pro au Jas de Bouffan, à Aix, en mai", ["Aix-en-Provence"]),
        ("Festival a berre l'etang", ["Berre-l'Étang"]),
        ("Entre Aubagne et La Ciotat ou à Aubagne", ["Aubagne", "La Ciotat"]),
        (
            "Les Pennes-Mirabeau puis Port-Saint-Louis",
            ["Les Pennes-Mirabeau", "Port-Saint-Louis-du-Rhône"],
        ),
    ],
)
def test_match_cities(question: str, expected: list[str]) -> None:
    assert match_cities(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Un entretien pro proche du parc Jourdan",
        "Le salon du livre",  # « Salon » seul n'est pas un alias de Salon-de-Provence
        "Marseillaise et istresiens",  # pas de correspondance dans un mot plus long
    ],
)
def test_sans_commune(question: str) -> None:
    assert match_cities(question) == []


def test_forme_longue_prioritaire_sur_l_alias() -> None:
    assert match_cities("Aix-en-Provence") == ["Aix-en-Provence"]


def test_strip_cities_ne_retire_que_les_communes_filtrees() -> None:
    keywords = ["Jas de Bouffan", "Aix", "aix-en-provence", "Marseille", "mairie d'Aix"]

    assert strip_cities(keywords, ["Aix-en-Provence"]) == [
        "Jas de Bouffan",
        "Marseille",
        "mairie d'Aix",
    ]
