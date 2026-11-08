"""Tests de la conversion des événements en documents (#6, sans réseau)."""

from __future__ import annotations

from typing import Any

from puls_events_rag.core.cities import resolve_commune
from puls_events_rag.core.documents import CHUNK_SIZE, event_to_documents, events_to_documents


def _event(uid: int = 43496199, **overrides: Any) -> dict[str, Any]:
    """Événement au format réel de l'API Open Agenda v2 (``detailed=1``)."""
    event: dict[str, Any] = {
        "uid": uid,
        "title": {"fr": "Kouss•Ftartchi", "en": "Kouss•Ftartchi"},
        "description": {"fr": "Cuisine et concert au menu !", "en": "Cuisine and concert!"},
        "longDescription": {
            "fr": "Cuisine et concert au menu !  \nLe projet tunisien **Ftartchi** est invité."
        },
        "keywords": {"fr": ["Concert", "Gratuit"]},
        "dateRange": {"fr": "Vendredi 28 août, 14h30"},
        "firstTiming": {"begin": "2026-08-28T14:30:00+02:00", "end": "2026-08-28T16:30:00+02:00"},
        "lastTiming": {"begin": "2026-08-28T14:30:00+02:00", "end": "2026-08-28T16:30:00+02:00"},
        "location": {
            "name": "La Cômerie",
            "address": "176 rue Breteuil, 13006 Marseille",
            "city": "Marseille",
        },
        "age": {"min": 6, "max": None},
        "conditions": {},
        "canonicalUrl": "https://openagenda.com/saison-mediterranee-2026/events/43496199",
    }
    event.update(overrides)
    return event


def test_event_to_documents_construit_texte_et_metadonnees() -> None:
    [doc] = event_to_documents(_event())

    assert doc.page_content.startswith("Titre : Kouss•Ftartchi\nDates : Vendredi 28 août")
    assert "Lieu : La Cômerie, 176 rue Breteuil, 13006 Marseille" in doc.page_content
    assert "Mots-clés : Concert, Gratuit" in doc.page_content
    assert "Public : à partir de 6 ans" in doc.page_content
    assert "Conditions" not in doc.page_content  # champ vide omis
    assert "**" not in doc.page_content  # Markdown retiré
    assert doc.page_content.count("Cuisine et concert au menu !") == 1  # résumé non dupliqué
    assert doc.metadata["uid"] == 43496199
    assert doc.metadata["city"] == "Marseille"
    assert doc.metadata["begin"] == "2026-08-28T14:30:00+02:00"
    assert doc.metadata["url"].endswith("/43496199")


def test_repli_sur_une_autre_langue_et_evenement_minimal() -> None:
    [doc] = event_to_documents({"uid": 1, "title": {"en": "Jazz night"}})

    assert doc.page_content == "Titre : Jazz night"
    assert doc.metadata["location"] == ""


def test_ignore_les_evenements_sans_titre_ou_uid() -> None:
    assert event_to_documents({"uid": 1, "title": {}}) == []
    assert event_to_documents({"title": {"fr": "Sans uid"}}) == []


def test_longue_description_decoupee_avec_en_tete_repete() -> None:
    long_text = " ".join(f"phrase{i}" for i in range(CHUNK_SIZE))
    docs = event_to_documents(_event(longDescription={"fr": long_text}, description={}))

    assert len(docs) > 1
    assert all(doc.page_content.startswith("Titre : Kouss•Ftartchi") for doc in docs)
    assert [doc.metadata["chunk"] for doc in docs] == list(range(len(docs)))


def test_events_to_documents_deduplique_par_uid() -> None:
    docs = events_to_documents([_event(1), _event(1), _event(2), {"uid": 3}])

    assert [doc.metadata["uid"] for doc in docs] == [1, 2]


def test_metadonnees_de_filtrage() -> None:
    location = {
        "name": "Cité du Livre",
        "city": "Aix-en-Provence",
        "adminLevel4": "Aix-en-Provence",
        "latitude": 43.5236,
        "longitude": 5.4398,
    }
    [doc] = event_to_documents(_event(location=location))

    assert doc.metadata["commune"] == "Aix-en-Provence"
    assert doc.metadata["lat"] == 43.5236
    assert doc.metadata["lng"] == 5.4398
    assert doc.metadata["begin"] == "2026-08-28T14:30:00+02:00"
    assert doc.metadata["end"] == "2026-08-28T16:30:00+02:00"
    assert "distance_km" not in doc.metadata


def test_commune_hors_referentiel_et_sans_coordonnees() -> None:
    [doc] = event_to_documents(_event(location={"name": "Salle", "city": "Lyon"}))

    assert doc.metadata["commune"] is None
    assert doc.metadata["lat"] is None
    assert doc.metadata["lng"] is None


def test_html_retire_de_la_description() -> None:
    [doc] = event_to_documents(_event(longDescription={"fr": "<p>Atelier <b>gratuit</b></p>"}))

    assert "<" not in doc.page_content
    assert "Atelier gratuit" in doc.page_content


def test_resolve_commune_par_alias() -> None:
    assert resolve_commune("aix") == "Aix-en-Provence"
    assert resolve_commune("AIX-EN-PROVENCE") == "Aix-en-Provence"
    assert resolve_commune("Lyon") is None
