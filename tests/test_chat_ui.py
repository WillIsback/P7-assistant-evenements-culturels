"""Tests des fonctions pures de l'UI Gradio de dev (sans serveur)."""

from __future__ import annotations

import pytest

# Gradio vit dans l'environnement du script (PEP 723), pas dans celui du projet.
pytest.importorskip("gradio", reason="Gradio est hors du lock (script PEP 723)")

from scripts.chat_ui import _session_path, format_sources, parse_sse


def test_parse_sse_ignore_les_lignes_hors_data() -> None:
    lines = iter(
        ["event: token", 'data: {"type": "token", "content": "Bon"}', "", 'data: {"type": "done"}']
    )

    assert [event["type"] for event in parse_sse(lines)] == ["token", "done"]


def test_format_sources_avec_lien_et_distance() -> None:
    text = format_sources(
        [
            {
                "title": "Atelier",
                "url": "https://openagenda.com/e/1",
                "date_range": "Jeudi 19 novembre",
                "location": "Marseille",
                "distance_km": 28.1,
            }
        ]
    )

    assert "[Atelier](https://openagenda.com/e/1)" in text
    assert "à 28.1 km" in text
    assert format_sources([]) == ""


def test_identifiant_de_session_valide_comme_uuid() -> None:
    sid = "9c2651b8-2d3d-42a5-bb71-6832edd8d4dd"

    assert _session_path(sid) == f"/chat/sessions/{sid}"
    with pytest.raises(ValueError):
        _session_path("../../users/1")
