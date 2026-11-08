"""Tests des fonctions pures de l'UI Gradio de dev (sans serveur)."""

from __future__ import annotations

from scripts.chat_ui import format_sources, parse_sse


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
