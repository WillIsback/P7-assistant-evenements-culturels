"""Scénarios d'interaction et de robustesse (#28).

- Unitaire : le fichier de scénarios est valide et couvre les cas exigés.
- Intégration (``-m integration``) : chaque scénario contre la vraie chaîne RAG
  (index régional ``data/index``, LLM du profil courant).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

SCENARIOS_PATH = Path(__file__).parent / "rag" / "scenarios.yaml"
SCENARIOS: list[dict[str, Any]] = yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))
REQUIRED = {"theme", "lieu-et-periode", "hors-sujet-meteo", "sans-resultat", "injection", "anglais"}
EXPECT_KEYS = {"sources", "communes", "period", "answer_any", "answer_none"}


def test_scenarios_valides() -> None:
    ids = [scenario["id"] for scenario in SCENARIOS]
    assert 10 <= len(SCENARIOS) <= 15
    assert len(ids) == len(set(ids))
    assert set(ids) >= REQUIRED
    for scenario in SCENARIOS:
        assert scenario["question"].strip()
        assert set(scenario["expect"]) <= EXPECT_KEYS
        assert scenario["expect"].get("sources", "some") in {"some", "none"}


@pytest.fixture(scope="module")
def answers() -> dict[str, Any]:
    """Toutes les réponses dans un seul event loop (client HTTP LangChain partagé)."""
    from dotenv import load_dotenv

    from puls_events_rag.core.cache import setup_cache
    from puls_events_rag.core.index import IndexLoadError, RegionalIndex
    from puls_events_rag.core.providers.openai_api import build_client, cached_embeddings
    from puls_events_rag.core.sync import IndexSettings
    from puls_events_rag.rag.chain import RagChain
    from puls_events_rag.rag.query import LLMQueryParser

    load_dotenv()
    setup_cache()

    async def run() -> dict[str, Any]:
        try:
            embeddings = cached_embeddings()
            index = RegionalIndex.load(
                IndexSettings.from_env().directory, embeddings, model=embeddings.model
            )
        except (RuntimeError, IndexLoadError) as exc:
            pytest.skip(f"index ou clés absents : {exc}")
        llm = build_client()
        chain = RagChain(index=index, llm=llm, parser=LLMQueryParser(llm))
        return {s["id"]: await chain.ask(s["question"]) for s in SCENARIOS}

    return asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(scenario: dict[str, Any], answers: dict[str, Any]) -> None:
    result = answers[scenario["id"]]
    expect = scenario["expect"]
    answer = result.answer.casefold()

    assert result.answer.strip()
    if expect.get("sources") == "some":
        assert result.sources, "au moins une source attendue"
    if expect.get("sources") == "none":
        assert not result.sources, [s.title for s in result.sources]
    for commune in expect.get("communes", []):
        assert commune in result.query.to_filter().communes
    if expect.get("period"):
        assert result.query.since is not None or result.query.until is not None
    if expect.get("answer_any"):
        assert any(text in answer for text in expect["answer_any"]), result.answer
    for text in expect.get("answer_none", []):
        assert text not in answer, result.answer
