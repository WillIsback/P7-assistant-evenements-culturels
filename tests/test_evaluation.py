"""Tests du jeu annoté (#13) et des métriques d'évaluation (#14), sans réseau."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from puls_events_rag.evaluation.dataset import EvalItem, load_dataset
from puls_events_rag.evaluation.metrics import (
    ItemResult,
    Judge,
    Judgement,
    first_hit_rank,
    hit_rate,
    mrr,
    score,
)
from puls_events_rag.rag.chain import Source

CATEGORIES = {"theme", "lieu", "date", "sans_reponse", "hors_sujet", "hors_zone"}


def test_jeu_annote_valide_et_couvrant() -> None:
    items = load_dataset()

    assert 30 <= len(items) <= 50
    assert {item.category for item in items} == CATEGORIES
    assert sum(not item.answerable for item in items) >= 8  # dont des questions sans réponse


def _item(category: str = "theme", uids: list[str] | None = None, **extra: Any) -> EvalItem:
    uids = ["1", "2"] if uids is None else uids
    return EvalItem(
        id="q",
        category=category,  # type: ignore[arg-type]
        question="Une question ?",
        reference="Référence.",
        expected_uids=uids,
        expected_total=len(uids),
        **extra,
    )


def test_schema_refuse_les_incoherences() -> None:
    with pytest.raises(ValidationError):
        _item("theme", uids=[])
    with pytest.raises(ValidationError):
        _item("sans_reponse", uids=["1"])
    with pytest.raises(ValidationError):
        EvalItem(
            id="Q majuscule",
            category="theme",
            question="Une question ?",
            reference="Référence.",
            expected_uids=["1"],
            expected_total=1,
        )


def test_doublons_refuses(tmp_path: Path) -> None:
    row = _item().model_dump()
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(row) for _ in range(2)), encoding="utf-8")

    with pytest.raises(ValueError, match="double"):
        load_dataset(path)


def test_rang_hit_rate_et_mrr() -> None:
    assert first_hit_rank(["b"], ["a", "b", "c"]) == 2
    assert first_hit_rank(["z"], ["a", "b"]) is None
    ranks = [1, 2, None, 4]
    assert hit_rate(ranks) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert hit_rate([]) == 0.0 and mrr([]) == 0.0


def _result(item: EvalItem, *, rank: int | None, faithful: bool, correct: bool) -> ItemResult:
    sources = [Source(uid="1", title="É", score=0.3)] if rank else []
    verdict = Judgement(faithful=faithful, correct=correct, rationale="ok")
    return ItemResult(item, "réponse", sources, 1000, verdict, rank=rank)


def test_agregation_des_scores() -> None:
    results = [
        _result(_item(), rank=1, faithful=True, correct=True),
        _result(_item(), rank=None, faithful=True, correct=False),
        _result(_item("sans_reponse", uids=[]), rank=None, faithful=True, correct=True),
        _result(_item("hors_sujet", uids=[]), rank=None, faithful=False, correct=False),
    ]

    scores = score(results)

    assert scores.hit_rate == 0.5
    assert scores.mrr == 0.5
    assert scores.refusal_rate == 0.5
    assert scores.faithfulness == 0.75
    assert scores.correctness == 0.5
    assert scores.counts == {"questions": 4, "avec_reponse": 2, "a_refuser": 2, "erreurs": 0}


class _StructuredFake(FakeListChatModel):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        verdict = Judgement(faithful=True, correct=False, rationale="refus")

        async def _answer(_: Any) -> Judgement:
            return verdict

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_answer)


def test_juge_transmet_le_contexte_et_la_categorie() -> None:
    judge = Judge(_StructuredFake(responses=[]))

    verdict = asyncio.run(judge(_item(), "Aucun événement.", ""))

    assert verdict == Judgement(faithful=True, correct=False, rationale="refus")
