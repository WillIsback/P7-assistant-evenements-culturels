"""Métriques d'évaluation : récupération (hit rate, MRR), refus, jugement LLM (#14)."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from puls_events_rag.evaluation.dataset import EvalItem
from puls_events_rag.rag.chain import Source


def first_hit_rank(expected: Sequence[str], retrieved: Sequence[str]) -> int | None:
    """Rang (1-indexé) du premier événement attendu parmi les sources, sinon ``None``."""
    wanted = set(expected)
    for rank, uid in enumerate(retrieved, start=1):
        if uid in wanted:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None]) -> float:
    return sum(rank is not None for rank in ranks) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None]) -> float:
    return sum(1 / rank for rank in ranks if rank) / len(ranks) if ranks else 0.0


class Judgement(BaseModel):
    """Verdict du LLM-juge sur une réponse."""

    faithful: bool = Field(
        description="vrai si chaque événement cité figure dans les sources fournies"
    )
    correct: bool = Field(description="vrai si la réponse est cohérente avec la référence")
    rationale: str = Field(description="justification en une phrase")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu évalues un assistant qui recommande des événements à partir de sources.\n"
            "- faithful : vrai si tout ce que la réponse affirme sur des événements (titres, "
            "dates, lieux, contenu) figure dans le contexte fourni à l'assistant ; un refus "
            "ou une demande de précision sans événement est fidèle.\n"
            "- correct : la catégorie attendue est « {expectation} ».\n"
            "  * « proposer » : vrai seulement si la réponse propose au moins un événement "
            "correspondant à la demande (idéalement un de la référence) ; un refus, « aucun "
            "événement » ou une demande de précision est FAUX.\n"
            "  * « refuser » : vrai si la réponse dit qu'aucun événement ne correspond, "
            "refuse poliment ou demande de préciser, sans inventer d'événement.\n"
            "  * « signaler » : vrai si la réponse signale que le lieu demandé est hors de la "
            "zone couverte.\n"
            "La formulation et l'exhaustivité ne comptent pas. Réponds par le JSON demandé.",
        ),
        (
            "human",
            "Question : {question}\n\nRéférence : {reference}\n\nContexte fourni à "
            "l'assistant :\n{context}\n\nRéponse de l'assistant :\n{answer}",
        ),
    ]
)


EXPECTATIONS = {
    "theme": "proposer",
    "lieu": "proposer",
    "date": "proposer",
    "sans_reponse": "refuser",
    "hors_sujet": "refuser",
    "hors_zone": "signaler",
}


class Judge:
    """LLM-juge par sortie structurée."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = JUDGE_PROMPT | llm.with_structured_output(Judgement)

    async def __call__(self, item: EvalItem, answer: str, context: str) -> Judgement:
        verdict = await self._chain.ainvoke(
            {
                "question": item.question,
                "reference": item.reference,
                "expectation": EXPECTATIONS[item.category],
                "context": context or "(aucun : l'assistant n'a reçu aucun événement)",
                "answer": answer,
            }
        )
        if not isinstance(verdict, Judgement):  # pragma: no cover - contrat LangChain
            raise TypeError("verdict du juge invalide")
        return verdict


@dataclass
class ItemResult:
    item: EvalItem
    answer: str
    sources: list[Source]
    latency_ms: int
    judgement: Judgement | None
    rank: int | None = None
    error: str | None = None

    @property
    def refused(self) -> bool:
        """Refus constaté : verdict du juge (catégorie « refuser »), sinon absence de source.

        Une réponse peut recevoir des sources et refuser quand même (« pas de météo ici »).
        """
        if self.judgement is not None:
            return self.judgement.correct
        return not self.sources


@dataclass
class Scores:
    hit_rate: float
    mrr: float
    refusal_rate: float
    faithfulness: float
    correctness: float
    latency_p50_ms: float
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "refusal_rate": self.refusal_rate,
            "faithfulness": self.faithfulness,
            "correctness": self.correctness,
        }


def score(results: Sequence[ItemResult]) -> Scores:
    answerable = [r for r in results if r.item.answerable]
    # Hors zone : l'assistant peut proposer des événements de la métropole en signalant
    # le lieu non couvert ; seuls « sans réponse » et « hors sujet » exigent zéro source.
    must_refuse = [r for r in results if r.item.category in {"sans_reponse", "hors_sujet"}]
    judged = [r.judgement for r in results if r.judgement is not None]
    ranks = [r.rank for r in answerable]
    return Scores(
        hit_rate=hit_rate(ranks),
        mrr=mrr(ranks),
        refusal_rate=(
            sum(r.refused for r in must_refuse) / len(must_refuse) if must_refuse else 1.0
        ),
        faithfulness=sum(j.faithful for j in judged) / len(judged) if judged else 0.0,
        correctness=sum(j.correct for j in judged) / len(judged) if judged else 0.0,
        latency_p50_ms=statistics.median(r.latency_ms for r in results) if results else 0.0,
        counts={
            "questions": len(results),
            "avec_reponse": len(answerable),
            "a_refuser": len(must_refuse),
            "erreurs": sum(r.error is not None for r in results),
        },
    )
