"""Évaluation de la chaîne RAG sur le jeu annoté — contrat de la gate de release (#14).

Usage ::

    uv run python scripts/evaluate.py --output reports/evaluation.md
    uv run python scripts/evaluate.py --limit 5            # essai rapide

Exécute ``POST /ask`` en interne (même chaîne) sur ``eval/dataset.jsonl``, calcule les
métriques (récupération : hit rate et MRR ; refus ; génération : fidélité et
exactitude par LLM-juge), écrit le rapport Markdown et sort en erreur (code 1) si un
seuil n'est pas atteint. Profil LLM : ``LLM_PROFILE`` (``ci`` en release).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback

from puls_events_rag import __version__
from puls_events_rag.core.cache import setup_cache
from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.core.providers.openai_api import build_client, cached_embeddings
from puls_events_rag.core.sync import IndexSettings
from puls_events_rag.evaluation.dataset import DATASET_PATH, EvalItem, load_dataset
from puls_events_rag.evaluation.metrics import ItemResult, Judge, Scores, first_hit_rank, score
from puls_events_rag.rag.chain import RagChain
from puls_events_rag.rag.query import LLMQueryParser

THRESHOLDS = {  # seuils par défaut, surchargés par --min-<métrique> ou EVAL_MIN_<MÉTRIQUE>
    "hit_rate": 0.70,
    "mrr": 0.50,
    "refusal_rate": 0.80,
    "faithfulness": 0.85,
    "correctness": 0.70,
}
CONCURRENCY = 4
# Estimation de coût ($ par million de jetons), à ajuster au modèle du profil.
PRICE_INPUT = float(os.getenv("EVAL_PRICE_INPUT_PER_M") or 0.10)
PRICE_OUTPUT = float(os.getenv("EVAL_PRICE_OUTPUT_PER_M") or 0.30)


async def evaluate_item(
    item: EvalItem, chain: RagChain, judge: Judge, semaphore: asyncio.Semaphore
) -> ItemResult:
    async with semaphore:
        try:
            answer = await chain.ask(item.question)
            uids = [source.uid for source in answer.sources]
            verdict = await judge(item, answer.answer, answer.context)
        except Exception as exc:  # une question en échec ne doit pas arrêter le lot
            return ItemResult(item, "", [], 0, None, error=f"{type(exc).__name__}: {exc}")
        return ItemResult(
            item,
            answer.answer,
            answer.sources,
            answer.latency_ms,
            verdict,
            rank=first_hit_rank(item.expected_uids, uids),
        )


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def render_report(
    results: list[ItemResult],
    scores: Scores,
    thresholds: dict[str, float],
    *,
    model: str,
    index: RegionalIndex,
    duration_s: float,
    tokens: tuple[int, int],
) -> str:
    values = scores.as_dict()
    cost = tokens[0] / 1e6 * PRICE_INPUT + tokens[1] / 1e6 * PRICE_OUTPUT
    manifest = index.manifest
    lines = [
        f"# Évaluation RAG — v{__version__}",
        "",
        f"- Date : {datetime.now(tz=UTC):%Y-%m-%d %H:%M} UTC ; "
        f"modèle : `{model}` (génération et juge)",
        f"- Index : {manifest.event_count} événements, "
        f"synchronisé le {manifest.synced_at:%Y-%m-%d}",
        f"- Questions : {scores.counts['questions']} ({scores.counts['avec_reponse']} avec "
        f"réponse attendue, {scores.counts['a_refuser']} à refuser), erreurs : "
        f"{scores.counts['erreurs']}",
        f"- Durée : {duration_s:.0f} s ; latence p50 par question : "
        f"{scores.latency_p50_ms / 1000:.1f} s",
        f"- Jetons : {tokens[0]} en entrée, {tokens[1]} en sortie ; coût estimé : {cost:.4f} $",
        "",
        "| Métrique | Valeur | Seuil | |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {values[name]:.2f} | {thresholds[name]:.2f} | "
        f"{_mark(values[name] >= thresholds[name])} |"
        for name in thresholds
    ]
    lines += [
        "",
        "## Détail par question",
        "",
        "| id | catégorie | rang | sources | fidèle | correcte | latence | justification |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r.judgement
        why = (r.error or (j.rationale if j else "")).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r.item.id} | {r.item.category} | {r.rank or '-'} | {len(r.sources)} | "
            f"{_mark(j.faithful) if j else '-'} | {_mark(j.correct) if j else '-'} | "
            f"{r.latency_ms / 1000:.1f} s | {why[:140]} |"
        )
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> int:
    items = load_dataset(args.dataset)[: args.limit or None]
    embeddings = cached_embeddings()
    index = RegionalIndex.load(
        IndexSettings.from_env().directory, embeddings, model=embeddings.model
    )
    llm = build_client()
    chain = RagChain(index=index, llm=llm, parser=LLMQueryParser(llm))
    judge = Judge(llm)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    started = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        results = await asyncio.gather(
            *(evaluate_item(item, chain, judge, semaphore) for item in items)
        )
    duration = time.perf_counter() - started
    tokens = (
        sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values()),
        sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values()),
    )

    scores = score(results)
    thresholds = {name: getattr(args, f"min_{name}") for name in THRESHOLDS}
    report = render_report(
        list(results),
        scores,
        thresholds,
        model=str(getattr(llm, "model_name", "inconnu")),
        index=index,
        duration_s=duration,
        tokens=tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report.split("## Détail")[0])
    failed = [name for name, value in scores.as_dict().items() if value < thresholds[name]]
    if failed:
        print(f"❌ Seuils non atteints : {', '.join(failed)}", file=sys.stderr)
        return 1
    print("✅ Tous les seuils sont atteints.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.md"))
    parser.add_argument("--limit", type=int, default=0, help="n premières questions (essai)")
    for name, default in THRESHOLDS.items():
        env = os.getenv(f"EVAL_MIN_{name.upper()}")
        parser.add_argument(
            f"--min-{name.replace('_', '-')}",
            dest=f"min_{name}",
            type=float,
            default=float(env) if env else default,
        )
    args = parser.parse_args()
    load_dotenv()
    setup_cache(os.getenv("CACHE_URL") or "disk://?directory=./data/cache&shards=4")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
