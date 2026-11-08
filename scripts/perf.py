"""Mesures de performance réelles (#12) : chargement de l'index, recherche, /ask.

Usage ::

    uv run python scripts/perf.py                                    # index seul
    uv run python scripts/perf.py --base-url http://127.0.0.1:8000   # + latence /ask

La recherche est mesurée hors appel d'embedding (vecteur de requête en cache) ; /ask
inclut le LLM (parseur + génération), donc dépend du profil LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx
from dotenv import load_dotenv

from puls_events_rag.core.cache import setup_cache
from puls_events_rag.core.cities import match_cities
from puls_events_rag.core.index import EventFilter, RegionalIndex
from puls_events_rag.core.providers.openai_api import cached_embeddings
from puls_events_rag.core.sync import IndexSettings

QUESTIONS = (
    "atelier d'immersion professionnelle à Aix-en-Provence",
    "forum de recrutement à Marseille",
    "création d'entreprise",
    "atelier CV",
    "job dating à Aubagne",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def measure_index(rounds: int) -> None:
    embeddings = cached_embeddings()
    started = time.perf_counter()
    index = RegionalIndex.load(
        IndexSettings.from_env().directory, embeddings, model=embeddings.model
    )
    load_s = time.perf_counter() - started
    print(
        f"Chargement de l'index : {load_s * 1000:.0f} ms ({index.manifest.event_count} événements)"
    )
    for question in QUESTIONS:  # vecteurs de requête mis en cache
        await embeddings.aembed_query(question)
    timings = []
    for _ in range(rounds):
        for question in QUESTIONS:
            where = EventFilter(communes=tuple(match_cities(question)))
            started = time.perf_counter()
            await index.search(question, k=5, where=where)
            timings.append((time.perf_counter() - started) * 1000)
    p50, p95 = statistics.median(timings), percentile(timings, 0.95)
    print(f"Recherche filtrée (k=5, {len(timings)} requêtes) : p50 {p50:.1f} ms, p95 {p95:.1f} ms")


def measure_ask(base_url: str, rounds: int) -> None:
    timings = []
    with httpx.Client(base_url=base_url, timeout=120) as client:
        for _ in range(rounds):
            for question in QUESTIONS:
                started = time.perf_counter()
                client.post("/ask", json={"question": question, "k": 3}).raise_for_status()
                timings.append(time.perf_counter() - started)
    print(
        f"POST /ask ({len(timings)} requêtes) : p50 {statistics.median(timings):.2f} s,"
        f" p95 {percentile(timings, 0.95):.2f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", help="API à mesurer (sinon index seul)")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    load_dotenv()
    setup_cache()  # CACHE_URL (Valkey) sinon mémoire
    asyncio.run(measure_index(args.rounds))
    if args.base_url:
        measure_ask(args.base_url, max(1, args.rounds // 2))


if __name__ == "__main__":
    main()
