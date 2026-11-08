"""Reconstruit l'index régional : collecte Open Agenda → documents → Faiss.

Usage ::

    uv run python scripts/build_index.py                 # collecte complète puis index
    uv run python scripts/build_index.py --skip-collect  # réindexe data/raw/events.jsonl

Les embeddings passent par un cache disque (``data/cache`` sauf ``CACHE_URL``) :
une reconstruction quotidienne ne paie que les passages nouveaux ou modifiés.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

from puls_events_rag.core.cache import setup_cache
from puls_events_rag.core.collect import RAW_PATH
from puls_events_rag.core.providers.openai_api import cached_embeddings
from puls_events_rag.core.sync import IndexSettings, sync_index

DEFAULT_SCRIPT_CACHE = "disk://?directory=./data/cache&shards=4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-collect", action="store_true", help=f"réutiliser {RAW_PATH} sans appeler l'API"
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_cache(os.getenv("CACHE_URL") or DEFAULT_SCRIPT_CACHE)
    settings = IndexSettings.from_env()
    embeddings = cached_embeddings()

    started = time.perf_counter()
    index = asyncio.run(
        sync_index(
            embeddings=embeddings,
            model=embeddings.model,
            directory=settings.directory,
            skip_collect=args.skip_collect,
        )
    )
    manifest = index.manifest
    print(
        f"Index régional : {manifest.event_count} événements, {manifest.document_count} passages,"
        f" modèle {manifest.embedding_model}, {time.perf_counter() - started:.0f} s"
        f" → {settings.directory}"
    )


if __name__ == "__main__":
    main()
