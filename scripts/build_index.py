"""Reconstruit l'index régional : collecte Open Agenda → documents → Faiss.

Usage ::

    uv run python scripts/build_index.py                 # collecte complète puis index
    uv run python scripts/build_index.py --skip-collect  # réindexe data/raw/events.jsonl

Avec ``CACHE_URL=redis://…`` (Valkey), les embeddings sont conservés : une
reconstruction quotidienne ne paie que les passages nouveaux ou modifiés. Sans
``CACHE_URL``, cache en mémoire : tout est revectorisé.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from dotenv import load_dotenv

from puls_events_rag.core.cache import setup_cache
from puls_events_rag.core.collect import RAW_PATH
from puls_events_rag.core.providers.openai_api import cached_embeddings
from puls_events_rag.core.sync import IndexSettings, sync_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-collect", action="store_true", help=f"réutiliser {RAW_PATH} sans appeler l'API"
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_cache()  # CACHE_URL (Valkey) sinon mémoire
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
