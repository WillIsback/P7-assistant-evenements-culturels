"""Démo en ligne de commande : question → événements de l'index régional."""

import asyncio

from dotenv import load_dotenv

from puls_events_rag.core.cities import match_cities
from puls_events_rag.core.index import EventFilter, RegionalIndex
from puls_events_rag.core.providers.openai_api import cached_embeddings
from puls_events_rag.core.sync import IndexSettings


async def main() -> None:
    load_dotenv()
    embeddings = cached_embeddings()
    index = RegionalIndex.load(
        IndexSettings.from_env().directory, embeddings, model=embeddings.model
    )
    query = input("Quels événements cherchez-vous ? ")

    result = await index.search(query, where=EventFilter(communes=tuple(match_cities(query))))
    if result.radius_km is not None:
        print(f"(rien dans la commune demandée : zone élargie à {result.radius_km:g} km)")
    for rank, event in enumerate(result.events, start=1):
        meta = event.metadata
        where = meta["location"]
        if event.distance_km is not None:
            where += f" (à {event.distance_km} km)"
        print(f"{rank}. {meta['title']} — {meta['date_range']} — {where}")
        print(f"   {meta['url']} (score {event.score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
