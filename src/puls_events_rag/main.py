"""Démo en ligne de commande : question -> événements recommandés."""

import asyncio

from puls_events_rag.core.providers.open_agenda import OpenAgendaClient
from puls_events_rag.core.providers.openai_api import LangOpenAI
from puls_events_rag.core.semantic_search import retrieve_events


async def main() -> None:
    llm = LangOpenAI()
    query = input("Quels événements cherchez-vous ? ")

    async with OpenAgendaClient() as client:
        results = await retrieve_events(query, llm=llm, embeddings=llm.embeddings, client=client)

    for rank, event in enumerate(results, start=1):
        meta = event.metadata
        where = meta["location"]
        if meta.get("distance_km") is not None:
            where += f" (à {meta['distance_km']} km)"
        print(f"{rank}. {meta['title']} — {meta['date_range']} — {where}")
        print(f"   {meta['url']} (score {event.score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
