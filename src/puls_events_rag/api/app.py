"""Application FastAPI."""

from fastapi import FastAPI

from puls_events_rag import __version__

app = FastAPI(
    title="Puls-Events RAG",
    version=__version__,
    description="Assistant de recommandation d'événements culturels (Open Agenda).",
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Sonde de vivacité : état et version déployée."""
    return {"status": "ok", "version": __version__}
