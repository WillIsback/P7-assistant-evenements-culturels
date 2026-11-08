"""Application FastAPI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from puls_events_rag import __version__
from puls_events_rag.api.rag_routes import index_status
from puls_events_rag.api.rag_routes import router as rag_router
from puls_events_rag.api.schemas import HealthResponse
from puls_events_rag.api.services import Services, build_services
from puls_events_rag.auth.router import router as auth_router
from puls_events_rag.core.providers.openai_api import build_client, cached_embeddings
from puls_events_rag.database.sql import create_db_and_tables
from puls_events_rag.database.users import router as users_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prépare la base, puis charge l'index régional et le LLM une seule fois.

    Sans configuration LLM/embeddings, l'API démarre quand même (auth, /health) ;
    les routes RAG répondent alors 503.
    """
    create_db_and_tables()
    try:
        embeddings = cached_embeddings()
        app.state.services = build_services(
            embeddings=embeddings, model=embeddings.model, llm=build_client()
        )
    except RuntimeError as exc:
        logger.error("Services RAG non initialisés : %s", exc)
        app.state.services = None
    yield
    services: Services | None = app.state.services
    if services is not None:
        await services.manager.stop()


app = FastAPI(
    title="Puls-Events RAG",
    version=__version__,
    description="Assistant de recommandation d'événements (Open Agenda, métropole "
    "Aix-Marseille-Provence) : RAG LangChain + Faiss.",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(rag_router)


@app.get("/health", tags=["ops"], response_model=HealthResponse, response_model_exclude_none=True)
def health(request: Request) -> HealthResponse:
    """Sonde de vivacité : état, version, profil LLM et index chargé."""
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        return HealthResponse(status="ok", version=__version__)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_profile=services.llm_profile,
        index=index_status(services.manager),
    )
