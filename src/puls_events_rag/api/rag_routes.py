"""Routes RAG : POST /ask (#10), POST /rebuild et GET /rebuild (#25)."""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from puls_events_rag import __version__
from puls_events_rag.api.schemas import (
    AppliedFilters,
    AskRequest,
    AskResponse,
    CommuneCount,
    IndexStatus,
    MetadataResponse,
)
from puls_events_rag.api.services import Services, get_services
from puls_events_rag.core.providers.open_agenda import DEFAULT_AGENDA_UID, DEFAULT_LOOKBACK_DAYS
from puls_events_rag.core.sync import IndexManager, IndexUnavailableError

router = APIRouter(tags=["rag"])

ServicesDep = Annotated[Services, Depends(get_services)]


def index_status(manager: IndexManager) -> IndexStatus:
    current = manager.current
    if current is None:
        return IndexStatus(loaded=False, fresh=False, refreshing=manager.refreshing)
    manifest = current.manifest
    return IndexStatus(
        loaded=True,
        fresh=manager.is_fresh(),
        refreshing=manager.refreshing,
        event_count=manifest.event_count,
        document_count=manifest.document_count,
        synced_at=manifest.synced_at,
        embedding_model=manifest.embedding_model,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question (réponse augmentée et sources)",
    responses={503: {"description": "Index régional pas encore disponible"}},
)
async def ask(payload: AskRequest, services: ServicesDep) -> AskResponse:
    try:
        index = await services.manager.ensure_fresh()
    except IndexUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    result = await services.chain(index).ask(payload.question, k=payload.k)
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        filters=AppliedFilters(
            query=result.query.query,
            communes=list(result.query.to_filter().communes),
            since=result.query.since,
            until=result.query.until,
            radius_km=result.radius_km,
        ),
        latency_ms=result.latency_ms,
    )


def require_rebuild_token(authorization: Annotated[str | None, Header()] = None) -> None:
    """Jeton ``REBUILD_TOKEN`` (en-tête ``Authorization: Bearer``) ; route coupée sans lui."""
    expected = os.getenv("REBUILD_TOKEN")
    if not expected:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Reconstruction désactivée")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Jeton requis",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(authorization.removeprefix("Bearer "), expected):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Jeton invalide")


RebuildAuth = Annotated[None, Depends(require_rebuild_token)]


@router.post(
    "/rebuild",
    response_model=IndexStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reconstruire l'index régional (collecte + indexation, en tâche de fond)",
    responses={409: {"description": "Reconstruction déjà en cours"}},
)
async def rebuild(services: ServicesDep, _: RebuildAuth) -> IndexStatus:
    manager = services.manager
    if manager.refreshing:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Reconstruction déjà en cours")
    manager.start_background_refresh()
    return index_status(manager)


@router.get("/rebuild", response_model=IndexStatus, summary="État de l'index régional")
async def rebuild_status(services: ServicesDep, _: RebuildAuth) -> IndexStatus:
    return index_status(services.manager)


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Couverture et état du système (zone, période, volume, modèles)",
    responses={503: {"description": "Index régional pas encore disponible"}},
)
def metadata(services: ServicesDep) -> MetadataResponse:
    index = services.manager.current
    if index is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Index non chargé")
    return MetadataResponse(
        version=__version__,
        agenda_uid=os.getenv("OPENAGENDA_AGENDA_UID") or DEFAULT_AGENDA_UID,
        zone="Métropole Aix-Marseille-Provence",
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        index=index_status(services.manager),
        llm_profile=services.llm_profile,
        llm_model=str(getattr(services.llm, "model_name", type(services.llm).__name__)),
        max_distance=services.chain_max_distance,
        communes=[
            CommuneCount(commune=name, events=count)
            for name, count in index.commune_counts().items()
        ],
        top_keywords=index.facets().top_keywords,
    )
