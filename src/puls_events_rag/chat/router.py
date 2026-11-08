"""Routes du chat : sessions (CRUD) et envoi d'un message en streaming SSE."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session, col, select

from puls_events_rag.api.services import Services, get_services
from puls_events_rag.auth.dependencies import ActiveUser
from puls_events_rag.chat.schemas import (
    ChatEvent,
    ChatMessageIn,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionRead,
)
from puls_events_rag.chat.service import ChatService
from puls_events_rag.core.sync import IndexUnavailableError
from puls_events_rag.database.models import ChatSession
from puls_events_rag.database.sql import SessionDep

router = APIRouter(prefix="/chat", tags=["chat"])

TITLE_LENGTH = 60


def get_chat(request: Request) -> ChatService:
    chat: ChatService | None = getattr(request.app.state, "chat", None)
    if chat is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat non initialisé")
    return chat


ChatDep = Annotated[ChatService, Depends(get_chat)]
ServicesDep = Annotated[Services, Depends(get_services)]


def _owned(session: Session, session_id: uuid.UUID, user_id: int | None) -> ChatSession:
    chat_session = session.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user_id:
        # Même réponse si la session existe chez un autre : rien n'est révélé.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session introuvable")
    return chat_session


def _sse(event: ChatEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"event: {event.type}\ndata: {data}\n\n"


@router.post(
    "/sessions",
    response_model=ChatSessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Ouvrir une conversation",
    responses={503: {"description": "Index régional pas encore disponible"}},
)
async def create_session(
    payload: ChatSessionCreate, session: SessionDep, user: ActiveUser, services: ServicesDep
) -> ChatSession:
    """Vérifie la fraîcheur de l'index (resynchro de fond si > 24 h) puis crée la session."""
    try:
        await services.manager.ensure_fresh()
    except IndexUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    assert user.id is not None  # noqa: S101 — utilisateur chargé depuis la base
    chat_session = ChatSession(user_id=user.id, title=payload.title)
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


@router.get("/sessions", response_model=list[ChatSessionRead], summary="Mes conversations")
def list_sessions(
    session: SessionDep,
    user: ActiveUser,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> Sequence[ChatSession]:
    query = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(col(ChatSession.updated_at).desc())
        .offset(offset)
        .limit(limit)
    )
    return session.exec(query).all()


@router.get(
    "/sessions/{session_id}", response_model=ChatSessionDetail, summary="Conversation et historique"
)
async def read_session(
    session_id: uuid.UUID, session: SessionDep, user: ActiveUser, chat: ChatDep
) -> ChatSessionDetail:
    chat_session = _owned(session, session_id, user.id)
    detail = ChatSessionDetail.model_validate(chat_session)
    detail.messages = await chat.history(session_id)
    return detail


@router.post(
    "/sessions/{session_id}/messages",
    summary="Envoyer un message (réponse en streaming SSE)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Événements token, tool_start, tool_end, sources, done, error",
        },
        409: {"description": "Une réponse est déjà en cours pour cette session"},
    },
)
async def send_message(
    session_id: uuid.UUID,
    payload: ChatMessageIn,
    session: SessionDep,
    user: ActiveUser,
    chat: ChatDep,
) -> StreamingResponse:
    chat_session = _owned(session, session_id, user.id)
    if chat.is_busy(session_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Réponse déjà en cours")
    if not chat_session.title:
        title = payload.content
        chat_session.title = title if len(title) <= TITLE_LENGTH else title[:TITLE_LENGTH] + "…"
    chat_session.updated_at = datetime.now(tz=UTC)
    session.add(chat_session)
    session.commit()

    async def events() -> AsyncIterator[str]:
        async for event in chat.stream(session_id, payload.content):
            yield _sse(event)

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer une conversation",
)
async def delete_session(
    session_id: uuid.UUID, session: SessionDep, user: ActiveUser, chat: ChatDep
) -> None:
    chat_session = _owned(session, session_id, user.id)
    session.delete(chat_session)
    session.commit()
    await chat.delete_thread(session_id)
