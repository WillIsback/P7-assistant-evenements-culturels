"""Modèles SQLModel persistés en base."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """Utilisateur de l'API ; le mot de passe n'est stocké que sous forme hachée."""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50)
    email: str | None = Field(default=None, index=True, unique=True, max_length=255)
    full_name: str | None = Field(default=None, index=True, max_length=255)
    disabled: bool = Field(default=False)
    hashed_password: str = Field(max_length=255)


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class ChatSession(SQLModel, table=True):
    """Une conversation ; l'historique des messages est tenu par le checkpointer
    LangGraph (``thread_id = str(session_id)``), cette table porte la propriété."""

    __tablename__ = "chat_sessions"

    session_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    title: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow, index=True)
