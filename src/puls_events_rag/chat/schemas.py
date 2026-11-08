"""Schémas Pydantic du chat : sessions, messages, événements SSE."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from puls_events_rag.rag.chain import Source


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("le message ne doit pas être vide")
        return value.strip()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    sources: list[Source] = Field(default_factory=list)


class ChatSessionDetail(ChatSessionRead):
    messages: list[ChatMessage] = Field(default_factory=list)


# -- événements SSE (champ ``type`` discriminant) -------------------------------------


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolEndEvent(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    tool: str
    summary: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    events: list[Source]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str


ChatEvent = TokenEvent | ToolStartEvent | ToolEndEvent | SourcesEvent | DoneEvent | ErrorEvent
