"""Schémas Pydantic de l'API RAG (/ask, /health, /rebuild)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

from puls_events_rag.rag.chain import Source


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        examples=[
            "Je recherche un atelier d'immersion professionnelle à Aix-en-Provence "
            "dans les 15 prochains jours"
        ],
    )
    k: int = Field(default=5, ge=1, le=10, description="nombre maximal d'événements")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("la question ne doit pas être vide")
        return value.strip()


class AppliedFilters(BaseModel):
    query: str
    communes: list[str]
    since: date | None
    until: date | None
    radius_km: float | None = Field(description="rayon d'élargissement, si appliqué")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    filters: AppliedFilters
    latency_ms: int


class IndexStatus(BaseModel):
    loaded: bool
    fresh: bool
    refreshing: bool
    event_count: int | None = None
    document_count: int | None = None
    synced_at: datetime | None = None
    embedding_model: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_profile: str | None = None
    index: IndexStatus | None = None
