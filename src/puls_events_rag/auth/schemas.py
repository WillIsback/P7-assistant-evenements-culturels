"""Schémas Pydantic exposés par l'API d'authentification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Token(BaseModel):
    """Réponse du point d'entrée OAuth2 ``/auth/token``."""

    access_token: str
    token_type: str = Field(default="bearer")


class TokenData(BaseModel):
    """Contenu minimal validé d'un JWT décodé."""

    sub: str | None = None


class UserCreate(BaseModel):
    """Données d'inscription ; le mot de passe est haché avant stockage."""

    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    email: str | None = Field(default=None, max_length=255)
    full_name: str | None = Field(default=None, max_length=255)


class UserRead(BaseModel):
    """Utilisateur renvoyé par l'API, sans le mot de passe haché."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str | None
    full_name: str | None
    disabled: bool
