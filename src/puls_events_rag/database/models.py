"""Modèles SQLModel persistés en base."""

from __future__ import annotations

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
