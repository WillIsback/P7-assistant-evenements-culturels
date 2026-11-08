"""Connexion à la base SQLModel et dépendance de session FastAPI."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine

import puls_events_rag.database.models  # noqa: F401  (enregistre les tables)

load_dotenv()

# ``or`` : une variable présente mais vide (``DATABASE_URL=``) prend aussi le défaut.
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./data/database.db"


def _ensure_sqlite_dir(url: str) -> None:
    """Crée le dossier d'un fichier SQLite (``sqlite:///./data/x.db`` -> ``./data``)."""
    prefix = "sqlite:///"
    path = url.removeprefix(prefix).split("?", 1)[0]
    if url.startswith(prefix) and path and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(DATABASE_URL)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Crée les tables manquantes (opération idempotente)."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Fournit une session par requête, fermée automatiquement."""
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
