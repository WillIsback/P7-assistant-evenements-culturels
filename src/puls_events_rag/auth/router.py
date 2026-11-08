"""Routes d'inscription et de connexion (OAuth2 password flow)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from puls_events_rag.auth.schemas import Token, UserCreate, UserRead
from puls_events_rag.auth.security import create_access_token, get_password_hash
from puls_events_rag.auth.service import (
    authenticate_user,
    get_user_by_email,
    get_user_by_username,
)
from puls_events_rag.database.models import User
from puls_events_rag.database.sql import SessionDep

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Créer un compte",
)
def register(payload: UserCreate, session: SessionDep) -> User:
    """Crée un utilisateur ; le mot de passe est haché avant stockage."""
    if get_user_by_username(session, payload.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    if payload.email is not None and get_user_by_email(session, payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=get_password_hash(payload.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/token", response_model=Token, summary="Obtenir un jeton JWT")
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    """Vérifie les identifiants et renvoie un jeton Bearer."""
    user = authenticate_user(session, form_data.username, form_data.password)
    if user is None or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Token(access_token=create_access_token(subject=user.username))
