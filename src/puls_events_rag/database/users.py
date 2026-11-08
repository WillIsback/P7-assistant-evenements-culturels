"""Routes CRUD des utilisateurs, protégées par JWT."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlmodel import select

from puls_events_rag.auth.dependencies import ActiveUser
from puls_events_rag.auth.schemas import UserRead
from puls_events_rag.database.models import User
from puls_events_rag.database.sql import SessionDep

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserRead, summary="Profil de l'utilisateur connecté")
def read_current_user(current_user: ActiveUser) -> User:
    return current_user


@router.get("/", response_model=list[UserRead], summary="Lister les utilisateurs")
def read_users(
    session: SessionDep,
    current_user: ActiveUser,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> Sequence[User]:
    return session.exec(select(User).offset(offset).limit(limit)).all()


@router.get("/{user_id}", response_model=UserRead, summary="Lire un utilisateur")
def read_user(user_id: int, session: SessionDep, current_user: ActiveUser) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer son propre compte",
)
def delete_user(user_id: int, session: SessionDep, current_user: ActiveUser) -> None:
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own account",
        )

    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    session.delete(user)
    session.commit()
