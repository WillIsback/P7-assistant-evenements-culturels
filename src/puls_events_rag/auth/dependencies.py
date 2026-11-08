"""Dépendances FastAPI d'authentification : jeton Bearer -> utilisateur courant."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError

from puls_events_rag.auth.security import decode_access_token
from puls_events_rag.auth.service import get_user_by_username
from puls_events_rag.database.models import User
from puls_events_rag.database.sql import SessionDep

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep,
) -> User:
    """Décode le JWT et charge l'utilisateur correspondant."""
    try:
        token_data = decode_access_token(token)
    except InvalidTokenError as exc:
        raise _credentials_error() from exc

    user = get_user_by_username(session, token_data.sub or "")
    if user is None:
        raise _credentials_error()
    return user


def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Refuse les comptes désactivés."""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


CurrentUser = Annotated[User, Depends(get_current_user)]
ActiveUser = Annotated[User, Depends(get_current_active_user)]
