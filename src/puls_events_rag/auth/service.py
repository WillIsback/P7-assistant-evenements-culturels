"""Accès aux utilisateurs et vérification des identifiants."""

from __future__ import annotations

from sqlmodel import Session, select

from puls_events_rag.auth.security import DUMMY_HASH, verify_password
from puls_events_rag.database.models import User


def get_user_by_username(session: Session, username: str) -> User | None:
    """Retourne l'utilisateur portant ce nom, ou ``None``."""
    return session.exec(select(User).where(User.username == username)).first()


def get_user_by_email(session: Session, email: str) -> User | None:
    """Retourne l'utilisateur portant cet email, ou ``None``."""
    return session.exec(select(User).where(User.email == email)).first()


def authenticate_user(session: Session, username: str, password: str) -> User | None:
    """Vérifie le couple identifiant / mot de passe.

    Si l'utilisateur n'existe pas, une vérification factice est effectuée pour
    garder un temps de réponse comparable et limiter l'énumération des comptes.
    """
    user = get_user_by_username(session, username)
    if user is None:
        verify_password(password, DUMMY_HASH)
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
