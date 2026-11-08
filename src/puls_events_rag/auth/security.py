"""Primitives de sécurité : hachage des mots de passe et jetons JWT."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from dotenv import load_dotenv
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from puls_events_rag.auth.schemas import TokenData

load_dotenv()

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30


def _load_secret_key() -> str:
    """Lit ``SECRET_KEY`` ; génère une clé éphémère en développement."""
    key = os.getenv("SECRET_KEY")
    if key:
        if len(key) < 32:
            logger.warning(
                "SECRET_KEY trop courte (%d caractères) : 32 caractères minimum recommandés",
                len(key),
            )
        return key
    logger.warning(
        "SECRET_KEY absente : génération d'une clé éphémère "
        "(les jetons seront invalidés au redémarrage)"
    )
    return secrets.token_urlsafe(32)


def _load_expire_minutes() -> int:
    raw = os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        str(DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    try:
        return int(raw)
    except ValueError:
        logger.warning("ACCESS_TOKEN_EXPIRE_MINUTES invalide (%r) : valeur par défaut", raw)
        return DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES


SECRET_KEY = _load_secret_key()
ACCESS_TOKEN_EXPIRE_MINUTES = _load_expire_minutes()

password_hash = PasswordHash.recommended()
DUMMY_HASH = password_hash.hash("dummypassword")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare un mot de passe en clair à son empreinte Argon2."""
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Calcule l'empreinte Argon2 d'un mot de passe."""
    return password_hash.hash(password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Signe un JWT dont le sujet est le nom d'utilisateur."""
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode({"sub": subject, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Décode et valide un JWT ; lève ``InvalidTokenError`` s'il est invalide."""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("Jeton sans sujet")
    return TokenData(sub=subject)
