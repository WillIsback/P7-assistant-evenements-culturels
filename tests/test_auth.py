"""Tests de l'authentification JWT et des routes utilisateurs (SQLite isolée)."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from puls_events_rag.api.app import app
from puls_events_rag.auth.security import create_access_token
from puls_events_rag.database.models import User
from puls_events_rag.database.sql import get_session

PASSWORD = "motdepasse123"  # noqa: S105 — mot de passe de test uniquement


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def client(db_engine: Engine) -> Iterator[TestClient]:
    def override_get_session() -> Generator[Session, None, None]:
        with Session(db_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()
        test_client.close()


def _register(
    client: TestClient,
    username: str = "alice",
    *,
    password: str = PASSWORD,
    email: str | None = None,
) -> dict[str, Any]:
    response = client.post(
        "/auth/register",
        json={
            "username": username,
            "password": password,
            "email": email if email is not None else f"{username}@example.com",
            "full_name": username.title(),
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _token(client: TestClient, username: str = "alice", password: str = PASSWORD) -> str:
    response = client.post(
        "/auth/token",
        data={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    token: str = response.json()["access_token"]
    return token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_inscription_retourne_l_utilisateur_sans_secret(client: TestClient) -> None:
    body = _register(client)

    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    assert body["disabled"] is False
    assert "hashed_password" not in body
    assert "password" not in body


def test_inscription_refuse_un_username_deja_pris(client: TestClient) -> None:
    _register(client)

    response = client.post(
        "/auth/register",
        json={"username": "alice", "password": PASSWORD},
    )

    assert response.status_code == 409


def test_inscription_refuse_un_email_deja_pris(client: TestClient) -> None:
    _register(client, email="partage@example.com")

    response = client.post(
        "/auth/register",
        json={"username": "bob", "password": PASSWORD, "email": "partage@example.com"},
    )

    assert response.status_code == 409


def test_mot_de_passe_stocke_hache(client: TestClient, db_engine: Engine) -> None:
    body = _register(client)

    with Session(db_engine) as session:
        user = session.get(User, body["id"])
        assert user is not None
        assert user.hashed_password != PASSWORD
        assert user.hashed_password.startswith("$argon2")


def test_connexion_et_profil(client: TestClient) -> None:
    _register(client)

    response = client.get("/users/me", headers=_headers(_token(client)))

    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert "hashed_password" not in response.json()


def test_acces_refuse_sans_jeton(client: TestClient) -> None:
    assert client.get("/users/me").status_code == 401
    assert client.get("/users/").status_code == 401


def test_connexion_refusee_avec_un_mauvais_mot_de_passe(client: TestClient) -> None:
    _register(client)

    response = client.post(
        "/auth/token",
        data={"username": "alice", "password": "mauvais-mot-de-passe"},
    )

    assert response.status_code == 401


def test_jeton_invalide_refuse(client: TestClient) -> None:
    response = client.get("/users/me", headers=_headers("pas-un-jeton"))

    assert response.status_code == 401


def test_jeton_expire_refuse(client: TestClient) -> None:
    _register(client)
    expired = create_access_token("alice", expires_delta=timedelta(seconds=-1))

    response = client.get("/users/me", headers=_headers(expired))

    assert response.status_code == 401


def test_liste_et_lecture_des_utilisateurs(client: TestClient) -> None:
    _register(client, "alice")
    bob = _register(client, "bob", email="bob@example.com")

    token = _token(client)
    listing = client.get("/users/", headers=_headers(token))
    detail = client.get(f"/users/{bob['id']}", headers=_headers(token))

    assert listing.status_code == 200
    assert {user["username"] for user in listing.json()} == {"alice", "bob"}
    assert detail.status_code == 200
    assert detail.json()["username"] == "bob"


def test_suppression_de_son_propre_compte(client: TestClient) -> None:
    alice = _register(client)
    token = _token(client)

    response = client.delete(f"/users/{alice['id']}", headers=_headers(token))

    assert response.status_code == 204
    assert (
        client.post(
            "/auth/token",
            data={"username": "alice", "password": PASSWORD},
        ).status_code
        == 401
    )


def test_suppression_du_compte_d_un_autre_est_refusee(client: TestClient) -> None:
    _register(client, "alice")
    bob = _register(client, "bob", email="bob@example.com")
    token = _token(client)

    response = client.delete(f"/users/{bob['id']}", headers=_headers(token))

    assert response.status_code == 403


def test_utilisateur_desactive_ne_peut_plus_se_connecter(
    client: TestClient,
    db_engine: Engine,
) -> None:
    alice = _register(client)
    token = _token(client)

    with Session(db_engine) as session:
        user = session.get(User, alice["id"])
        assert user is not None
        user.disabled = True
        session.add(user)
        session.commit()

    assert client.get("/users/me", headers=_headers(token)).status_code == 403
    assert (
        client.post(
            "/auth/token",
            data={"username": "alice", "password": PASSWORD},
        ).status_code
        == 401
    )
