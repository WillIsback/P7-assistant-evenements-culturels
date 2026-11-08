from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from puls_events_rag import __version__
from puls_events_rag.api.app import app


def test_health_retourne_ok_et_version() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_lifespan_cree_les_tables_meme_sans_llm(monkeypatch: MonkeyPatch) -> None:
    appels: list[bool] = []
    monkeypatch.setattr(
        "puls_events_rag.api.app.create_db_and_tables",
        lambda: appels.append(True),
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.post("/ask", json={"question": "Un concert ?"}).status_code == 503

    assert appels == [True]
