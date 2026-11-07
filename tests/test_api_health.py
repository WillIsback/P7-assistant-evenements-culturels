from fastapi.testclient import TestClient

from puls_events_rag import __version__
from puls_events_rag.api.app import app


def test_health_retourne_ok_et_version() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
