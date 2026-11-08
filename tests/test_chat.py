"""Tests du chat multi-tour (#30) : sessions, SSE, historique, persistance — sans réseau."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.engine import Engine

from puls_events_rag.api.app import app
from puls_events_rag.api.services import Services
from puls_events_rag.chat.agent import build_agent
from puls_events_rag.chat.service import ChatService
from puls_events_rag.chat.setup import build_chat, open_checkpointer
from puls_events_rag.chat.tools import build_tools
from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.database.sql import get_session
from tests.test_api_rag import INDEX, _services
from tests.test_auth import _headers, _register, _token, db_engine  # noqa: F401

SEARCH_CALL = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "search_events",
            "args": {"query": "atelier immersion", "communes": ["Aix-en-Provence"]},
            "id": "call-1",
        }
    ],
)


class ToolCallingFake(GenericFakeChatModel):
    """Modèle factice compatible ``create_agent`` (``bind_tools`` sans effet, pas de
    streaming : un message d'appel d'outil n'a pas de texte à découper)."""

    disable_streaming: bool = True

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ToolCallingFake:
        return self


def _script(*replies: AIMessage) -> ToolCallingFake:
    return ToolCallingFake(messages=iter(list(replies)))


def _chat(llm: ToolCallingFake, checkpointer: Any | None = None) -> ChatService:
    tools: list[BaseTool] = build_tools(lambda: INDEX, max_distance=None)
    return ChatService(build_agent(llm, tools, checkpointer=checkpointer or InMemorySaver()))


def _events(body: str) -> list[dict[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        data = [
            line.removeprefix("data: ") for line in block.splitlines() if line.startswith("data: ")
        ]
        if data:
            events.append(json.loads(data[0]))
    return events


@pytest.fixture
def chat_client(db_engine: Engine, tmp_path: Path) -> Iterator[TestClient]:  # noqa: F811
    from sqlmodel import Session

    def override() -> Iterator[Session]:
        with Session(db_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override
    app.state.services = _services(tmp_path, INDEX)
    app.state.chat = _chat(
        _script(
            SEARCH_CALL,
            AIMessage(content="Un atelier à Aix-en-Provence le 10 novembre."),
            AIMessage(content="Toujours à Aix, rien d'autre."),
        )
    )
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        del app.state.services, app.state.chat
        client.close()


def _login(client: TestClient, username: str = "alice") -> dict[str, str]:
    _register(client, username)
    return _headers(_token(client, username))


def _new_session(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post("/chat/sessions", json={}, headers=headers)
    assert response.status_code == 201, response.text
    session_id: str = response.json()["session_id"]
    return session_id


def test_message_streame_outils_sources_et_texte(chat_client: TestClient) -> None:
    headers = _login(chat_client)
    session_id = _new_session(chat_client, headers)

    response = chat_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"content": "Un atelier d'immersion à Aix ?"},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    types = [event["type"] for event in events]
    assert types[0] == "tool_start" and events[0]["tool"] == "search_events"
    assert "tool_end" in types and "token" in types
    assert types[-2:] == ["sources", "done"]
    assert [source["uid"] for source in events[-2]["events"]] == ["1"]
    text = "".join(event["content"] for event in events if event["type"] == "token")
    assert text == "Un atelier à Aix-en-Provence le 10 novembre."


def test_historique_multi_tour_et_titre(chat_client: TestClient) -> None:
    headers = _login(chat_client)
    session_id = _new_session(chat_client, headers)
    url = f"/chat/sessions/{session_id}/messages"
    chat_client.post(url, json={"content": "Un atelier d'immersion à Aix ?"}, headers=headers)
    chat_client.post(url, json={"content": "Et d'autres ?"}, headers=headers)

    detail = chat_client.get(f"/chat/sessions/{session_id}", headers=headers).json()

    roles = [message["role"] for message in detail["messages"]]
    assert roles == ["user", "tool", "assistant", "user", "assistant"]
    assert detail["messages"][2]["sources"][0]["title"] == "Atelier immersion"
    assert detail["title"] == "Un atelier d'immersion à Aix ?"
    listed = chat_client.get("/chat/sessions", headers=headers).json()
    assert [item["session_id"] for item in listed] == [session_id]


def test_session_d_un_autre_utilisateur_introuvable(chat_client: TestClient) -> None:
    alice = _login(chat_client, "alice")
    bob = _login(chat_client, "bob")
    session_id = _new_session(chat_client, alice)

    assert chat_client.get(f"/chat/sessions/{session_id}", headers=bob).status_code == 404
    message = {"content": "Coucou"}
    url = f"/chat/sessions/{session_id}/messages"
    assert chat_client.post(url, json=message, headers=bob).status_code == 404
    assert chat_client.delete(f"/chat/sessions/{session_id}", headers=bob).status_code == 404
    assert chat_client.get("/chat/sessions", headers=bob).json() == []


def test_authentification_requise(chat_client: TestClient) -> None:
    assert chat_client.post("/chat/sessions", json={}).status_code == 401


def test_message_vide_et_session_inconnue(chat_client: TestClient) -> None:
    headers = _login(chat_client)
    session_id = _new_session(chat_client, headers)

    url = f"/chat/sessions/{session_id}/messages"
    assert chat_client.post(url, json={"content": "  "}, headers=headers).status_code == 422
    unknown = f"/chat/sessions/{uuid.uuid4()}"
    assert chat_client.get(unknown, headers=headers).status_code == 404


def test_un_seul_message_a_la_fois(chat_client: TestClient) -> None:
    headers = _login(chat_client)
    session_id = _new_session(chat_client, headers)
    chat: ChatService = app.state.chat
    chat._active.add(uuid.UUID(session_id))

    response = chat_client.post(
        f"/chat/sessions/{session_id}/messages", json={"content": "Bonjour"}, headers=headers
    )

    assert response.status_code == 409


def test_suppression_de_session(chat_client: TestClient) -> None:
    headers = _login(chat_client)
    session_id = _new_session(chat_client, headers)

    assert chat_client.delete(f"/chat/sessions/{session_id}", headers=headers).status_code == 204
    assert chat_client.get(f"/chat/sessions/{session_id}", headers=headers).status_code == 404


def test_creation_refusee_sans_index(chat_client: TestClient, tmp_path: Path) -> None:
    headers = _login(chat_client)
    app.state.services = _services(tmp_path, None)

    assert chat_client.post("/chat/sessions", json={}, headers=headers).status_code == 503


def test_erreur_du_modele_dans_le_flux() -> None:
    chat = _chat(_script())  # aucun message prévu : le modèle factice lève une erreur

    async def run() -> list[str]:
        return [event.type async for event in chat.stream(uuid.uuid4(), "Bonjour")]

    assert asyncio.run(run()) == ["error"]


def test_historique_persiste_apres_redemarrage(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.db"
    session_id = uuid.uuid4()

    async def first_run() -> None:
        async with open_checkpointer(path) as saver:
            chat = _chat(_script(SEARCH_CALL, AIMessage(content="Réponse.")), saver)
            async for _ in chat.stream(session_id, "Un atelier à Aix ?"):
                pass

    async def after_restart() -> list[str]:
        async with open_checkpointer(path) as saver:
            chat = _chat(_script(), saver)
            return [message.role for message in await chat.history(session_id)]

    asyncio.run(first_run())

    assert asyncio.run(after_restart()) == ["user", "tool", "assistant"]


def test_outils_signalent_hors_zone_et_facettes() -> None:
    tools = {tool.name: tool for tool in build_tools(lambda: INDEX, max_distance=None)}

    async def run() -> tuple[Any, Any]:
        search = await tools["search_events"].ainvoke(
            {
                "type": "tool_call",
                "id": "c1",
                "name": "search_events",
                "args": {"query": "atelier", "communes": ["Lyon"]},
            }
        )
        themes = await tools["list_event_themes"].ainvoke({"communes": ["Aix-en-Provence"]})
        return search, themes

    search, themes = asyncio.run(run())

    assert "Hors de la métropole" in search.content and "Lyon" in search.content
    assert "1 événement(s) correspondent" in themes


def test_assemblage_du_chat(tmp_path: Path) -> None:
    services: Services = _services(tmp_path, INDEX)
    services.llm = _script()

    chat = build_chat(services, InMemorySaver())

    assert isinstance(chat, ChatService)


def test_index_courant_relu_a_chaque_appel() -> None:
    holder: dict[str, RegionalIndex] = {"index": INDEX}
    tools = build_tools(lambda: holder["index"], max_distance=None)

    assert {tool.name for tool in tools} == {"search_events", "list_event_themes"}
