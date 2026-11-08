"""Fenêtre de chat Gradio pour le développement : valide l'aller-retour front ↔ API.

Elle ne passe **que** par l'API HTTP (jamais par l'agent ni la base), comme un vrai
front. Lancer l'API puis ::

    uv run python scripts/chat_ui.py                  # http://127.0.0.1:7860
    API_URL=http://127.0.0.1:8001 GRADIO_SERVER_PORT=7861 uv run python scripts/chat_ui.py
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import gradio as gr
import httpx

API_URL = os.getenv("API_URL") or "http://127.0.0.1:8000"
TIMEOUT = httpx.Timeout(120.0, connect=5.0)
DEMO_USER, DEMO_PASSWORD = "demo", "demo-puls-2026"  # compte de démo du README

Messages = list[dict[str, str]]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Décode un flux ``text/event-stream`` en événements JSON (champ ``data``)."""
    for line in lines:
        if line.startswith("data: "):
            yield json.loads(line.removeprefix("data: "))


def format_sources(sources: list[dict[str, Any]]) -> str:
    lines = []
    for source in sources:
        title = source.get("title") or "Événement"
        link = f"[{title}]({source['url']})" if source.get("url") else title
        where = source.get("location") or ""
        if source.get("distance_km") is not None:
            where += f" — à {source['distance_km']} km"
        lines.append(f"- {link} · {source.get('date_range') or ''} · {where}")
    return "\n\n**Sources**\n" + "\n".join(lines) if lines else ""


def login(username: str, password: str) -> tuple[str, str, Any]:
    response = httpx.post(
        f"{API_URL}/auth/token", data={"username": username, "password": password}, timeout=10
    )
    if response.status_code != 200:
        return "", f"❌ Connexion refusée ({response.status_code})", gr.update(choices=[])
    token: str = response.json()["access_token"]
    return token, f"✅ Connecté en tant que {username}", gr.update(choices=list_sessions(token))


def list_sessions(token: str) -> list[tuple[str, str]]:
    if not token:
        return []
    response = httpx.get(f"{API_URL}/chat/sessions", headers=_headers(token), timeout=10)
    response.raise_for_status()
    return [
        (item["title"] or "Nouvelle conversation", item["session_id"]) for item in response.json()
    ]


def new_session(token: str) -> tuple[Any, Messages, str]:
    if not token:
        raise gr.Error("Connectez-vous d'abord.")
    response = httpx.post(f"{API_URL}/chat/sessions", json={}, headers=_headers(token), timeout=30)
    if response.status_code != 201:
        raise gr.Error(f"Création impossible ({response.status_code}) : {response.text}")
    session_id: str = response.json()["session_id"]
    return gr.update(choices=list_sessions(token), value=session_id), [], ""


def load_session(token: str, session_id: str | None) -> Messages:
    if not token or not session_id:
        return []
    response = httpx.get(
        f"{API_URL}/chat/sessions/{session_id}", headers=_headers(token), timeout=10
    )
    response.raise_for_status()
    history: Messages = []
    for message in response.json()["messages"]:
        if message["role"] == "user":
            history.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            content = message["content"] + format_sources(message["sources"])
            history.append({"role": "assistant", "content": content})
    return history


def send(
    message: str, history: Messages, token: str, session_id: str | None
) -> Iterator[tuple[Messages, str, str]]:
    """Envoie le message et affiche la réponse au fil des événements SSE."""
    if not token or not session_id:
        raise gr.Error("Connectez-vous et ouvrez une conversation.")
    history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    tools_log: list[str] = []
    text, sources = "", ""
    yield history, "", ""
    with httpx.stream(
        "POST",
        f"{API_URL}/chat/sessions/{session_id}/messages",
        json={"content": message},
        headers=_headers(token),
        timeout=TIMEOUT,
    ) as response:
        if response.status_code != 200:
            response.read()
            raise gr.Error(f"Erreur {response.status_code} : {response.text}")
        for event in parse_sse(response.iter_lines()):
            kind = event["type"]
            if kind == "token":
                text += event["content"]
            elif kind == "tool_start":
                tools_log.append(
                    f"🔧 `{event['tool']}` {json.dumps(event['args'], ensure_ascii=False)}"
                )
            elif kind == "tool_end":
                tools_log.append(f"   ↳ {event['summary']}")
            elif kind == "sources":
                sources = format_sources(event["events"])
            elif kind == "error":
                text += f"\n\n⚠️ {event['detail']}"
            history[-1] = {"role": "assistant", "content": text + sources}
            yield history, "", "\n\n".join(tools_log)


def build_ui() -> gr.Blocks:
    ui = gr.Blocks(title="Puls-Events — chat (dev)")
    with ui:
        token = gr.State("")
        gr.Markdown(f"## Puls-Events — fenêtre de chat de développement\nAPI : `{API_URL}`")
        with gr.Row():
            username = gr.Textbox(value=DEMO_USER, label="Utilisateur", scale=1)
            password = gr.Textbox(value=DEMO_PASSWORD, label="Mot de passe", type="password")
            login_button = gr.Button("Se connecter", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            sessions = gr.Dropdown(label="Conversations", choices=[], interactive=True, scale=3)
            new_button = gr.Button("Nouvelle conversation")
        chatbot = gr.Chatbot(label="Conversation", height=480)
        prompt = gr.Textbox(placeholder="Votre question…", label="Message")
        with gr.Accordion("Outils appelés (débogage)", open=False):
            tools = gr.Markdown()

        login_button.click(login, [username, password], [token, status, sessions])
        new_button.click(new_session, [token], [sessions, chatbot, tools])
        sessions.change(load_session, [token, sessions], [chatbot])
        prompt.submit(send, [prompt, chatbot, token, sessions], [chatbot, prompt, tools])
    return ui


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT") or 7860)
    build_ui().launch(server_name="127.0.0.1", server_port=port)
