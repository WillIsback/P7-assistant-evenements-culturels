"""Exécution de l'agent pour une session : flux d'événements et historique."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from puls_events_rag.chat.schemas import (
    ChatEvent,
    ChatMessage,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from puls_events_rag.rag.chain import Source

logger = logging.getLogger(__name__)

MODEL_NODE = "model"


def _config(session_id: uuid.UUID) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(session_id)}}


def _sources(artifact: Any) -> list[Source]:
    if not isinstance(artifact, list):
        return []
    return [Source.model_validate(item) for item in artifact if isinstance(item, dict)]


def _dedupe(sources: Iterable[Source]) -> list[Source]:
    return list({source.uid: source for source in sources}.values())


def _summary(content: Any) -> str:
    text = content if isinstance(content, str) else str(content)
    return text.splitlines()[0][:200] if text else ""


class ChatService:
    """Agent partagé ; un seul message en cours par session (verrou en mémoire)."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._active: set[uuid.UUID] = set()

    def is_busy(self, session_id: uuid.UUID) -> bool:
        return session_id in self._active

    async def stream(self, session_id: uuid.UUID, content: str) -> AsyncIterator[ChatEvent]:
        """Envoie ``content`` et traduit le flux LangGraph en événements SSE."""
        self._active.add(session_id)
        sources: list[Source] = []
        try:
            async for mode, chunk in self.agent.astream(
                {"messages": [HumanMessage(content=content)]},
                _config(session_id),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message, metadata = chunk
                    # Fragment (modèle en streaming) ou message complet (modèle sans).
                    if (
                        isinstance(message, AIMessage)
                        and metadata.get("langgraph_node") == MODEL_NODE
                        and message.text
                    ):
                        yield TokenEvent(content=message.text)
                    continue
                for update in (chunk or {}).values():
                    for message in (
                        (update or {}).get("messages", []) if isinstance(update, dict) else []
                    ):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                yield ToolStartEvent(tool=call["name"], args=call["args"])
                        elif isinstance(message, ToolMessage):
                            found = _sources(message.artifact)
                            sources.extend(found)
                            yield ToolEndEvent(
                                tool=message.name or "outil", summary=_summary(message.content)
                            )
            if sources:
                yield SourcesEvent(events=_dedupe(sources))
            yield DoneEvent()
        except Exception as exc:  # l'erreur part dans le flux : le statut HTTP est déjà 200
            logger.exception("Échec du tour de chat %s", session_id)
            yield ErrorEvent(detail=f"Erreur pendant la réponse : {type(exc).__name__}")
        finally:
            self._active.discard(session_id)

    async def history(self, session_id: uuid.UUID) -> list[ChatMessage]:
        """Reconstruit l'historique depuis l'état du checkpointer."""
        state = await self.agent.aget_state(_config(session_id))
        messages: list[BaseMessage] = (state.values or {}).get("messages", [])
        history: list[ChatMessage] = []
        pending: list[Source] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                pending = []
                history.append(ChatMessage(role="user", content=message.text))
            elif isinstance(message, ToolMessage):
                found = _sources(message.artifact)
                pending.extend(found)
                history.append(
                    ChatMessage(
                        role="tool", content=message.text, tool_name=message.name, sources=found
                    )
                )
            elif isinstance(message, AIMessage) and message.text.strip():
                history.append(
                    ChatMessage(role="assistant", content=message.text, sources=_dedupe(pending))
                )
                pending = []
        return history

    async def delete_thread(self, session_id: uuid.UUID) -> None:
        checkpointer = getattr(self.agent, "checkpointer", None)
        if checkpointer is not None:
            await checkpointer.adelete_thread(str(session_id))
