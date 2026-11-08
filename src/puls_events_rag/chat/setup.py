"""Assemblage du chat au démarrage : checkpointer SQLite, outils, agent."""

from __future__ import annotations

import os
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Checkpointer

from puls_events_rag.api.services import Services
from puls_events_rag.chat.agent import build_agent
from puls_events_rag.chat.service import ChatService
from puls_events_rag.chat.tools import build_tools
from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.core.sync import IndexUnavailableError

DEFAULT_CHECKPOINT_DB = Path("data/checkpoints.db")


def open_checkpointer(path: Path | None = None) -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Checkpointer LangGraph persistant (fichier dédié ``CHECKPOINT_DB``)."""
    path = path or Path(os.getenv("CHECKPOINT_DB") or DEFAULT_CHECKPOINT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(path))


def build_chat(services: Services, checkpointer: Checkpointer) -> ChatService:
    def current_index() -> RegionalIndex:
        index = services.manager.current
        if index is None:
            raise IndexUnavailableError("Index régional indisponible")
        return index

    tools = build_tools(current_index, max_distance=services.chain_max_distance)
    return ChatService(build_agent(services.llm, tools, checkpointer=checkpointer))
