"""Services partagés par l'API : index régional, LLM et chaîne RAG, créés une fois."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial

from fastapi import HTTPException, Request, status
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from puls_events_rag.core.index import RegionalIndex
from puls_events_rag.core.sync import IndexManager, IndexSettings, sync_index
from puls_events_rag.rag.chain import RagChain
from puls_events_rag.rag.query import LLMQueryParser, QueryParser

logger = logging.getLogger(__name__)


@dataclass
class Services:
    manager: IndexManager
    llm: BaseChatModel
    parser: QueryParser
    llm_profile: str

    def chain(self, index: RegionalIndex) -> RagChain:
        """Chaîne sur l'index courant (léger : prompt | llm, index déjà chargé)."""
        return RagChain(index=index, llm=self.llm, parser=self.parser)


def build_services(
    *,
    embeddings: Embeddings,
    model: str,
    llm: BaseChatModel,
    settings: IndexSettings | None = None,
) -> Services:
    """Assemble les services ; l'index est chargé depuis le disque s'il existe."""
    settings = settings or IndexSettings.from_env()
    manager = IndexManager(
        settings,
        loader=lambda directory: RegionalIndex.load(directory, embeddings, model=model),
        builder=partial(
            sync_index, embeddings=embeddings, model=model, directory=settings.directory
        ),
    )
    manager.load()
    return Services(
        manager=manager,
        llm=llm,
        parser=LLMQueryParser(llm),
        llm_profile=(os.getenv("LLM_PROFILE") or "local").lower(),
    )


def get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service non initialisé")
    return services
