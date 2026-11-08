"""Adaptateur compatible OpenAI : chat (vLLM ou OpenRouter) et embeddings (OpenRouter)."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from puls_events_rag.core.cache import (
    TTL_KEYWORDS,
    CachedEmbeddings,
    cache_get,
    cache_set,
    fingerprint,
    make_key,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "qwen3.8-flash-next-a5b"
DEFAULT_CI_MODEL = "mistralai/mistral-small-2603"
DEFAULT_EMBEDDING_MODEL = "mistralai/mistral-embed-2312"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_KEYWORDS = 8

SYSTEM_PROMPT = """Tu extrais les mots-clés d'une question sur des événements culturels.
Règles :
- 1 à 8 mots-clés, en français, courts : thème, discipline, public, lieu ou nom propre.
- Conserve les noms propres tels quels et n'invente aucun terme absent de la question.
- Ignore les mots vides (quel, événement, lieu, à, de...).
- Réponds UNIQUEMENT par un objet JSON valide, sans texte autour :
  {"keywords": ["premier mot-clé", "second mot-clé"]}
"""


class KeywordExtractionError(RuntimeError):
    """Le LLM n'a pas renvoyé le contrat JSON attendu."""


class EventKeywords(BaseModel):
    """Sortie structurée du LLM, validée avant l'appel à Open Agenda."""

    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def _normalize(cls, keywords: list[str]) -> list[str]:
        """Nettoie, déduplique (insensible à la casse) et borne la liste."""
        unique: dict[str, str] = {}
        for keyword in keywords:
            cleaned = " ".join(keyword.split())
            if cleaned and cleaned.casefold() not in unique:
                unique[cleaned.casefold()] = cleaned
        return list(unique.values())[:MAX_KEYWORDS]


def _env(name: str) -> str | None:
    """Lit une variable d'environnement non vide."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _with_scheme(url: str) -> str:
    """Ajoute http:// si l'URL est saisie sans schéma (ex. 192.168.1.87:30000/v1)."""
    return url if "://" in url else f"http://{url}"


def _openrouter_api_key(usage: str) -> str:
    """Renvoie la clé OpenRouter ou lève une erreur explicite."""
    api_key = _env("OPENROUTER_API_KEY")
    if api_key is None:
        raise RuntimeError(f"OPENROUTER_API_KEY manquante pour {usage}")
    return api_key


def build_client() -> ChatOpenAI:
    """Construit le client de chat choisi par ``LLM_PROFILE`` (local ou ci)."""
    profile = (_env("LLM_PROFILE") or "local").lower()

    if profile == "ci":
        base_url = _env("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
        api_key = _openrouter_api_key("LLM_PROFILE=ci")
        model = _env("OPENROUTER_CHAT_MODEL") or DEFAULT_CI_MODEL
        extra_body = {"reasoning": {"enabled": False}}
    elif profile == "local":
        # Compatibilité avec l'ancien couple LLM_URL / LLM_API_KEY.
        base_url = _env("VLLM_BASE_URL") or _env("LLM_URL") or "http://localhost:8000/v1"
        api_key = _env("VLLM_API_KEY") or _env("LLM_API_KEY") or "not-needed"
        model = _env("VLLM_CHAT_MODEL") or _env("LLM_MODEL") or DEFAULT_LOCAL_MODEL
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        raise RuntimeError(f"LLM_PROFILE inconnu : {profile!r} (attendu : local ou ci)")

    return ChatOpenAI(
        base_url=_with_scheme(base_url),
        api_key=SecretStr(api_key),
        model=model,
        temperature=0,
        extra_body=extra_body,
        timeout=30,
        max_retries=2,
    )


def build_embeddings() -> OpenAIEmbeddings:
    """Construit le client d'embeddings OpenRouter, identique dans tous les profils.

    Le vLLM local n'expose pas ``/v1/embeddings`` et un index Faiss n'est valide
    qu'avec le modèle qui l'a construit : on passe donc toujours par OpenRouter.
    """
    return OpenAIEmbeddings(
        base_url=_env("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
        api_key=SecretStr(_openrouter_api_key("les embeddings")),
        model=_env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        # Modèle non OpenAI : envoyer le texte brut, pas des jetons tiktoken.
        check_embedding_ctx_length=False,
        # Lots modestes : l'API Mistral plafonne le nombre de jetons par requête.
        chunk_size=32,
        timeout=30,
        max_retries=2,
    )


def cached_embeddings() -> CachedEmbeddings:
    """Embeddings OpenRouter derrière le cache (seuls les textes nouveaux sont payés)."""
    embeddings = build_embeddings()
    return CachedEmbeddings(embeddings, model=embeddings.model)


class LangOpenAI:
    """Façade LLM du projet ; chat et embeddings sont injectables pour les tests."""

    def __init__(
        self,
        client: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.client: BaseChatModel = client if client is not None else build_client()
        self._embeddings = embeddings
        self._parser = JsonOutputParser(pydantic_object=EventKeywords)

    @property
    def embeddings(self) -> Embeddings:
        """Client d'embeddings, construit à la première utilisation.

        La construction paresseuse évite d'exiger ``OPENROUTER_API_KEY`` quand
        seule l'extraction de mots-clés est utilisée.
        """
        if self._embeddings is None:
            self._embeddings = cached_embeddings()
        return self._embeddings

    @property
    def model_name(self) -> str:
        """Identifiant du modèle de chat, inclus dans les clés de cache."""
        name = getattr(self.client, "model_name", None)
        return str(name) if name else type(self.client).__name__

    async def get_keywords(self, query: str) -> list[str]:
        """Extrait une liste de mots-clés validés depuis ``query``.

        Lève :class:`KeywordExtractionError` si le LLM ne respecte pas le
        contrat JSON, afin que l'appelant puisse décider d'un repli. Les
        réponses valides sont mises en cache (question normalisée, modèle,
        prompt) ; une réponse hors contrat ne l'est jamais.
        """
        if not query.strip():
            return []

        normalized = " ".join(query.casefold().split())
        key = make_key("keywords", self.model_name, fingerprint(SYSTEM_PROMPT), normalized)
        cached = await cache_get(key)
        if isinstance(cached, list):
            logger.info("Mots-clés en cache pour %r", query)
            return [str(keyword) for keyword in cached]

        response = await self.client.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
        )
        try:
            parsed = self._parser.parse(response.text)
            keywords = EventKeywords.model_validate(parsed).keywords
        except (OutputParserException, ValidationError) as exc:
            logger.warning("Réponse LLM hors contrat : %s", response.text[:200])
            raise KeywordExtractionError(
                "Le LLM n'a pas renvoyé un objet JSON {'keywords': [...]}"
            ) from exc
        await cache_set(key, keywords, TTL_KEYWORDS)
        return keywords
