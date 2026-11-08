"""Cache des appels coûteux (LLM, embeddings, Open Agenda) sur Valkey via cashews.

``CACHE_URL`` choisit le backend :

- ``disk://?directory=/app/cache`` : disque (diskcache/SQLite), valeur de
  l'image Docker : partagé entre workers, survit aux redémarrages ;
- ``redis://hote:6379/0`` : Valkey (protocole Redis), si un serveur existe ;
- absent : cache **en mémoire** du processus (dev, tests) ;
- ``CACHE_ENABLED=false`` désactive tout cache.

Les valeurs sont sérialisées en **JSON**, jamais en pickle : relire un pickle
depuis un cache partagé permettrait d'exécuter du code arbitraire. Si Valkey
est injoignable, cashews ignore l'erreur (``suppress``) et l'appel est
simplement refait : le cache ne fait jamais échouer une requête.

Les clés sont versionnées par tout ce qui change le résultat (modèle, prompt,
paramètres) puis hachées : changer de modèle invalide de fait l'ancien cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

from cashews import Cache
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Durées de vie (secondes).
TTL_KEYWORDS = 7 * 24 * 3600  # même question + même modèle + même prompt -> mêmes mots-clés
TTL_EMBEDDINGS = 30 * 24 * 3600  # un texte a toujours le même vecteur pour un modèle donné
TTL_OPEN_AGENDA = 30 * 60  # l'agenda évolue : fraîcheur d'une demi-heure

MEMORY_URL = "mem://?size=10000"

cache = Cache()


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def setup_cache(url: str | None = None, *, enabled: bool | None = None) -> None:
    """Configure le cache global (idempotent : seul le premier appel compte)."""
    if cache.is_setup():
        return
    if enabled is None:
        enabled = (_env("CACHE_ENABLED") or "true").lower() not in {"0", "false", "no"}
    url = url or _env("CACHE_URL") or MEMORY_URL
    cache.setup(url, pickle_type="json", enable=enabled)
    logger.info("Cache %s : %s", "actif" if enabled else "désactivé", url.split("@")[-1])


def make_key(namespace: str, *parts: Any) -> str:
    """Clé ``puls:<namespace>:<sha256>`` stable pour des ``parts`` JSON-sérialisables."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"puls:{namespace}:{hashlib.sha256(payload.encode()).hexdigest()}"


def fingerprint(text: str) -> str:
    """Empreinte courte d'un texte (ex. un prompt) à inclure dans une clé."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def cache_get(key: str) -> Any:
    setup_cache()
    return await cache.get(key)


async def cache_set(key: str, value: Any, expire: int) -> None:
    setup_cache()
    await cache.set(key, value, expire=expire)


class CachedEmbeddings(Embeddings):
    """Enveloppe des embeddings : seuls les textes jamais vus sont envoyés au modèle.

    C'est le cache le plus rentable : l'index Faiss reste reconstruit à la volée
    pour chaque client, mais les passages déjà vectorisés ne coûtent plus rien.
    Seules les méthodes asynchrones (utilisées par Faiss) passent par le cache.
    """

    def __init__(self, underlying: Embeddings, model: str, ttl: int = TTL_EMBEDDINGS) -> None:
        self.underlying = underlying
        self.model = model
        self.ttl = ttl

    def _key(self, kind: str, text: str) -> str:
        return make_key("embedding", self.model, kind, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.underlying.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.underlying.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        setup_cache()
        keys = [self._key("document", text) for text in texts]
        cached: Sequence[Any] = await cache.get_many(*keys)
        missing = [i for i, vector in enumerate(cached) if vector is None]
        vectors: list[list[float] | None] = list(cached)
        if missing:
            computed = await self.underlying.aembed_documents([texts[i] for i in missing])
            for i, vector in zip(missing, computed, strict=True):
                vectors[i] = vector
            await cache.set_many(
                {keys[i]: vector for i, vector in zip(missing, computed, strict=True)},
                expire=self.ttl,
            )
        hits = len(texts) - len(missing)
        logger.info("Embeddings : %d en cache, %d calculé(s)", hits, len(missing))
        return [vector for vector in vectors if vector is not None]

    async def aembed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        cached = await cache_get(key)
        if cached is not None:
            return list(cached)
        vector = await self.underlying.aembed_query(text)
        await cache_set(key, vector, self.ttl)
        return vector
