"""Index Faiss régional persistant et retrieval filtré (#7).

Tout l'agenda de la région est vectorisé hors ligne (:mod:`puls_events_rag.core.sync`) ;
le filtrage par commune et par période se fait **ici**, sur les métadonnées des
passages, et non dans l'appel à l'API Open Agenda.

Format sur disque (aucun pickle) ::

    <directory>/index.faiss       vecteurs (``faiss.write_index``)
    <directory>/documents.jsonl   passages, dans l'ordre des vecteurs
    <directory>/manifest.json     modèle d'embeddings, dimension, date de synchronisation…

Le filtre est **exhaustif** : la recherche porte sur tous les vecteurs
(``fetch_k = ntotal``) avant filtrage, ce qui reste instantané pour quelques
milliers d'événements (index plat).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from puls_events_rag.core.cities import city_centers, normalize
from puls_events_rag.core.documents import events_to_documents
from puls_events_rag.core.geo import GeoPoint, haversine_km

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
DOCUMENTS_FILE = "documents.jsonl"
MANIFEST_FILE = "manifest.json"
DEFAULT_TOP_K = 5
FALLBACK_RADII_KM = (5.0, 10.0, 20.0, 40.0)
TOP_KEYWORDS = 10
# Hors de la commune demandée, on ne propose que du clairement pertinent : seuil resserré.
EXPANSION_MARGIN = 0.05
# Recherche hybride : la distance L2 est diminuée de LEXICAL_WEIGHT multiplié par la part
# (pondérée par l'IDF) des termes de la requête présents dans le titre ou les mots-clés.
LEXICAL_WEIGHT = 0.15
# Bonus accordé seulement si la majorité (pondérée) des termes est présente : un seul mot
# courant (« atelier ») ne doit pas faire remonter un événement hors sujet.
LEXICAL_MIN_OVERLAP = 0.5
_STOPWORDS = frozenset(
    (
        "avec", "dans", "pour", "sans", "sous", "sur", "des", "les", "une", "aux", "par",
        "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "est", "sont",
        "evenement", "evenements",
    )
)  # fmt: skip


class IndexLoadError(RuntimeError):
    """Index absent, incomplet ou incompatible avec le modèle d'embeddings courant."""


@dataclass(frozen=True)
class IndexManifest:
    """Description d'un index sauvegardé."""

    embedding_model: str
    dimension: int
    event_count: int
    document_count: int
    synced_at: datetime
    data_fingerprint: str

    def is_fresh(self, ttl: timedelta, now: datetime | None = None) -> bool:
        """Vrai tant que l'index a moins de ``ttl`` (24 h par défaut côté synchro)."""
        return (now or datetime.now(tz=UTC)) - self.synced_at < ttl

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "synced_at": self.synced_at.isoformat()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        return cls(**{**data, "synced_at": datetime.fromisoformat(data["synced_at"])})


@dataclass(frozen=True)
class EventFilter:
    """Filtre de métadonnées : communes (OU) et chevauchement avec une période."""

    communes: tuple[str, ...] = ()
    since: datetime | None = None
    until: datetime | None = None


NO_FILTER = EventFilter()


@dataclass(frozen=True)
class RetrievedEvent:
    """Un événement retrouvé : meilleur passage, distance L2 (plus bas = mieux)
    et, en cas d'élargissement, distance en km à la commune demandée."""

    uid: Any
    score: float
    document: Document
    distance_km: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


@dataclass(frozen=True)
class SearchResult:
    """Résultat d'une recherche ; ``radius_km`` est renseigné si la zone a été élargie."""

    events: list[RetrievedEvent]
    radius_km: float | None = None


@dataclass(frozen=True)
class Facets:
    """Vue d'ensemble des événements correspondant à un filtre."""

    event_count: int
    top_keywords: list[tuple[str, int]] = field(default_factory=list)


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _in_period(metadata: Mapping[str, Any], where: EventFilter) -> bool:
    if where.since is None and where.until is None:
        return True
    begin, end = _parse(metadata.get("begin")), _parse(metadata.get("end"))
    if begin is None or end is None:
        return False
    if where.since is not None and end < where.since:
        return False
    return not (where.until is not None and begin > where.until)


def _point(metadata: Mapping[str, Any]) -> GeoPoint | None:
    lat, lng = metadata.get("lat"), metadata.get("lng")
    if isinstance(lat, int | float) and isinstance(lng, int | float):
        return GeoPoint(float(lat), float(lng))
    return None


def _distance_km(metadata: Mapping[str, Any], centers: Sequence[GeoPoint]) -> float | None:
    point = _point(metadata)
    if point is None or not centers:
        return None
    return round(min(haversine_km(point, center) for center in centers), 1)


def query_terms(query: str) -> list[str]:
    """Racines (5 premières lettres) des mots significatifs de la requête."""
    words = normalize(query).split()
    return list(dict.fromkeys(w[:5] for w in words if len(w) >= 4 and w not in _STOPWORDS))


def lexical_bonus(
    terms: Sequence[str], metadata: Mapping[str, Any], idf: Mapping[str, float] | None = None
) -> float:
    """Réduction de distance accordée pour une correspondance lexicale majoritaire."""
    overlap = lexical_overlap(terms, metadata, idf)
    return LEXICAL_WEIGHT * overlap if overlap >= LEXICAL_MIN_OVERLAP else 0.0


def _stems(metadata: Mapping[str, Any]) -> set[str]:
    text = normalize(" ".join([str(metadata.get("title") or ""), *metadata.get("keywords", [])]))
    return {word[:5] for word in text.split()}


def lexical_overlap(
    terms: Sequence[str], metadata: Mapping[str, Any], idf: Mapping[str, float] | None = None
) -> float:
    """Part (pondérée par l'IDF) des racines de la requête présentes dans le titre et les
    mots-clés : « immersion », rare, compte bien plus qu'« atelier », omniprésent."""
    if not terms:
        return 0.0
    weights = {term: (idf or {}).get(term, 1.0) for term in terms}
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    stems = _stems(metadata)
    return sum(weight for term, weight in weights.items() if term in stems) / total
    haystack = normalize(
        " ".join([str(metadata.get("title") or ""), *metadata.get("keywords", [])])
    )
    stems = {word[:5] for word in haystack.split()}
    return sum(term in stems for term in terms) / len(terms)


def fingerprint_events(events: Iterable[Mapping[str, Any]]) -> str:
    """Empreinte des données : ``uid`` et date de mise à jour de chaque événement."""
    pairs = sorted((str(e.get("uid")), str(e.get("updatedAt", ""))) for e in events)
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


class RegionalIndex:
    """Index Faiss de tout l'agenda régional, chargé une fois et partagé."""

    def __init__(self, store: FAISS, manifest: IndexManifest) -> None:
        self._store = store
        self.manifest = manifest
        self._idf: dict[str, float] | None = None

    @property
    def idf(self) -> dict[str, float]:
        """IDF des racines des titres et mots-clés, par événement (calculé une fois)."""
        if self._idf is None:
            frequency: Counter[str] = Counter()
            seen: set[Any] = set()
            for document in self._documents():
                uid = document.metadata["uid"]
                if uid not in seen:
                    seen.add(uid)
                    frequency.update(_stems(document.metadata))
            total = max(len(seen), 1)
            self._idf = {stem: math.log((1 + total) / (1 + df)) for stem, df in frequency.items()}
        return self._idf

    # -- construction et persistance -------------------------------------------------

    @classmethod
    async def build(
        cls,
        events: Sequence[Mapping[str, Any]],
        embeddings: Embeddings,
        *,
        model: str,
        now: datetime | None = None,
    ) -> Self:
        """Vectorise tous les événements (``CachedEmbeddings`` : seuls les passages
        nouveaux sont payés)."""
        documents = events_to_documents(events)
        if not documents:
            raise ValueError("Aucun événement indexable : index non construit")
        store = await FAISS.afrom_documents(documents, embeddings)
        manifest = IndexManifest(
            embedding_model=model,
            dimension=int(store.index.d),
            event_count=len({doc.metadata["uid"] for doc in documents}),
            document_count=len(documents),
            synced_at=now or datetime.now(tz=UTC),
            data_fingerprint=fingerprint_events(events),
        )
        logger.info(
            "Index régional : %d événement(s), %d passage(s)",
            manifest.event_count,
            manifest.document_count,
        )
        return cls(store, manifest)

    def save(self, directory: Path) -> None:
        """Écrit l'index dans un dossier temporaire puis le substitue à ``directory``."""
        directory = Path(directory)
        staging = directory.with_name(directory.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        faiss.write_index(self._store.index, str(staging / INDEX_FILE))
        with (staging / DOCUMENTS_FILE).open("w", encoding="utf-8") as handle:
            for position in range(self._store.index.ntotal):
                doc_id = self._store.index_to_docstore_id[position]
                document = self._document(doc_id)
                record = {
                    "id": doc_id,
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (staging / MANIFEST_FILE).write_text(
            json.dumps(self.manifest.to_json(), indent=2), encoding="utf-8"
        )
        previous = directory.with_name(directory.name + ".old")
        shutil.rmtree(previous, ignore_errors=True)
        if directory.exists():
            directory.rename(previous)
        staging.rename(directory)
        shutil.rmtree(previous, ignore_errors=True)

    @staticmethod
    def read_manifest(directory: Path) -> IndexManifest:
        """Lit le manifeste seul (contrôle de fraîcheur sans charger les vecteurs)."""
        path = Path(directory) / MANIFEST_FILE
        try:
            return IndexManifest.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexLoadError(f"Manifeste d'index illisible : {path}") from exc

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings, *, model: str) -> Self:
        """Recharge un index ; refuse un index construit avec un autre modèle."""
        directory = Path(directory)
        manifest = cls.read_manifest(directory)
        if manifest.embedding_model != model:
            raise IndexLoadError(
                f"Index construit avec {manifest.embedding_model!r}, modèle courant {model!r} :"
                " reconstruire l'index (scripts/build_index.py)"
            )
        try:
            index = faiss.read_index(str(directory / INDEX_FILE))
            with (directory / DOCUMENTS_FILE).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        except (OSError, RuntimeError, ValueError) as exc:
            raise IndexLoadError(f"Index illisible dans {directory}") from exc
        if index.ntotal != len(records) or index.d != manifest.dimension:
            raise IndexLoadError(f"Index incohérent dans {directory} (vecteurs ≠ passages)")
        docstore = InMemoryDocstore(
            {
                r["id"]: Document(page_content=r["page_content"], metadata=r["metadata"])
                for r in records
            }
        )
        mapping = {position: r["id"] for position, r in enumerate(records)}
        store = FAISS(embeddings, index, docstore, mapping)
        return cls(store, manifest)

    # -- interrogation ---------------------------------------------------------------

    def _document(self, doc_id: str) -> Document:
        document = self._store.docstore.search(doc_id)
        if not isinstance(document, Document):  # pragma: no cover - index cohérent
            raise IndexLoadError(f"Passage {doc_id} absent du docstore")
        return document

    def _documents(self) -> Iterable[Document]:
        for position in range(self._store.index.ntotal):
            yield self._document(self._store.index_to_docstore_id[position])

    async def _ranked(
        self,
        query: str,
        keep: Callable[[dict[str, Any]], bool],
        k: int,
        max_distance: float | None = None,
    ) -> list[tuple[Document, float]]:
        total = int(self._store.index.ntotal)
        vector = await self._store.embeddings.aembed_query(query)  # type: ignore[union-attr]
        hits = await self._store.asimilarity_search_with_score_by_vector(
            vector, k=total, fetch_k=total, filter=keep
        )
        terms = query_terms(query)
        idf = self.idf
        adjusted = sorted(
            (
                (
                    document,
                    float(score) - LEXICAL_WEIGHT * lexical_overlap(terms, document.metadata, idf),
                )
                for document, score in hits
            ),
            key=lambda pair: pair[1],
        )
        best: dict[Any, tuple[Document, float]] = {}
        for document, score in adjusted:
            if max_distance is not None and score > max_distance:
                break  # résultats triés : tous les suivants sont moins pertinents
            best.setdefault(document.metadata["uid"], (document, float(score)))
            if len(best) == k:
                break
        return list(best.values())

    async def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_TOP_K,
        where: EventFilter = NO_FILTER,
        expand: bool = True,
        max_distance: float | None = None,
    ) -> SearchResult:
        """Au plus ``k`` événements distincts, du plus au moins pertinent.

        Si les communes demandées n'ont aucun résultat et que ``expand`` est vrai,
        la zone est élargie par rayons successifs (:data:`FALLBACK_RADII_KM`) autour
        de leur centre ; chaque événement porte alors ``distance_km``.
        ``max_distance`` écarte les passages trop éloignés de la requête (distance
        L2 des embeddings) : « rien de pertinent à Aix » déclenche aussi l'élargissement.
        """
        if not query.strip() or k < 1:
            return SearchResult([])
        communes = set(where.communes)

        def in_communes(metadata: dict[str, Any]) -> bool:
            return (not communes or metadata.get("commune") in communes) and _in_period(
                metadata, where
            )

        hits = await self._ranked(query, in_communes, k, max_distance)
        centers = city_centers(list(where.communes))
        if hits or not communes or not expand or not centers:
            return SearchResult([RetrievedEvent(d.metadata["uid"], s, d) for d, s in hits])

        for radius in FALLBACK_RADII_KM:

            def nearby(metadata: dict[str, Any], radius: float = radius) -> bool:
                distance = _distance_km(metadata, centers)
                return distance is not None and distance <= radius and _in_period(metadata, where)

            strict = None if max_distance is None else max_distance - EXPANSION_MARGIN
            hits = await self._ranked(query, nearby, k, strict)
            if hits:
                logger.info("Aucun résultat à %s : élargissement à %g km", communes, radius)
                events = [
                    RetrievedEvent(d.metadata["uid"], s, d, _distance_km(d.metadata, centers))
                    for d, s in hits
                ]
                return SearchResult(events, radius_km=radius)
        return SearchResult([])

    def facets(self, where: EventFilter = NO_FILTER) -> Facets:
        """Nombre d'événements et mots-clés dominants pour un filtre (sans requête)."""
        communes = set(where.communes)
        seen: set[Any] = set()
        keywords: Counter[str] = Counter()
        for document in self._documents():
            metadata = document.metadata
            uid = metadata["uid"]
            if uid in seen:
                continue
            if communes and metadata.get("commune") not in communes:
                continue
            if not _in_period(metadata, where):
                continue
            seen.add(uid)
            keywords.update(metadata.get("keywords") or [])
        return Facets(len(seen), keywords.most_common(TOP_KEYWORDS))
