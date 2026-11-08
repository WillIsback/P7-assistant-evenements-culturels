# Index régional (lot feat/0.1.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** collecter tout l'agenda régional, le transformer en documents filtrables et le
servir depuis un index Faiss persistant, valide 24 h, avec un retrieval filtré exhaustif.

**Architecture:** `OpenAgendaClient.iter_events` pagine l'API (curseur `after`) ;
`core/documents.py` produit des `Document` aux métadonnées normalisées ; `core/index.py`
construit, sauvegarde (sans pickle), charge et interroge l'index (`RegionalIndex`) ;
`core/sync.py` orchestre collecte → index et la fraîcheur (`IndexManager`) ;
`scripts/build_index.py` est la commande hors ligne.

**Tech Stack:** Python 3.11, httpx, LangChain `FAISS` (`langchain-community`), faiss-cpu,
`CachedEmbeddings` existant, pytest + `httpx.MockTransport` + embeddings factices.

**Spec:** `docs/superpowers/specs/2026-11-08-chat-multitour-design.md` (section 0) ;
issues #5, #6, #7, #8, #27.

## Global Constraints

- Agenda `21769447` ; fenêtre : `timings[gte]` = aujourd'hui (UTC, minuit) − 365 jours.
- Pagination : `size=300` (`MAX_PAGE_SIZE`), curseur `after[]` renvoyé par l'API, arrêt
  quand `events` est vide ou `after` absent.
- Index valide `INDEX_TTL_HOURS` heures (défaut 24) après `synced_at` du manifeste.
- Répertoire d'index `INDEX_DIR` (défaut `./data/index`) ; données brutes
  `data/raw/events.jsonl`.
- Refus de charger un index dont `embedding_model` diffère du modèle courant.
- Pas de pickle : `faiss.write_index` + `documents.jsonl` + `manifest.json`.
- Tests unitaires sans réseau ; textes et commits en français ; ruff, mypy strict.

## Review Focus

1. Ville citée sans aucun événement → élargissement par rayons, résultats triés par
   distance, `distance_km` renseigné (test `test_search_elargit_par_rayon`).
2. Événement sans coordonnées ou hors référentiel → indexé, filtrable par période,
   jamais retenu par un filtre de commune ni par l'élargissement (test
   `test_evenement_sans_coordonnees_ignore_par_filtre_geo`).
3. Événement multi-dates chevauchant la période sans y commencer → retenu
   (test `test_filtre_periode_par_chevauchement`).
4. Index absent, corrompu ou d'un autre modèle → erreur explicite, pas d'index vide
   silencieux (test `test_chargement_refuse_autre_modele`).
5. Erreur HTTP 429/5xx au milieu de la pagination → réessai borné puis `OpenAgendaError`
   (test `test_pagination_reessaie_puis_echoue`).

---

### Task 1: Pagination complète (#5)

**Files:** Modify `src/puls_events_rag/core/providers/open_agenda.py` ; Test `tests/test_open_agenda.py`

**Interfaces — Produces:** `async def iter_events(self, *, since: date | datetime | None = None, detailed: bool = True, page_size: int = MAX_PAGE_SIZE, max_retries: int = 3) -> AsyncIterator[list[dict[str, Any]]]` (une page par itération, sans cache).

- [ ] Tests : `test_pagination_suit_le_curseur` (3 pages simulées, 2ᵉ requête porte `after[]` de la 1ʳᵉ, 5 événements au total) ; `test_pagination_reessaie_puis_echoue` (429 ×4 → `OpenAgendaError`, 4 appels) ; `test_pagination_reessaie_puis_reussit` (429 puis 200).
- [ ] Implémenter (attente exponentielle injectable, `asyncio.sleep` remplacé en test).
- [ ] ruff + mypy + pytest verts ; commit `feat(data): pagination complète de l'agenda Open Agenda (#5)`.

### Task 2: Collecte vers `data/raw` (#5)

**Files:** Create `src/puls_events_rag/core/collect.py` ; Test `tests/test_collect.py`

**Interfaces — Produces:** `async def collect_events(client: OpenAgendaClient, *, since: datetime | None = None) -> list[dict[str, Any]]` (dédoublonné par `uid`, événements dont `lastTiming.end` < `since` retirés) ; `def write_jsonl(events, path: Path) -> None` ; `def read_jsonl(path: Path) -> list[dict[str, Any]]`.

- [ ] Tests : doublons d'`uid` retirés ; événement terminé il y a 400 jours retiré (critère #5 « tous < 12 mois ») ; aller-retour JSONL.
- [ ] Implémenter ; commit `feat(data): collecte de l'agenda régional en JSONL (#5)`.

### Task 3: Métadonnées de filtrage (#6)

**Files:** Modify `src/puls_events_rag/core/semantic_search.py` → déplacer la conversion dans Create `src/puls_events_rag/core/documents.py` (ré-export conservé) ; Test `tests/test_documents.py`

**Interfaces — Produces:** métadonnées de chaque `Document` : `uid`, `title`, `url`, `location`, `city`, `commune: str | None` (nom officiel du référentiel depuis `location.adminLevel4`, sinon `city`, normalisé via `cities.resolve_commune(name) -> str | None`), `lat: float | None`, `lng: float | None`, `begin`, `end` (ISO), `date_range`, `keywords: list[str]`, `chunk`. Plus de `distance_km` dans les documents.

- [ ] Tests : `adminLevel4="Aix-en-Provence"` → `commune == "Aix-en-Provence"` ; commune inconnue → `None` ; `lat`/`lng` repris ; HTML retiré (`<p>` → texte) ; doublons ignorés.
- [ ] Ajouter `resolve_commune` dans `core/cities.py` (nom ou alias normalisé → nom officiel).
- [ ] Implémenter ; commit `feat(data): métadonnées de commune, coordonnées et dates pour le filtrage (#6)`.

### Task 4: Index persistant et manifeste (#7)

**Files:** Create `src/puls_events_rag/core/index.py` ; Test `tests/test_index.py`

**Interfaces — Produces:**
- `@dataclass(frozen=True) class IndexManifest: embedding_model: str; dimension: int; event_count: int; document_count: int; synced_at: datetime; data_fingerprint: str` + `is_fresh(self, ttl: timedelta, now: datetime | None = None) -> bool`.
- `class RegionalIndex` : `@classmethod async def build(cls, events, embeddings, *, model: str) -> RegionalIndex` ; `def save(self, directory: Path) -> None` ; `@classmethod def load(cls, directory: Path, embeddings, *, model: str) -> RegionalIndex` (lève `IndexLoadError`) ; `manifest: IndexManifest`.

- [ ] Tests : build → save → load conserve documents et manifeste ; `test_chargement_refuse_autre_modele` ; répertoire absent → `IndexLoadError` ; `is_fresh` vrai à 23 h, faux à 25 h.
- [ ] Implémenter (FAISS LangChain reconstruit depuis `faiss.read_index` + `InMemoryDocstore`, pas de pickle) ; commit `feat(index): index Faiss régional persistant avec manifeste (#7)`.

### Task 5: Retrieval filtré exhaustif (#7, prépare #9 et #30)

**Files:** Modify `src/puls_events_rag/core/index.py` ; Test `tests/test_index.py`

**Interfaces — Produces:** `@dataclass(frozen=True) class EventFilter: communes: tuple[str, ...] = (); since: datetime | None = None; until: datetime | None = None` ; `async def search(self, query: str, *, k: int = 5, where: EventFilter = EventFilter(), expand: bool = True) -> SearchResult` où `SearchResult(events: list[RetrievedEvent], radius_km: float | None)` (`radius_km` renseigné si élargissement) ; `def facets(self, where: EventFilter) -> Facets(event_count: int, top_keywords: list[tuple[str, int]])`.

- [ ] Tests : `test_filtre_commune_exhaustif` (bon événement classé au-delà des 4·k premiers voisins quand même retrouvé) ; `test_filtre_periode_par_chevauchement` ; `test_search_elargit_par_rayon` ; `test_evenement_sans_coordonnees_ignore_par_filtre_geo` ; `facets` compte et classe les mots-clés.
- [ ] Implémenter : `fetch_k = index.ntotal` + filtre callable LangChain, meilleur passage par `uid`, rayons `FALLBACK_RADII_KM` (5, 10, 20, 40 km) sur `lat`/`lng`.
- [ ] Commit `feat(index): retrieval filtré par commune et période, élargissement géographique (#7)`.

### Task 6: Synchronisation 24 h et commande hors ligne (#7)

**Files:** Create `src/puls_events_rag/core/sync.py`, `scripts/build_index.py` ; Test `tests/test_sync.py`

**Interfaces — Produces:** `async def sync_index(*, client, embeddings, model: str, directory: Path, raw_path: Path | None, skip_collect: bool = False) -> RegionalIndex` ; `class IndexManager(directory, ttl, builder)` : `current: RegionalIndex | None`, `load()`, `async ensure_fresh() -> RegionalIndex` (frais → tel quel ; expiré → tâche de fond unique, échange atomique ; absent → lance la synchro et lève `IndexUnavailableError`), `async refresh() -> RegionalIndex` (verrou : `RefreshInProgressError` si déjà en cours).

- [ ] Tests : index frais non reconstruit ; expiré → une seule tâche lancée pour deux appels, ancien index servi puis remplacé ; absent → `IndexUnavailableError` ; `refresh` concurrent → `RefreshInProgressError`.
- [ ] Script : `uv run python scripts/build_index.py [--skip-collect]` affiche nombre d'événements, de passages, durée.
- [ ] `.env.example` : `INDEX_DIR`, `INDEX_TTL_HOURS` ; README : section « Reconstruire l'index ».
- [ ] Retirer `EventIndex`/`retrieve_events`/`find_events` devenus inutiles et leurs tests ; adapter `main.py` à `RegionalIndex`.
- [ ] Commit `feat(index): synchronisation de l'index (fraîcheur 24 h) et build_index.py (#7)`.

### Task 7: Tests d'indexation et intégration (#8)

- [ ] Test d'intégration `@pytest.mark.integration` : `sync_index` réel sur 1 page (`page_size` réduit), index chargé, une recherche « Aix-en-Provence » renvoie ≥ 1 événement.
- [ ] Couverture ≥ 80 % ; commit `test(index): …` (#8).

### Task 8: Correctifs Debian dans l'image (#27)

**Files:** Modify `Dockerfile` (étage final) ; sous-branche `fix/0.1.0-trivy`.

- [ ] `apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*` ; volume `/app/cache` ; `docker build` puis `trivy image --ignore-unfixed --severity HIGH,CRITICAL` si disponible.
- [ ] Commit `fix(docker): appliquer les correctifs de sécurité Debian dans l'image (#27)`.
