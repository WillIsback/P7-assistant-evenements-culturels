# Livraison — checklist et autoévaluation

Grille officielle : `docs/templates/fiche-autoevaluation.pdf`. Chaque indicateur est relié à
ce qui le démontre dans le dépôt. Les cases « à faire par William » demandent une action
humaine (répétition, revue avec le mentor).

## Livrables

| Livrable | Emplacement |
|---|---|
| Rapport technique (template Puls-Events) | `docs/rapport-technique.md` (+ `docs/rapport-technique.pdf`) |
| Code du système RAG | `src/puls_events_rag/` (`core/`, `rag/`, `api/`, `chat/`) |
| Scripts | `scripts/build_index.py`, `scripts/evaluate.py`, `scripts/api_test.py`, `scripts/perf.py`, `scripts/chat_ui.py` |
| Tests unitaires et d'intégration | `tests/` (145 tests sans réseau, couverture 92 % ; `-m integration`) |
| Jeu de test annoté | `eval/dataset.jsonl` (40 questions) |
| API REST | `src/puls_events_rag/api/` ; Swagger `/docs` |
| Déploiement | `Dockerfile`, `compose.yaml`, image GHCR à chaque release |
| Présentation et démo | `docs/presentation/` (HTML + PDF), `docs/demo.md` |

## Autoévaluation

### Identifier ou créer un modèle adapté aux contraintes et besoins métier

- [x] **Schéma UML de l'architecture** : rapport §2 (composants et séquence `/ask`, Mermaid).
- [x] **Rôle de chaque composant** : rapport §2, tableau « Rôle de chaque composant ».
- [x] **Modèle pertinent (performance, précision)** : rapport §4 ; mesures dans `docs/evaluation/`.
- [x] **Choix justifiés (modèle, architecture, résultats, limites, améliorations)** : rapport §2 à §8.
- [x] **Modèle adapté au métier (prompts, base de connaissance)** : prompts `rag/query.py`, `rag/chain.py`, `chat/agent.py` ; index régional `core/index.py`.
- [x] **Descriptions vectorisées via Mistral** : `core/documents.py`, `core/index.py`, `mistral-embed-2312` (rapport §3).
- [x] **Chaîne LangChain + Faiss** : `rag/chain.py` (LCEL), `core/index.py` (`FAISS`), `chat/agent.py` (`create_agent`).
- [x] **Réponses cohérentes sur le jeu annoté** : `scripts/evaluate.py` → fidélité 0,97, exactitude 0,88 (`docs/evaluation/evaluation.md`).

### Créer les processus de test

- [x] **Tests unitaires récupération, nettoyage, vectorisation, interrogation, génération** : `tests/test_open_agenda.py`, `test_collect.py`, `test_documents.py`, `test_index.py`, `test_sync.py`, `test_rag.py`, `test_chat.py`, `test_api_rag.py`.
- [x] **Tests relançables automatiquement** : `uv run pytest` ; CI `ci-test` à chaque PR ; gate d'évaluation à chaque release.
- [x] **Résultats documentés (logs, résumé lisible)** : `reports/evaluation.md` (joint à chaque GitHub Release), `docs/evaluation/*.md`, sortie de `scripts/api_test.py`.
- [x] **Indicateurs clairs** : hit rate, MRR, taux de refus, fidélité, exactitude ; latences p50/p95.
- [x] **Erreurs fréquentes et limites identifiées** : `docs/evaluation/evaluation.md` (corrections), `scenarios-robustesse.md`, rapport §8.
- [x] **Dépôt lisible** : `src/`, `scripts/`, `tests/`, `eval/`, `docs/`, `data/` (rapport §9).
- [x] **Évaluateur autonome** : `README.md` (installation, index, API, Docker, évaluation).
- [x] **Jeu de données annoté** : `eval/dataset.jsonl`, schéma validé par `tests/test_evaluation.py`.
- [x] **Script comparant réponses obtenues et attendues** : `scripts/evaluate.py`.

### Configurer une API compatible et l'intégrer

- [x] **API REST avec `/ask` et `/rebuild`** : `api/rag_routes.py`.
- [x] **API robuste (erreurs, validation, pagination)** : validation Pydantic (422), 503, 401/403/404/409 ; pagination `offset`/`limit` sur les listes (`/users`, `/chat/sessions`) ; tests `test_api_rag.py`.
- [x] **Script pour tout lancer en local** : `docker compose up` (`compose.yaml`) ou `scripts/build_index.py` + `uvicorn` (README).
- [x] **Données nettoyées et préparées pour l'index** : `core/documents.py` (rapport §3).
- [x] **Endpoints documentés** : Swagger `/docs`, README « Lancer l'API », rapport §6.
- [x] **API testée localement, exemples** : `scripts/api_test.py` (7/7 local et conteneur), `curl` dans le README et `docs/demo.md`.

### Exposer les résultats aux directions métier

- [x] **API accessible, réponses enrichies** : `/ask` renvoie réponse, sources, filtres déduits, latence.
- [x] **Endpoints métier (`/ask`, `/metadata`, `/health`)** : `api/rag_routes.py`, `api/app.py`.
- [x] **Exemples concrets (curl, Python)** : README, `docs/demo.md`, `scripts/api_test.py` (Python).
- [x] **Cas limites (pas de réponse, erreurs de requête)** : réponses « aucun événement », hors zone, demande de précision ; codes d'erreur ; scénarios de robustesse.
- [x] **JSON exploitable** : schémas Pydantic `AskResponse`, `MetadataResponse` (Swagger).
- [ ] **Démonstration fluide avec cas réalistes** : scénario `docs/demo.md` prêt ; *répétition chronométrée à faire par William*.
- [ ] **Choix expliqués à des non-techniciens** : « Le RAG en une minute » (`docs/demo.md`), présentation ; *à répéter par William*.

## Vérification de bout en bout depuis un clone vierge

Rejouée le 2026-11-08 sur un clone neuf (voir la PR de la v0.3.0) :
`uv sync --locked` → `pytest` → `build_index.py` → `docker build` / `docker run` →
`api_test.py` → `evaluate.py`.

## Questions possibles du jury

| Question | Éléments de réponse |
|---|---|
| Pourquoi Faiss plutôt qu'une base vectorielle serveur ? | 5 000 passages : index plat exact en 6 ms, aucun service à opérer, embarqué dans l'image ; au-delà de quelques dizaines de milliers de passages, Qdrant ou pgvector. |
| Pourquoi indexer tout l'agenda plutôt que d'interroger l'API à chaque question ? | Le filtrage sémantique et géographique est meilleur sur l'index (élargissement par rayon, recherche hybride) ; latence et coût maîtrisés ; fraîcheur 24 h suffisante pour un agenda. |
| Comment évitez-vous les hallucinations ? | Prompt « uniquement depuis le contexte », seuil de pertinence, refus explicites ; mesuré : fidélité 0,97. |
| Pourquoi un LLM-juge et pas Ragas ? | Les refus sont des réponses attendues ; juge par catégorie, structuré, peu coûteux ; limite : même modèle que le générateur. |
| Comment le seuil 0,52 a-t-il été choisi ? | Calibré sur le jeu annoté et les scénarios (0,53 → 0,52 améliore toutes les métriques), cf. `docs/evaluation/evaluation.md`. |
| Que se passe-t-il quand l'index a plus de 24 h ? | Il sert immédiatement ; une seule resynchronisation tourne en fond et le remplace d'un bloc. |
| Pourquoi deux fournisseurs LLM ? | vLLM local gratuit pour le développement, `mistral-small` via OpenRouter en CI et en démo ; même adaptateur, choix par `LLM_PROFILE`. |
| Coût en production ? | ≈ 0,0003 $ par question ; embeddings mis en cache, seuls les nouveaux passages sont payés chaque jour. |
| Sécurité ? | Pas de pickle pour l'index, JWT pour le chat, jeton pour `/rebuild`, image scannée (Trivy) et attestée, aucun secret dans le dépôt. |
| Limites principales ? | Agenda à dominante emploi, seuil calibré sur cet agenda, deux appels LLM par question, jeu annoté par une seule personne. |
