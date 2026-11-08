# Puls-Events RAG — assistant de recommandation d'événements culturels

POC OpenClassrooms (parcours AI Engineer, projet 7) : un chatbot qui répond aux
questions des utilisateurs en s'appuyant sur les événements récents (moins d'un an)
de l'API [Open Agenda](https://openagenda.com).

- **RAG** : LangChain, modèles Mistral, index vectoriel Faiss.
- **API REST** : FastAPI.
- **Déploiement** : image Docker publiée sur GHCR à chaque release.

> Projet en construction : le rapport technique (architecture, choix, modèles,
> résultats, pistes d'amélioration) sera dans `docs/`.

## Objectifs

- Collecter les événements culturels récents d'une zone géographique (API Open Agenda).
- Les nettoyer, les découper et les vectoriser dans un index Faiss.
- Répondre en langage naturel aux questions des utilisateurs (RAG LangChain), avec les
  événements sources cités.
- Exposer le système par une API REST et le livrer dans une image Docker.

## Reproduire l'environnement

### Prérequis

- **Python 3.11** (fixé par `.python-version` et `requires-python`).
- **[uv](https://docs.astral.sh/uv/)** ≥ 0.13 (recommandé), ou `pip` + `venv`.
- Une clé [OpenRouter](https://openrouter.ai) (embeddings `mistral-embed`, et chat en
  profil `ci`) ; un serveur vLLM pour le profil `local` (facultatif).
- Docker (facultatif, pour l'image).

### Installation avec uv (source de vérité : `pyproject.toml` + `uv.lock`)

```bash
git clone https://github.com/WillIsback/P7-assistant-evenements-culturels.git
cd P7-assistant-evenements-culturels
uv sync --locked          # crée .venv (non versionné) avec les versions exactes du lock
```

### Installation avec pip

`requirements.txt` est exporté de `uv.lock` (versions identiques, la CI vérifie qu'il
est à jour) :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # dépendances + projet (mode éditable)
pip install pytest pytest-cov            # pour lancer les tests
```

### Secrets

Les clés d'API ne sont jamais versionnées : elles vivent dans `.env` (ignoré par Git).

```bash
cp .env.example .env      # puis renseigner OPENROUTER_API_KEY (et VLLM_*)
```

`LLM_PROFILE` choisit le modèle de chat (`local` : vLLM, `ci` : OpenRouter) ; les
embeddings utilisent `mistralai/mistral-embed-2312` dans tous les profils.

### Vérifier l'installation

```bash
uv run pytest tests/test_imports.py --no-cov   # Faiss, LangChain, clients LLM, FastAPI
uv run pytest -m "not integration"            # tous les tests unitaires (sans réseau)
```

Avec pip, mêmes commandes sans `uv run` (venv activé).

Installation « propre » vérifiée le 07/11/2026 (copie du dépôt sans `.venv`) :
`pip install --no-cache-dir -r requirements.txt` dans un venv Python 3.11 vierge (≈ 30 s)
et `uv sync --locked` avec un cache uv vide, suivis des tests qui passent.

## Reconstruire l'index régional

Tout l'agenda Open Agenda de la métropole Aix-Marseille-Provence (agenda `21769447`,
événements des 365 derniers jours et à venir) est collecté, découpé en passages et
vectorisé (`mistral-embed`) dans un index Faiss :

```bash
uv run python scripts/build_index.py                 # collecte complète puis indexation
uv run python scripts/build_index.py --skip-collect  # réindexe data/raw/events.jsonl
```

| Sortie | Contenu |
|---|---|
| `data/raw/events.jsonl` | événements bruts collectés (un par ligne) |
| `data/index/` | `index.faiss`, `documents.jsonl` (passages et métadonnées), `manifest.json` |

Le manifeste note le modèle d'embeddings (un index construit avec un autre modèle est
refusé au chargement) et la date de synchronisation : l'index est valide
`INDEX_TTL_HOURS` heures (24 par défaut), puis l'API le resynchronise en tâche de fond.
Les embeddings sont mis en cache (`data/cache/`) : une reconstruction quotidienne ne
paie que les passages nouveaux ou modifiés. Le filtrage par commune et par période se
fait au retrieval, sur les métadonnées des passages, avec élargissement par rayons
(5, 10, 20, 40 km) quand la commune demandée n'a aucun résultat.

## Lancer l'API

```bash
uv run uvicorn puls_events_rag.api.app:app --reload
curl http://127.0.0.1:8000/health      # état, version, profil LLM, index chargé et fraîcheur
```

Documentation interactive : `http://127.0.0.1:8000/docs`. L'index régional est chargé
une fois au démarrage (`data/index`, voir ci-dessus) ; s'il a plus de 24 h, il continue
de servir pendant qu'une resynchronisation tourne en tâche de fond.

**Poser une question** (`POST /ask`) :

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Un atelier d'"'"'immersion professionnelle à Aix-en-Provence dans les 15 prochains jours ?", "k": 3}'
```

La réponse contient `answer` (texte généré à partir des seuls événements retrouvés),
`sources` (titre, dates, lieu, commune, URL, distance si la zone a été élargie),
`filters` (thème, communes, période et rayon d'élargissement déduits de la question) et
`latency_ms`. Codes : `422` question vide ou trop longue (1 000 caractères), `503` tant
qu'aucun index n'est disponible.

**Reconstruire l'index** (`POST /rebuild`, jeton `REBUILD_TOKEN` de `.env`) : lance la
collecte et l'indexation en tâche de fond (`202`), `409` si une reconstruction est déjà
en cours ; `GET /rebuild` donne l'état de l'index. Sans `REBUILD_TOKEN`, la route est
désactivée (`404`).

```bash
curl -s -X POST http://127.0.0.1:8000/rebuild -H "Authorization: Bearer $REBUILD_TOKEN"
curl -s http://127.0.0.1:8000/rebuild -H "Authorization: Bearer $REBUILD_TOKEN"
```

Avec Docker :

```bash
docker run --rm -p 8000:8000 --env-file .env ghcr.io/willisback/p7-assistant-evenements-culturels:latest
```

## Organisation

| Chemin | Contenu |
|---|---|
| `src/puls_events_rag/` | code du système RAG et de l'API (`api/` : application FastAPI) |
| `scripts/` | collecte Open Agenda, reconstruction de l'index, évaluation |
| `tests/` | tests unitaires (sans réseau) et d'intégration (`-m integration`) |
| `docs/` | rapport technique et présentation |
| `data/` | données brutes et index (générés, non versionnés) |
| `pyproject.toml`, `uv.lock` | dépendances (source de vérité) |
| `requirements.txt` | export pip de `uv.lock` |
| `.env.example` | modèle de configuration (copier en `.env`) |
| `Dockerfile` | image de l'API |

### Mettre à jour les dépendances

```bash
uv add <paquet>                                              # met à jour pyproject.toml et uv.lock
uv export --no-hashes --no-dev --frozen -o requirements.txt  # puis régénérer requirements.txt
```

### Bibliothèques clés

| Bibliothèque | Rôle |
|---|---|
| `faiss-cpu` | index vectoriel (version CPU, portable) |
| `langchain-community` | intégration `FAISS` de LangChain |
| `langchain-openai` | chat et embeddings via une API compatible OpenAI (vLLM, OpenRouter) |
| `fastapi`, `uvicorn` | API REST |

`langchain-community` n'est plus maintenu activement par LangChain, mais c'est toujours
lui qui fournit l'intégration FAISS officielle (aucun paquet autonome officiel
n'existe). Une `DeprecationWarning` s'affiche donc à l'import.

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) (workflow par lots de version, gates de CI, release).

## Licence

MIT — William Derue.
