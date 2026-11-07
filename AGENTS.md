# AGENTS.md — Puls-Events RAG (OpenClassrooms AI Engineer, projet 7)

POC d'assistant qui recommande des événements culturels : RAG (LangChain + Mistral + Faiss)
sur les données Open Agenda, exposé par une API REST FastAPI, conteneurisé avec Docker.

## Workflow (obligatoire)

Skill `semver-batch-workflow` (https://github.com/WillIsback/agent-skills). Résumé :

- Une issue typée par changement (`.github/ISSUE_TEMPLATE`).
- Lots de version traités dans l'ordre **patch → minor → major**.
- Branche de lot `<type>/<version>` (ex. `fix/0.2.1`) ; une sous-branche par issue
  `<type>/<version>-<tâche>`, fusionnée dans la branche de lot par
  `<type>: merge <type>/<version>-<tâche> (#N)`.
- **Jamais de commit ni de merge direct sur `main`** : une PR par lot, avec
  `Closes #N` pour chaque issue.
- Conventional Commits ; `feat` → minor, `fix`/`perf` → patch, `!` → major.
- Preflight uniquement si la CI de PR ne couvre pas un job de la release.
- Toujours minimiser le temps de CI/CD : ne rien construire deux fois, cacher,
  filtrer par chemins, garder le lourd hors PR.

## Fournisseurs LLM

Un seul adaptateur compatible OpenAI (`langchain-openai`, `base_url`), choisi par
`LLM_PROFILE` (voir `.env.example`). Modèles toujours **sans réflexion** :

| Profil | Chat | Désactivation de la réflexion |
|---|---|---|
| `local` | vLLM `http://192.168.1.87:30000/v1` (`qwen3.8-flash-next-a5b`) | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` |
| `ci` | OpenRouter `mistralai/mistral-small-2603` | `extra_body={"reasoning": {"enabled": False}}` |

Embeddings : `mistralai/mistral-embed-2312` via OpenRouter **dans tous les profils**
(le vLLM n'expose pas `/v1/embeddings` ; un index Faiss n'est valide qu'avec le
modèle qui l'a construit). Pas de `openai/gpt-oss-120b` (réflexion obligatoire)
ni de `openrouter/free` pour l'évaluation (modèle non reproductible).

## Commandes

```bash
uv sync                                        # dépendances
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -m "not integration"             # tests unitaires, sans réseau
uv run uvicorn puls_events_rag.api.app:app --reload
docker build -t puls-events-rag . && docker run --rm -p 8000:8000 --env-file .env puls-events-rag
```

## Conventions

- Code dans `src/puls_events_rag/`, scripts dans `scripts/`, tests dans `tests/`,
  documentation dans `docs/`.
- Les tests unitaires n'appellent jamais de réseau (LLM, embeddings, Open Agenda) :
  simuler. Les tests réels sont marqués `@pytest.mark.integration`.
- `scripts/evaluate.py` (à venir) : contrat de la gate `gate-evaluation-rag` de la
  release — écrit `reports/evaluation.md`, sort en erreur sous les seuils.
- Explorer le code avec CodeGraph (`codegraph_explore`) avant grep/lecture.
- Textes, docs, commits et PR en français.
