# Puls-Events RAG — assistant de recommandation d'événements culturels

POC OpenClassrooms (parcours AI Engineer, projet 7) : un chatbot qui répond aux
questions des utilisateurs en s'appuyant sur les événements récents (moins d'un an)
de l'API [Open Agenda](https://openagenda.com).

- **RAG** : LangChain, modèles Mistral, index vectoriel Faiss.
- **API REST** : FastAPI.
- **Déploiement** : image Docker publiée sur GHCR à chaque release.

> Projet en construction : le rapport technique (architecture, choix, modèles,
> résultats, pistes d'amélioration) sera dans `docs/`.

## Démarrage rapide

```bash
cp .env.example .env    # renseigner les clés
uv sync
uv run uvicorn puls_events_rag.api.app:app --reload
curl http://127.0.0.1:8000/health
```

Avec Docker :

```bash
docker run --rm -p 8000:8000 --env-file .env ghcr.io/willisback/p7-assistant-evenements-culturels:latest
```

## Organisation

| Dossier | Contenu |
|---|---|
| `src/puls_events_rag/` | code du système RAG et de l'API |
| `scripts/` | collecte Open Agenda, reconstruction de l'index, évaluation |
| `tests/` | tests unitaires (sans réseau) et d'intégration |
| `docs/` | rapport technique et présentation |
| `data/` | données brutes et index (générés, non versionnés) |

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) (workflow par lots de version, gates de CI, release).

## Licence

MIT — William Derue.
