# Scénario de démonstration (≈ 5 min)

Public : équipes produit et marketing de Puls-Events. Objectif : montrer que l'assistant
trouve les bons événements, dit quand il n'y en a pas, et s'intègre par une simple API.

## Le RAG en une minute, sans jargon

> Un modèle de langage comme ChatGPT parle très bien, mais il ne connaît pas notre agenda
> et peut inventer. Notre assistant fonctionne comme un bon libraire : avant de répondre,
> il **cherche** dans les 4 300 événements de la métropole ceux qui correspondent à la
> demande (thème, ville, dates), puis il **rédige** une réponse à partir de ces seuls
> événements, en citant chacun avec sa date, son lieu et son lien. S'il ne trouve rien, il
> le dit au lieu d'inventer. Chaque nuit, l'agenda est relu pour rester à jour.

## Préparation (la veille)

```bash
docker pull ghcr.io/willisback/p7-assistant-evenements-culturels:latest   # index embarqué
docker compose up -d                                                       # lit .env (LLM_PROFILE=ci)
uv run python scripts/api_test.py                                          # 7/7 attendu
```

Vérifier dans `.env` : `OPENROUTER_API_KEY`, `OPENAGENDA_API_KEY`, `LLM_PROFILE=ci`. Ouvrir
à l'avance `http://127.0.0.1:8000/docs` et un terminal avec les commandes ci-dessous.

## Déroulé

| Temps | Action | Message |
|---|---|---|
| 0:00 | `docker compose up` (déjà lancé) puis `curl -s localhost:8000/metadata` | « Un seul conteneur, l'agenda de toute la métropole est déjà dedans : 4 294 événements, rafraîchis toutes les 24 h. » |
| 0:45 | **Question 1** (ci-dessous) | « Il comprend la ville et "les 15 prochains jours". Rien à Aix ? Il le dit, et propose l'atelier de Marseille à 28 km. » |
| 2:00 | **Question 2** | « Même chose pour la culture : le théâtre des Salins en décembre. Chaque réponse renvoie à la fiche Open Agenda. » |
| 2:45 | **Question 3** | « Hors sujet : il ne répond pas à côté, il recentre. » |
| 3:15 | Fenêtre de chat (`uv run --script scripts/chat_ui.py`) : « Des événements à Aix dans les 15 prochains jours ? » puis « Je cherche un atelier d'immersion professionnelle. » | « En conversation, il garde le contexte : la deuxième question réutilise Aix et les dates. » |
| 4:15 | Swagger `/docs` | « Pour vos équipes techniques : une API documentée, du JSON exploitable tel quel. » |
| 4:40 | Conclusion | Qualité mesurée à chaque version (gate d'évaluation), pistes : reranking, ajout à l'agenda. |

### Requêtes préparées

```bash
# 1. Lieu + période + élargissement
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Je recherche un atelier d'"'"'immersion professionnelle à Aix-en-Provence dans les 15 prochains jours", "k": 3}' | jq '{answer, sources: [.sources[] | {title, date_range, commune, distance_km}], filters}'

# 2. Culture
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Des spectacles au théâtre des Salins à Martigues en décembre 2026 ?", "k": 3}' | jq '{answer, sources: [.sources[].title]}'

# 3. Hors sujet
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Quel temps fera-t-il à Marseille demain ?"}' | jq .answer
```

Les dates relatives (« 15 prochains jours ») suivent la date du jour : la veille, rejouer
les requêtes et ajuster si besoin (`docs/demo/reponses-de-secours/` donne l'allure attendue).

## Plan B

| Incident | Parade |
|---|---|
| Pas de réseau vers OpenRouter (LLM) | Profil local : `LLM_PROFILE=local` vers le vLLM du réseau domestique, sinon montrer les réponses enregistrées `docs/demo/reponses-de-secours/*.json` (vraies réponses de l'API, 2026-11-08). |
| Pas d'accès à GHCR | Image construite la veille : `docker build -t puls-events-rag .` (l'index local `data/index/` est copié dedans). |
| Docker indisponible | `uv run uvicorn puls_events_rag.api.app:app` sur le poste, même index. |
| Réponse lente (> 10 s) | Montrer la réponse enregistrée correspondante, puis expliquer la latence (deux appels LLM, piste d'amélioration). |
| Plus rien ne marche | Présentation (`docs/presentation/`) : captures et exemples JSON. |

## Questions probables du public

- *« Pourquoi pas juste ChatGPT ? »* : il ne connaît pas notre agenda et invente ; ici,
  chaque recommandation est vérifiable.
- *« Et si un événement est ajouté ce matin ? »* : l'index est rafraîchi toutes les 24 h
  (ou à la demande avec `/rebuild`).
- *« Combien ça coûte ? »* : environ 0,0003 $ par question avec `mistral-small` ; l'évaluation
  complète (40 questions) coûte environ 0,01 $.
- *« Ça marche ailleurs qu'à Marseille ? »* : il suffit de changer d'agenda Open Agenda et
  de référentiel de communes.
