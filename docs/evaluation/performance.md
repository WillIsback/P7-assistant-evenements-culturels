# Performance du système (#12)

Mesures du 2026-11-08 (`uv run python scripts/perf.py --base-url …`), poste de
développement, index régional de 4 294 événements / 4 854 passages (mistral-embed,
dimension 1 024, Faiss plat).

| Mesure | Résultat | Commentaire |
|---|---|---|
| Construction complète de l'index | ≈ 3 min | collecte (15 pages) + embeddings OpenRouter ; resynchro quotidienne bien plus rapide grâce au cache d'embeddings |
| Taille de l'index | 26 Mo | embarqué dans l'image (534 Mo au total) |
| Chargement de l'index | 134 ms | au démarrage de l'API, une seule fois |
| Recherche filtrée exhaustive (k = 5) | p50 5,9 ms, p95 6,3 ms | hors embedding de la requête (≈ 0,2-0,5 s via OpenRouter, mis en cache) |
| `POST /ask` de bout en bout | p50 5,0 s, p95 6,6 s | profil `local` (vLLM) : extraction de la requête + génération ; le LLM domine |

En CI de PR, `tests/test_perf.py` vérifie sans réseau, sur 5 000 événements factices de
dimension réelle, que le chargement reste sous 3 s et la recherche filtrée sous 0,5 s
(≈ 1 s ajoutée à la CI).

Pistes : streaming de `/ask` (déjà en place côté chat), parseur et génération en un seul
appel LLM, modèle plus petit pour l'extraction de la requête.
