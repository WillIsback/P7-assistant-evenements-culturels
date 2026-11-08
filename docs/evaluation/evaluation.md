# Évaluation de la chaîne RAG (#13, #14)

## Jeu de test annoté — `eval/dataset.jsonl`

40 questions en six catégories :

| Catégorie | Nombre | Attendu |
|---|---|---|
| `theme` | 11 | proposer un événement du thème (« install party », « métiers du BTP »…) |
| `lieu` | 9 | thème + commune (« job dating à Martigues »…) |
| `date` | 9 | dates **absolues** (« le 23 mai 2026 », « en décembre 2026 ») |
| `sans_reponse` | 4 | rien dans l'agenda (« rugby à Gardanne ») : le dire, sans inventer |
| `hors_sujet` | 5 | météo, culture générale, code, injection de prompt : refuser |
| `hors_zone` | 2 | Lyon, Paris : signaler la zone non couverte |

**Méthode d'annotation.** Chaque question est rédigée à la main à partir des données
collectées (`data/raw/events.jsonl`, 4 294 événements). Les événements attendus
(`expected_uids`) sont repérés par des critères explicites sur ces données (titre,
mots-clés, commune `adminLevel4`, chevauchement de dates), puis relus. Pour les questions
sans réponse, on vérifie qu'aucun événement ne satisfait les critères. La référence
(`reference`) cite jusqu'à trois événements attendus. Seules des dates absolues sont
utilisées : une date relative (« ce mois-ci ») changerait de sens d'une release à l'autre.
Le schéma est validé en CI (`tests/test_evaluation.py`).

## Métriques — `scripts/evaluate.py`

| Métrique | Calcul | Seuil par défaut |
|---|---|---|
| `hit_rate` | part des questions avec réponse dont un événement attendu est dans les sources | 0,70 |
| `mrr` | moyenne de 1/rang du premier événement attendu | 0,50 |
| `refusal_rate` | part des questions `sans_reponse` / `hors_sujet` refusées (verdict du juge) | 0,80 |
| `faithfulness` | réponses qui n'affirment rien d'absent du contexte fourni (LLM-juge) | 0,85 |
| `correctness` | réponses conformes à l'attendu de leur catégorie (LLM-juge) | 0,70 |

Seuils configurables : `--min-hit-rate 0.8` ou `EVAL_MIN_HIT_RATE=0.8`. Le script écrit
`reports/evaluation.md` (joint à la GitHub Release) et sort en erreur sous un seuil : c'est
la gate `gate-evaluation-rag` de la release (profil `ci`, OpenRouter, sans réflexion).

**Choix du juge.** Un LLM-juge maison par sortie structurée (`Judgement` : fidèle,
correct, justification) plutôt que Ragas ou DeepEval :
- les métriques de Ragas (faithfulness, answer relevancy…) supposent une réponse qui
  reprend le contexte ; ici, un refus ou une demande de précision peut être la bonne
  réponse, ce que le juge évalue selon la catégorie attendue ;
- aucune dépendance lourde de plus dans l'image ;
- coût et durée maîtrisés (voir ci-dessous).

Limite : le juge est le même modèle que le générateur (`mistral-small-2603`). Un juge plus
puissant ou différent réduirait le biais d'auto-évaluation (piste d'amélioration).

## Résultats

2026-11-08, profil `ci`, index du jour :

| Métrique | Valeur | Seuil |
|---|---|---|
| hit_rate | 0,83 | 0,70 |
| mrr | 0,74 | 0,50 |
| refusal_rate | 1,00 | 0,80 |
| faithfulness | 0,97 | 0,85 |
| correctness | 0,88 | 0,70 |

Durée ≈ 30 s (4 questions en parallèle), environ 90 000 jetons, **coût estimé ≈ 0,011 $**
(objectif : < 5 min, < 0,05 $).

## Ce que l'évaluation a permis de corriger

| Version | hit_rate | mrr | refus | fidélité | exactitude | Correction |
|---|---|---|---|---|---|---|
| Première mesure | 0,48 | 0,44 | 0,89* | 0,82* | 0,80* | — |
| + ancrage du parseur | 0,86 | 0,77 | 0,67 | 0,90 | 0,90 | communes et périodes inventées par le LLM supprimées si absentes de la question |
| + recherche hybride | 0,90 | 0,83 | 1,00 | 0,95 | 0,97 | bonus lexical, seuil resserré à l'élargissement, juge plus strict |
| + IDF, majorité, seuil 0,52 | 0,83 | 0,74 | 1,00 | 0,97 | 0,88 | bonus pondéré par la rareté des termes, réservé aux correspondances majoritaires ; scénario Aix → Marseille rétabli |

\* juge initial trop indulgent (il validait des refus là où un événement était attendu) : les
valeurs ne sont pas comparables.

## Preflight des lots qui touchent la chaîne RAG

La CI de PR ne lance pas de vrai LLM. Avant d'ouvrir la PR d'un lot qui modifie `rag/`,
`core/index.py`, `chat/` ou le jeu annoté, rejouer en local et reporter le résultat dans la
section « Preflight » de la PR :

```bash
LLM_PROFILE=ci uv run python scripts/evaluate.py --output reports/evaluation.md
uv run pytest tests/test_scenarios.py -m integration --no-cov
```
