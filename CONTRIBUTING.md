# Contribuer

## Flux de travail

Le dépôt suit le workflow « lots de version semver »
([skill `semver-batch-workflow`](https://github.com/WillIsback/agent-skills)) :

1. Une **issue** par changement (modèles : bug, feature, refactor, docs, chore).
2. Les issues sont regroupées en **lots**, un lot = la prochaine version, traités
   dans l'ordre **patch → minor → major**.
3. Branche de lot `<type>/<version>` (ex. `feat/0.2.0`) créée depuis `main`.
4. Chaque issue sur une sous-branche `<type>/<version>-<tâche>`, fusionnée dans la
   branche de lot par `<type>: merge <type>/<version>-<tâche> (#N)`.
5. Lot complet (et preflight si nécessaire) → **une PR vers `main`** suivant le
   [modèle de PR](.github/PULL_REQUEST_TEMPLATE.md). Jamais de merge direct sur `main`.
6. Gates vertes → merge → release automatique avec le bump approprié.

Titre de PR et commits au format **Conventional Commits**. Seuls `feat` (minor),
`fix`/`perf` (patch) et `!` (major) déclenchent une release.

## Gates de CI (PR vers `main`)

Chaque gate ne s'exécute que si les fichiers qui la concernent changent (un job
ignoré compte comme réussi pour la protection de branche) ; caches uv et Docker.

| Workflow | Vérifie |
|---|---|
| `ci-code-quality` | ruff (lint, format), mypy strict, lockfile uv ; titre de PR, actionlint, zizmor |
| `ci-test` | pytest sans réseau + couverture ≥ 80 % ; build de l'image Docker si Dockerfile ou dépendances changent |
| `ci-security` | Dependency Review (licences), registre d'exceptions, Trivy (uv.lock), Gitleaks |
| `ci-codeql` | CodeQL `security-extended` Python et Actions, si modifiés |
| `ci-traceability` | issue référencée, type de PR, cohérence semver |
| Aikido | app GitHub Aikido (SAST, secrets, dépendances) |

## Release (push sur `main`)

| Étape | Rôle |
|---|---|
| `detect` | bump semver depuis les commits depuis le dernier tag |
| `gate-version-consistency` | tag inexistant et supérieur au précédent |
| `gate-evaluation-rag` | évaluation sur le jeu de test annoté (OpenRouter), rapport joint |
| `image` | image Docker construite **une fois**, scannée (Trivy), publiée sur GHCR, attestée |
| `publish` | tag `vX.Y.Z` + GitHub Release (changelog, rapport d'évaluation) |

Lint, tests, CodeQL et SCA ne sont pas rejoués : ils ont tourné sur la PR de lot.

### Preflight

Nécessaire seulement si un job de la release n'est pas couvert par la CI de PR.
Aujourd'hui : l'évaluation RAG (appels LLM réels). Avant d'ouvrir la PR d'un lot
qui touche la chaîne RAG, la rejouer localement :

```bash
uv run python scripts/evaluate.py --output reports/evaluation.md
```

## Vérifications locales

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -m "not integration"
docker build -t puls-events-rag .
```

## Secrets

- Local : `.env` (jamais commité, modèle `.env.example`).
- CI : secret de dépôt `OPENROUTER_API_KEY` ; `RELEASE_TOKEN` optionnel (voir
  `release.yml`).

## Exceptions de sécurité

Une alerte acceptée temporairement est inscrite dans `.github/security-exceptions.yml`
(justification, auteur, date d'expiration) et reportée dans `.trivyignore`. Une
exception expirée fait échouer la CI.
