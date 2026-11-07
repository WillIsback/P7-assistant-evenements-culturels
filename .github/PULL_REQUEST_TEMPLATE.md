<!--
  PR de lot vers main : titre Conventional Commits, toutes les issues du lot référencées.
  Workflow : skill semver-batch-workflow (voir AGENTS.md).
-->

# Pull Request

## Lot

- Branche de lot : `<type>/<version>` (ex. `fix/0.2.1`)
- Version visée :

## Type de changement

- [ ] Bug fix (`fix:`)
- [ ] Nouvelle fonctionnalité (`feat:`)
- [ ] Refactoring (`refactor:`)
- [ ] Documentation (`docs:`)
- [ ] Maintenance / outillage (`chore:`)
- [ ] Performance (`perf:`)
- [ ] Tests (`test:`)
- [ ] CI (`ci:`)
- [ ] Build / dépendances (`build:`)
- [ ] Annulation d'un commit (`revert:`)
- [ ] **BREAKING CHANGE** (`<type>!:`)

## Issues du lot

<!-- Obligatoire : une ligne "Closes #..." par issue du lot. -->

- Closes #

## Impact semver attendu

<!-- Une divergence avec le titre émet une alerte (non bloquante) ; la release calcule la version depuis les commits de main. -->

- [ ] **Patch** — `fix:` ou `perf:` (x.y.Z)
- [ ] **Minor** — `feat:` (x.Y.z)
- [ ] **Major** — breaking change, `<type>!:` (X.y.z)
- [ ] **Aucune release** — `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, `build:`, `revert:`

## Preuves

```shell
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest
```

## Preflight

<!-- Uniquement si un job de la release n'est pas couvert par la CI de PR. Sinon : « sans objet ». -->

- [ ] Sans objet
- [ ] Rejoué localement : <!-- quoi, résultat -->

## Checklist

- [ ] Le titre de la PR suit **Conventional Commits**
- [ ] La documentation (`README.md`, `CONTRIBUTING.md`, `docs/`) est à jour
- [ ] Aucun secret, token ou fichier `.env` commité
- [ ] Aucun appel réseau réel (LLM, embeddings, Open Agenda) dans les tests unitaires
- [ ] Le temps de CI n'augmente pas sans justification

## Notes pour le reviewer

- Périmètre de revue :
- Points de risque :
