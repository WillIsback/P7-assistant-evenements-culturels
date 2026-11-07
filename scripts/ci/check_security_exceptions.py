"""Valide le registre des exceptions de sécurité (.github/security-exceptions.yml).

Chaque exception doit être motivée (`reason`), signée (`author`), rattachée à un
identifiant (`cve`) et datée (`expires`, au format AAAA-MM-JJ). Une exception
expirée fait échouer ce script (et donc la CI) : il faut la renouveler ou la
supprimer.

Usage : ``uv run python scripts/ci/check_security_exceptions.py`` (PyYAML requis).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[2] / ".github" / "security-exceptions.yml"
REQUIRED_FIELDS = ("cve", "reason", "author", "expires")


def check_registry(path: Path, today: dt.date) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    exceptions = data.get("exceptions") or []
    if not isinstance(exceptions, list):
        return [f"{path}: la clé « exceptions » doit être une liste."]

    for index, exception in enumerate(exceptions, start=1):
        if not isinstance(exception, dict):
            errors.append(f"Exception #{index} : doit être un mapping YAML.")
            continue
        missing = [field for field in REQUIRED_FIELDS if not exception.get(field)]
        if missing:
            errors.append(
                f"Exception #{index} ({exception.get('cve', '?')}) : "
                f"champs obligatoires manquants : {', '.join(missing)}."
            )
            continue
        try:
            expires = dt.date.fromisoformat(str(exception["expires"]))
        except ValueError:
            errors.append(
                f"Exception #{index} ({exception['cve']}) : date « expires » "
                f"invalide ({exception['expires']!r}), attendu AAAA-MM-JJ."
            )
            continue
        if expires < today:
            errors.append(
                f"Exception #{index} ({exception['cve']}) : expirée depuis le "
                f"{expires.isoformat()} — renouveler ou supprimer l'entrée."
            )
    return errors


def main() -> int:
    if not REGISTRY.is_file():
        print(f"❌ Registre introuvable : {REGISTRY}")
        return 1
    errors = check_registry(REGISTRY, dt.datetime.now(dt.UTC).date())
    if errors:
        print("❌ Registre des exceptions de sécurité invalide :")
        for error in errors:
            print(f"   - {error}")
        return 1
    print("✅ Registre des exceptions de sécurité valide (aucune exception expirée).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
