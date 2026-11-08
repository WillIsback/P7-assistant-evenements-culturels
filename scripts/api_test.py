"""Test fonctionnel de l'API contre un serveur réel (local ou conteneur Docker).

Usage ::

    uv run python scripts/api_test.py                       # http://127.0.0.1:8000
    uv run python scripts/api_test.py --base-url http://127.0.0.1:8001

Vérifie /health (index chargé), /ask (questions de démo → réponse et sources,
validations 422) et /rebuild sans jeton (401, ou 404 si désactivé) sans déclencher
de reconstruction. Code de sortie non nul au premier échec constaté.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEMO_QUESTIONS = (
    "Je recherche un atelier d'immersion professionnelle à Aix-en-Provence "
    "dans les 15 prochains jours",
    "Quels événements sur la création d'entreprise à Marseille ?",
    "Y a-t-il des forums de recrutement dans la métropole ce mois-ci ?",
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    latency_ms: int


def _timed(name: str, call: Callable[[], tuple[bool, str]]) -> Check:
    started = time.perf_counter()
    try:
        ok, detail = call()
    except httpx.HTTPError as exc:
        ok, detail = False, f"erreur HTTP : {exc}"
    return Check(name, ok, detail, int((time.perf_counter() - started) * 1000))


def run_checks(client: httpx.Client) -> list[Check]:
    """Exécute tous les contrôles ; réutilisé par le test d'intégration."""

    def health() -> tuple[bool, str]:
        body = client.get("/health").json()
        index = body.get("index") or {}
        ok = body.get("status") == "ok" and bool(index.get("loaded"))
        return ok, f"version {body.get('version')}, {index.get('event_count')} événements"

    def ask(question: str) -> Callable[[], tuple[bool, str]]:
        def call() -> tuple[bool, str]:
            response = client.post("/ask", json={"question": question, "k": 3})
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            body = response.json()
            ok = bool(body["answer"].strip()) and len(body["sources"]) >= 1
            titles = ", ".join(source["title"][:40] for source in body["sources"])
            return ok, f"{len(body['sources'])} source(s) : {titles}"

        return call

    def rejects(payload: object) -> Callable[[], tuple[bool, str]]:
        def call() -> tuple[bool, str]:
            status = client.post("/ask", json=payload).status_code
            return status == 422, f"HTTP {status} (attendu 422)"

        return call

    def rebuild_without_token() -> tuple[bool, str]:
        status = client.post("/rebuild").status_code
        return status in (401, 404), f"HTTP {status} (attendu 401, ou 404 si désactivé)"

    checks = [_timed("GET /health", health)]
    checks += [_timed(f"POST /ask « {q[:45]}… »", ask(q)) for q in DEMO_QUESTIONS]
    checks += [
        _timed("POST /ask question vide → 422", rejects({"question": "  "})),
        _timed("POST /ask requête mal formée → 422", rejects({"k": 3})),
        _timed("POST /rebuild sans jeton → 401", rebuild_without_token),
    ]
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        checks = run_checks(client)
    for check in checks:
        mark = "OK " if check.ok else "KO "
        print(f"{mark} {check.name:60s} {check.latency_ms:6d} ms  {check.detail}")
    failed = [check for check in checks if not check.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} contrôles réussis contre {args.base_url}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
