"""Variante pytest de ``scripts/api_test.py`` (serveur réel requis, hors CI de PR).

``API_BASE_URL`` (défaut http://127.0.0.1:8000) ; ignoré si le serveur ne répond pas.
"""

from __future__ import annotations

import os

import httpx
import pytest
from scripts.api_test import DEFAULT_BASE_URL, run_checks

pytestmark = pytest.mark.integration


def test_api_fonctionnelle() -> None:
    base_url = os.getenv("API_BASE_URL") or DEFAULT_BASE_URL
    try:
        httpx.get(f"{base_url}/health", timeout=3)
    except httpx.HTTPError:
        pytest.skip(f"API injoignable sur {base_url}")

    with httpx.Client(base_url=base_url, timeout=60) as client:
        checks = run_checks(client)

    failed = [f"{check.name} : {check.detail}" for check in checks if not check.ok]
    assert not failed, failed
