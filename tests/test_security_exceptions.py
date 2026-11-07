import datetime as dt
from pathlib import Path

from scripts.ci.check_security_exceptions import REGISTRY, check_registry

TODAY = dt.date(2026, 11, 7)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "exceptions.yml"
    path.write_text(content, encoding="utf-8")
    return path


def test_registre_du_depot_est_valide() -> None:
    assert check_registry(REGISTRY, dt.datetime.now(dt.UTC).date()) == []


def test_exception_valide(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "exceptions:\n  - cve: CVE-1\n    reason: r\n    author: a\n    expires: '2027-01-01'\n",
    )
    assert check_registry(path, TODAY) == []


def test_exception_expiree(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "exceptions:\n  - cve: CVE-1\n    reason: r\n    author: a\n    expires: '2026-01-01'\n",
    )
    assert "expirée" in check_registry(path, TODAY)[0]


def test_champs_manquants_et_date_invalide(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "exceptions:\n  - cve: CVE-1\n  - cve: CVE-2\n    reason: r\n    author: a\n"
        "    expires: demain\n  - pas-un-mapping\n",
    )
    errors = check_registry(path, TODAY)
    assert len(errors) == 3
