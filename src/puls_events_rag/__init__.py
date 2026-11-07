"""Assistant RAG Puls-Events : recommandation d'événements culturels (Open Agenda)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("puls-events-rag")
except PackageNotFoundError:  # pragma: no cover - paquet non installé
    __version__ = "0.0.0"
