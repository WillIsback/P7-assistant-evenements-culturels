# syntax=docker/dockerfile:1
# Image de l'API : dépendances installées dans une couche dédiée (cache), code
# du projet ensuite, image finale sans uv ni outils de build, utilisateur non root.
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.11-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Couche dépendances : invalidée seulement si pyproject.toml / uv.lock changent.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Pas de .git dans le contexte : version fournie par la release (tag vX.Y.Z).
ARG VERSION=0.0.0
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION#v}" uv sync --locked --no-dev --no-editable

FROM python:3.11-slim
# Correctifs de sécurité Debian publiés après l'image de base (scan Trivy de la release),
# et retrait de setuptools/wheel du Python système (inutiles : l'appli tourne dans le venv).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y -q setuptools wheel \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /app/data
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Index régional construit par la release (vide en CI de PR : synchro au démarrage).
COPY --chown=app:app data/index/ /app/data/index/
# /app appartient à root : l'utilisateur app n'écrit que dans /app/data.
#   /app/data  : index régional, données collectées et base SQLite (comptes, sessions de
#                chat), à monter en volume pour les garder ;
# Cache : CACHE_URL=redis://… (Valkey, cf. compose.yaml) ; sans lui, mémoire du processus.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL="sqlite:///./data/database.db" \
    INDEX_DIR="/app/data/index"
VOLUME ["/app/data"]
USER app
WORKDIR /app
EXPOSE 8000
# Sain quand l'API répond ET que l'index régional est chargé.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import json, urllib.request; h = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if h.get('index', {}).get('loaded') else 1)"]
CMD ["uvicorn", "puls_events_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
