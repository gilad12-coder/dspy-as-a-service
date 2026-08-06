# Air-gap operators: see AIRGAP.html for
# REGISTRY_PREFIX / DEBIAN_MIRROR / PIP_INDEX_URL / PIP_TRUSTED_HOST build-args.
# With those set nothing in this file fetches from public mirrors.
ARG REGISTRY_PREFIX=docker.io
# 3.12 for PEP 683 immortal objects: core interpreter objects stop taking
# refcount writes, so job children fork()ed from the fat parent dirty far
# fewer copy-on-write pages. Full test suite verified green on 3.12.
FROM ${REGISTRY_PREFIX}/python:3.12-slim

WORKDIR /app

ARG DEBIAN_MIRROR
RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -exec \
            sed -i "s|http[s]\?://deb.debian.org|$DEBIAN_MIRROR|g; s|http[s]\?://security.debian.org|$DEBIAN_MIRROR|g" {} \; ; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock requirements.txt requirements.stable.txt ./

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
# Dependency installer toggle. PKG_MANAGER=uv (default) installs the exact
# uv.lock wheel set via uv — the dspy 3.3.0 line. PKG_MANAGER=pip installs a
# pinned requirements file; DSPY_LINE then selects which line: "preview"
# (requirements.txt, dspy 3.3.0) or "stable" (requirements.stable.txt,
# dspy 3.2.1). The agent layer bridges classic ReAct vs ReActV2, so the source
# is identical either way — only the wheel set differs:
#   docker build --build-arg PKG_MANAGER=pip --build-arg DSPY_LINE=stable ...
# Either way the deps land in their own layer (before the source copy) so it
# caches until the lockfile / requirements change. Both requirements files are
# complete compiled lockfiles, so we install with --no-deps to skip
# re-resolution. That is load-bearing for BOTH lines: each pins gepa==0.1.4
# (for GEPA PxN batched sampling), but dspy's metadata hard-pins an older gepa —
# ==0.0.27 on dspy 3.2.1, ==0.1.1 on dspy 3.3.0 — so a resolving install fails
# with ResolutionImpossible. gepa 0.1.4 runs cleanly against both dspy versions
# at runtime; only the resolver objects, and --no-deps sidesteps it. The uv path
# carries the same override in uv.lock ([tool.uv] override-dependencies).
ARG PKG_MANAGER=uv
ARG DSPY_LINE=preview
RUN if [ "$PKG_MANAGER" = "pip" ]; then \
        if [ "$DSPY_LINE" = "stable" ]; then REQ=requirements.stable.txt; else REQ=requirements.txt; fi; \
        pip install --no-cache-dir --no-deps \
            ${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL} \
            ${PIP_TRUSTED_HOST:+--trusted-host $PIP_TRUSTED_HOST} \
            -r "$REQ"; \
    else \
        pip install --no-cache-dir \
            ${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL} \
            ${PIP_TRUSTED_HOST:+--trusted-host $PIP_TRUSTED_HOST} \
            uv \
        && UV_INDEX_URL=${PIP_INDEX_URL:-} \
           UV_PYTHON_DOWNLOADS=never \
           uv sync --frozen --no-install-project; \
    fi

COPY . .

# Install the project itself. pip does an editable --no-deps install (every
# dependency is already present from the layer above); uv re-syncs now that the
# project source is in place.
RUN if [ "$PKG_MANAGER" = "pip" ]; then \
        pip install --no-cache-dir --no-deps \
            ${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL} \
            ${PIP_TRUSTED_HOST:+--trusted-host $PIP_TRUSTED_HOST} \
            -e .; \
    else \
        UV_PYTHON_DOWNLOADS=never uv sync --frozen; \
    fi

ARG GIT_SHA=unknown

# Use UID 1001 with GID 0 — works everywhere (Docker, Podman, OpenShift restricted-v2 SCC).
# OpenShift assigns a random UID at runtime but always with GID 0, so /app must be
# group-0-owned and group-writable for the editable install metadata to be accessible.
RUN groupadd -g 1001 appuser 2>/dev/null || true \
    && useradd -u 1001 -g 0 -m -s /sbin/nologin appuser \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

USER 1001

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV WORKER_CONCURRENCY=4
ENV WORKER_POLL_INTERVAL=2.0
ENV SKYNET_CODE_VERSION=$GIT_SHA
ENV LOG_LEVEL=INFO
# Containerized runs default to structured logs; override to "text" only
# when running the image interactively for debugging.
ENV LOG_FORMAT=json
# air-gap — litellm uses its bundled cost map instead of fetching model_prices_and_context_window.json from GitHub on import.
ENV LITELLM_LOCAL_MODEL_COST_MAP=True
# Recommendation embeddings are served by an internal OpenAI-compatible API;
# no model weights are loaded inside the backend container.
# REMOTE_DB_URL must be set at runtime (e.g., via docker-compose or -e flag)
# ENV REMOTE_DB_URL=postgresql://user:password@host:5432/skynet

EXPOSE 8000

# One image, two roles: the API (default) and the standalone job worker
# (SERVICE_ROLE=worker, no HTTP). Railway deploys both services from this
# same Dockerfile; the env var picks the entrypoint so neither service needs
# a dashboard-only custom start command.
# Bind the platform-provided PORT when present (Railway/Heroku-style runtimes
# inject it and route edge traffic to it) so the bound port always matches
# settings.port and the generalist agent's derived MCP self-dial URL. The
# sh -c + exec keeps the process as PID 1 for clean SIGTERM handling.
CMD ["sh", "-c", "if [ \"$SERVICE_ROLE\" = \"worker\" ]; then exec python worker_main.py; else exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]
