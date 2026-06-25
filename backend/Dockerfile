# Air-gap operators: see AIRGAP.html for
# REGISTRY_PREFIX / DEBIAN_MIRROR / PIP_INDEX_URL / PIP_TRUSTED_HOST build-args.
# With those set nothing in this file fetches from public mirrors.
ARG REGISTRY_PREFIX=docker.io
FROM ${REGISTRY_PREFIX}/python:3.11-slim

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
# uv.lock wheel set via uv — always the 3.3 preview DSPy line. PKG_MANAGER=pip
# installs a pinned requirements file; DSPY_LINE then selects which line:
# "preview" (requirements.txt, dspy 3.3) or "stable" (requirements.stable.txt,
# dspy 3.2.1). The agent layer bridges classic ReAct vs ReActV2, so the source
# is identical either way — only the wheel set differs:
#   docker build --build-arg PKG_MANAGER=pip --build-arg DSPY_LINE=stable ...
# Either way the deps land in their own layer (before the source copy) so it
# caches until the lockfile / requirements change. Both requirements files are
# complete compiled lockfiles, so we install with --no-deps to skip
# re-resolution. That is load-bearing for the stable line: dspy==3.2.1's
# metadata hard-pins gepa[dspy]==0.0.27, which collides with the deliberately
# overridden gepa==0.1.1 in the lockfile and makes a resolving install fail
# with ResolutionImpossible. gepa 0.1.1 runs cleanly against dspy 3.2.1 at
# runtime; only the resolver objects, and --no-deps sidesteps it.
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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
