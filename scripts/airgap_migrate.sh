#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHART_DIR="$ROOT_DIR/deploy/helm/skynet"

RELEASE="${RELEASE:-skynet}"
NAMESPACE="${NAMESPACE:-skynet}"
VALUES_OUT="${VALUES_OUT:-$ROOT_DIR/deploy/helm/skynet/values-airgap.generated.yaml}"
REGISTRY="${REGISTRY:-artifactory.your-company.com/skynet}"
IMAGE_TAG="${IMAGE_TAG:-0.1.0}"
PULL_SECRET="${PULL_SECRET:-artifactory-pull-secret}"
BACKEND_REPOSITORY="${BACKEND_REPOSITORY:-skynet/backend}"
FRONTEND_REPOSITORY="${FRONTEND_REPOSITORY:-skynet/frontend}"
POSTGRES_REPOSITORY="${POSTGRES_REPOSITORY:-pgvector/pgvector}"
EXTERNAL_DB_HOST="${EXTERNAL_DB_HOST:-pgvector.internal}"
EXTERNAL_DB_SECRET="${EXTERNAL_DB_SECRET:-skynet-db-password}"
BACKEND_SECRET="${BACKEND_SECRET:-skynet-backend-secrets}"
FRONTEND_SECRET="${FRONTEND_SECRET:-skynet-frontend-secrets}"
INTERNAL_CA_SECRET="${INTERNAL_CA_SECRET:-}"
INTERNAL_CA_FILENAME="${INTERNAL_CA_FILENAME:-ca-bundle.pem}"
INTERNAL_CA_MOUNT_DIR="${INTERNAL_CA_MOUNT_DIR:-/etc/skynet/ca}"
LLM_BASE_URL="${LLM_BASE_URL:-https://llm-gateway.internal/v1}"
CODE_AGENT_MODEL="${CODE_AGENT_MODEL:-gpt-5}"
GENERALIST_AGENT_MODEL="${GENERALIST_AGENT_MODEL:-gpt-5}"
SEARCH_BACKEND="${SEARCH_BACKEND:-lexical}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-$LLM_BASE_URL}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-jina-embeddings-v4}"
OIDC_ISSUER="${OIDC_ISSUER:-https://idp.internal/realms/skynet}"
OIDC_CLIENT_ID="${OIDC_CLIENT_ID:-skynet}"
OIDC_SCOPE="${OIDC_SCOPE:-openid profile email groups}"
AUTH_GROUP_CLAIM="${AUTH_GROUP_CLAIM:-groups}"
AUTH_ADMIN_GROUPS="${AUTH_ADMIN_GROUPS:-}"
AUTH_ADMINS="${AUTH_ADMINS:-}"
FRONTEND_HOST="${FRONTEND_HOST:-skynet.apps.internal}"
BACKEND_HOST="${BACKEND_HOST:-skynet-api.apps.internal}"
LLM_EGRESS_CIDR="${LLM_EGRESS_CIDR:-10.0.5.0/24}"
EMBEDDING_EGRESS_CIDR="${EMBEDDING_EGRESS_CIDR:-$LLM_EGRESS_CIDR}"
IDP_EGRESS_CIDR="${IDP_EGRESS_CIDR:-10.0.6.0/24}"
DB_EGRESS_CIDR="${DB_EGRESS_CIDR:-10.0.4.0/24}"
LDAP_EGRESS_CIDR="${LDAP_EGRESS_CIDR:-}"
LDAP_EGRESS_PORT="${LDAP_EGRESS_PORT:-636}"

usage() {
  cat <<'EOF'
Usage: scripts/airgap_migrate.sh <command>

Migration plan (matches AIRGAP.html):
  1. clone repo on the air-gapped host
  2. `todos`               list every TODO marker the operator must change
  3. edit URLs/secrets/CIDRs in those files
  4. `check`               verify lockfiles and alembic dir present
  5. `validate-migrations` offline alembic --sql dump (no DB required)
  6. `build-images`        docker build backend + frontend against internal mirrors
  7. `push-images`         docker push to internal Artifactory
  8. `values` + `render`   generate + lint Helm values file
  9. `install`             helm upgrade --install (runs migration Job first)
  10. `status`             rollout + smoke-test commands

Commands:
  configure            Prompt for on-prem values, write values file, optionally render/install.
  todos                Print every TODO: On-premise marker the operator must edit.
  check                Verify local repo artifacts needed for an air-gapped rollout.
  validate-migrations  Run `alembic upgrade head --sql` offline (no DB) to review schema.
  build-images         docker build backend + frontend with internal mirror build args.
  push-images          docker push backend + frontend tags to the internal registry.
  values               Generate deploy/helm/skynet/values-airgap.generated.yaml.
  render               Run helm lint + helm template with the generated values.
  install              Run helm upgrade --install; the Helm migration hook runs first.
  status               Print rollout status commands for backend/frontend.
  all                  Run check, validate-migrations, values, render, install, status.
                       NOTE: `all` does NOT build or push images — run build-images +
                       push-images (after `docker login <registry>`) first, or install
                       fails on ImagePullBackOff.

Common environment overrides:
  RELEASE=skynet
  NAMESPACE=skynet
  REGISTRY=artifactory.example.com/skynet
  IMAGE_TAG=2026.04.30
  PULL_SECRET=artifactory-pull-secret
  EXTERNAL_DB_HOST=pgvector.internal
  EXTERNAL_DB_SECRET=skynet-db-password
  BACKEND_SECRET=skynet-backend-secrets
  FRONTEND_SECRET=skynet-frontend-secrets
  INTERNAL_CA_SECRET=skynet-internal-ca
  LLM_BASE_URL=https://llm-gateway.internal/v1
  CODE_AGENT_MODEL=gpt-5
  GENERALIST_AGENT_MODEL=gpt-5
  EMBEDDING_BASE_URL=https://llm-gateway.internal/v1
  EMBEDDING_MODEL=jina-embeddings-v4
  OIDC_ISSUER=https://idp.internal/realms/skynet
  OIDC_CLIENT_ID=skynet
  OIDC_SCOPE="openid profile email groups"
  AUTH_GROUP_CLAIM=groups
  AUTH_ADMIN_GROUPS=Skynet-Admins
  AUTH_ADMINS=break-glass-admin@example.com
  FRONTEND_HOST=skynet.apps.internal
  BACKEND_HOST=skynet-api.apps.internal
  LLM_EGRESS_CIDR=10.0.5.0/24
  EMBEDDING_EGRESS_CIDR=10.0.5.0/24              # set separately if embedding gateway differs
  IDP_EGRESS_CIDR=10.0.6.0/24
  DB_EGRESS_CIDR=10.0.4.0/24                   # managed Postgres subnet; blank to omit dbEgress
  LDAP_EGRESS_CIDR=10.0.8.0/24                 # blank to omit ldapEgress
  LDAP_EGRESS_PORT=636                         # 389 if you really must use ldap://

Image build / push overrides (build-images, push-images):
  DOCKER=docker                                # or `podman`
  REGISTRY_PREFIX=artifactory.example.com/docker-remote
  DEBIAN_MIRROR=https://artifactory.example.com/debian-remote
  PIP_INDEX_URL=https://artifactory.example.com/api/pypi/pypi-remote/simple
  PIP_TRUSTED_HOST=artifactory.example.com
  BASE_IMAGE=artifactory.example.com/docker-remote/node:20-alpine
  NPM_REGISTRY=https://artifactory.example.com/api/npm/npm-remote/

Assumptions:
  - backend/frontend/postgres images already exist in the internal registry,
    or you ran `build-images` + `push-images` against your Artifactory.
  - the namespace already has image pull, DB password, AUTH_SECRET,
    BACKEND_AUTH_SECRET, and OIDC client-secret Kubernetes secrets.
  - kubectl or oc is already authenticated to the target cluster.
EOF
}

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

prompt() {
  local var_name="$1"
  local label="$2"
  local current="${!var_name}"
  local answer
  read -r -p "$label [$current]: " answer
  if [[ -n "$answer" ]]; then
    printf -v "$var_name" '%s' "$answer"
  fi
}

prompt_secret_name() {
  local var_name="$1"
  local label="$2"
  local current="${!var_name}"
  local answer
  read -r -p "$label [$current]: " answer
  if [[ -n "$answer" ]]; then
    printf -v "$var_name" '%s' "$answer"
  fi
}

confirm() {
  local label="$1"
  local answer
  read -r -p "$label [y/N]: " answer
  [[ "$answer" == "y" || "$answer" == "Y" || "$answer" == "yes" || "$answer" == "YES" ]]
}

cmd_configure() {
  if [[ ! -t 0 ]]; then
    echo "configure requires an interactive terminal" >&2
    exit 1
  fi

  cat <<'EOF'
Skynet air-gap setup

Press Enter to accept a default. Secrets are referenced by Kubernetes Secret
name only; this script does not ask for secret values or write credentials.
EOF

  prompt RELEASE "Helm release name"
  prompt NAMESPACE "Kubernetes namespace"
  prompt VALUES_OUT "Output values file"
  prompt REGISTRY "Internal image registry prefix"
  prompt IMAGE_TAG "Backend/frontend image tag"
  prompt PULL_SECRET "Image pull secret name"
  prompt BACKEND_REPOSITORY "Backend image repository under registry"
  prompt FRONTEND_REPOSITORY "Frontend image repository under registry"
  prompt POSTGRES_REPOSITORY "pgvector image repository under registry"
  prompt EXTERNAL_DB_HOST "External pgvector host"
  prompt_secret_name EXTERNAL_DB_SECRET "DB password Secret name"
  prompt_secret_name BACKEND_SECRET "Backend Secret name"
  prompt_secret_name FRONTEND_SECRET "Frontend Secret name"
  prompt_secret_name INTERNAL_CA_SECRET "Internal CA Secret name (blank to skip)"
  if [[ -n "$INTERNAL_CA_SECRET" ]]; then
    prompt INTERNAL_CA_FILENAME "Internal CA filename in Secret"
    prompt INTERNAL_CA_MOUNT_DIR "Internal CA mount directory"
  fi
  prompt LLM_BASE_URL "Internal OpenAI-compatible LLM base URL"
  prompt CODE_AGENT_MODEL "Code-agent model id served by the gateway"
  prompt GENERALIST_AGENT_MODEL "Generalist-agent model id served by the gateway"
  prompt EMBEDDING_BASE_URL "Internal OpenAI-compatible embedding base URL"
  prompt EMBEDDING_MODEL "Embedding model id"
  prompt OIDC_ISSUER "Internal ADFS/OIDC issuer URL"
  prompt OIDC_CLIENT_ID "ADFS/OIDC client ID"
  prompt OIDC_SCOPE "ADFS/OIDC scopes"
  prompt AUTH_GROUP_CLAIM "ADFS/OIDC group claim name"
  prompt AUTH_ADMIN_GROUPS "Comma-separated admin ADFS/OIDC groups"
  prompt AUTH_ADMINS "Comma-separated break-glass admin users/emails"
  prompt FRONTEND_HOST "Frontend route host"
  prompt BACKEND_HOST "Backend route host"
  prompt LLM_EGRESS_CIDR "LLM gateway egress CIDR"
  prompt EMBEDDING_EGRESS_CIDR "Embedding gateway egress CIDR"
  prompt IDP_EGRESS_CIDR "IdP egress CIDR"
  prompt DB_EGRESS_CIDR "Managed Postgres egress CIDR (blank to omit)"
  prompt LDAP_EGRESS_CIDR "LDAP/AD controller egress CIDR (blank to omit)"
  if [[ -n "$LDAP_EGRESS_CIDR" ]]; then
    prompt LDAP_EGRESS_PORT "LDAP/AD egress port (636 ldaps / 389 ldap)"
  fi

  cmd_values

  echo
  echo "Next required secrets:"
  cat <<EOF
  $PULL_SECRET                  docker-registry pull secret
  $EXTERNAL_DB_SECRET           key: password
  $BACKEND_SECRET               keys: OPENAI_API_KEY, BACKEND_AUTH_SECRET
  $FRONTEND_SECRET              keys: AUTH_SECRET, AUTH_SSO_CLIENT_SECRET, BACKEND_AUTH_SECRET
EOF
  if [[ -n "$INTERNAL_CA_SECRET" ]]; then
    echo "  $INTERNAL_CA_SECRET              key: $INTERNAL_CA_FILENAME"
  fi

  if confirm "Run artifact check now"; then
    cmd_check
  fi
  if confirm "Render Helm chart now"; then
    cmd_render
  fi
  if confirm "Install/upgrade now (runs Alembic migration hook)"; then
    cmd_install
    cmd_status
  else
    cmd_status
  fi
}

cmd_check() {
  need helm
  [[ -f "$ROOT_DIR/backend/Dockerfile" ]] || { echo "missing backend/Dockerfile" >&2; exit 1; }
  [[ -f "$ROOT_DIR/frontend/Dockerfile" ]] || { echo "missing frontend/Dockerfile" >&2; exit 1; }
  [[ -f "$ROOT_DIR/frontend/package-lock.json" ]] || { echo "missing frontend/package-lock.json" >&2; exit 1; }
  [[ -f "$ROOT_DIR/backend/uv.lock" ]] || { echo "missing backend/uv.lock" >&2; exit 1; }
  [[ -f "$ROOT_DIR/backend/requirements.txt" ]] || { echo "missing backend/requirements.txt" >&2; exit 1; }
  [[ -d "$ROOT_DIR/backend/alembic/versions" ]] || { echo "missing backend/alembic/versions" >&2; exit 1; }
  echo "air-gap artifact check passed"
}

cmd_todos() {
  echo "TODO: On-premise markers — every place an operator must touch:"
  echo
  if command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse >/dev/null 2>&1; then
    git -C "$ROOT_DIR" grep -n "TODO: On-premise\|TODO: On-prem" -- \
      ':!*.lock' ':!*.lockb' ':!node_modules' ':!.venv'
  else
    grep -rn "TODO: On-premise\|TODO: On-prem" \
      --include="*.py" --include="*.ts" --include="*.tsx" \
      --include="*.yaml" --include="*.yml" --include="*.md" \
      --include="*.toml" --include="*.json" --include="*.example" \
      --include="*.sh" --include="Dockerfile*" \
      "$ROOT_DIR" 2>/dev/null
  fi
  echo
  echo "Action: edit each line above before running 'values' / 'install'."
}

# Offline alembic SQL emission. Sandbox-friendly: needs no Postgres connection,
# imports the same env.py as the migration Job. Use this to review the exact
# schema delta before pushing the backend image to your internal registry.
# Honours the PKG_MANAGER toggle (uv by default, pip when set to "pip"); needs
# no network IF the backend deps are already importable, or an internal PyPI
# mirror is reachable via UV_INDEX_URL (uv) / PIP_INDEX_URL (pip).
cmd_validate_migrations() {
  local backend_dir="$ROOT_DIR/backend"
  local out="${MIGRATION_SQL_OUT:-$ROOT_DIR/migration.sql}"
  [[ -d "$backend_dir/alembic/versions" ]] || {
    echo "missing backend/alembic/versions" >&2; exit 1
  }
  local pkg_manager="${PKG_MANAGER:-uv}"
  local runner=""
  if [[ "$pkg_manager" == "uv" ]] && command -v uv >/dev/null 2>&1; then
    # Materialize the lockfile-pinned env first. Without this the script
    # falls through to whatever 'alembic' the system Python happens to
    # expose, which is how validate-migrations silently produced an empty
    # SQL file from a stale .venv. UV_PYTHON_DOWNLOADS=never mirrors
    # backend/Dockerfile so uv never reaches out to download a Python build;
    # if UV_INDEX_URL is set we pass it through so an internal PyPI mirror
    # satisfies the sync with no internet.
    ( cd "$backend_dir" && UV_PYTHON_DOWNLOADS=never ${UV_INDEX_URL:+UV_INDEX_URL="$UV_INDEX_URL"} uv sync --frozen --quiet ) || {
      echo "uv sync --frozen failed; resolve dep conflicts before validate-migrations" >&2
      exit 1
    }
    runner="uv run"
  else
    # pip path — also the fallback when PKG_MANAGER=uv but uv is absent.
    # Confirm the migration deps import; if not, materialize the pinned
    # requirements.txt the same way backend/Dockerfile's pip path does.
    # alembic/env.py never imports ldap3, so we do not require it — demanding
    # it wrongly rejects otherwise-capable hosts.
    if ! python3 -c "import alembic, sqlalchemy, pgvector" 2>/dev/null; then
      python3 -m pip --version >/dev/null 2>&1 || {
        echo "backend deps not importable and pip unavailable; install alembic/sqlalchemy/pgvector before validate-migrations" >&2
        exit 1
      }
      ( cd "$backend_dir" && python3 -m pip install --quiet \
          ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
          ${PIP_TRUSTED_HOST:+--trusted-host "$PIP_TRUSTED_HOST"} \
          -r requirements.txt ) || {
        echo "pip install -r requirements.txt failed; resolve dep conflicts before validate-migrations" >&2
        exit 1
      }
      python3 -c "import alembic, sqlalchemy, pgvector" 2>/dev/null || {
        echo "backend deps still missing after pip install; aborting" >&2
        exit 1
      }
    fi
    runner="python3 -m"
  fi
  # env.py reads REMOTE_DB_URL but offline --sql only needs the dialect, so
  # supply a placeholder when the operator hasn't set one. A real value is
  # still honoured if exported.
  local db_url="${REMOTE_DB_URL:-postgresql+psycopg2://placeholder/skynet}"
  ( cd "$backend_dir" && REMOTE_DB_URL="$db_url" $runner alembic upgrade head --sql ) >"$out"
  if [[ ! -s "$out" ]]; then
    echo "alembic produced an empty $out; aborting" >&2
    exit 1
  fi
  echo "wrote offline migration SQL: $out"
  echo "review this file; the in-cluster migration Job will execute the same statements against REMOTE_DB_URL."
}

cmd_build_images() {
  need "${DOCKER:-docker}"
  local docker_bin="${DOCKER:-docker}"
  local backend_args=(
    --build-arg "REGISTRY_PREFIX=${REGISTRY_PREFIX:-docker.io}"
  )
  [[ -n "${DEBIAN_MIRROR:-}" ]] && backend_args+=(--build-arg "DEBIAN_MIRROR=$DEBIAN_MIRROR")
  [[ -n "${PIP_INDEX_URL:-}" ]] && backend_args+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
  [[ -n "${PIP_TRUSTED_HOST:-}" ]] && backend_args+=(--build-arg "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST")
  local frontend_args=()
  [[ -n "${BASE_IMAGE:-}" ]] && frontend_args+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
  [[ -n "${NPM_REGISTRY:-}" ]] && frontend_args+=(--build-arg "NPM_REGISTRY=$NPM_REGISTRY")

  local backend_tag="$REGISTRY/$BACKEND_REPOSITORY:$IMAGE_TAG"
  local frontend_tag="$REGISTRY/$FRONTEND_REPOSITORY:$IMAGE_TAG"

  echo "Building $backend_tag"
  "$docker_bin" build "$ROOT_DIR/backend" -t "$backend_tag" "${backend_args[@]}"
  echo "Building $frontend_tag"
  "$docker_bin" build "$ROOT_DIR/frontend" -t "$frontend_tag" "${frontend_args[@]}"
  echo "Built:"
  echo "  $backend_tag"
  echo "  $frontend_tag"
}

cmd_push_images() {
  need "${DOCKER:-docker}"
  local docker_bin="${DOCKER:-docker}"
  local backend_tag="$REGISTRY/$BACKEND_REPOSITORY:$IMAGE_TAG"
  local frontend_tag="$REGISTRY/$FRONTEND_REPOSITORY:$IMAGE_TAG"
  echo "Reminder: '$docker_bin login $REGISTRY' (or 'podman login') must have succeeded or these pushes 401." >&2
  "$docker_bin" push "$backend_tag"
  "$docker_bin" push "$frontend_tag"
  echo "Pushed:"
  echo "  $backend_tag"
  echo "  $frontend_tag"
}

cmd_values() {
  mkdir -p "$(dirname "$VALUES_OUT")"
  local ca_bundle_path=""
  local ca_backend_env=""
  local ca_backend_mounts=""
  local ca_frontend_mounts=""
  if [[ -n "$INTERNAL_CA_SECRET" ]]; then
    ca_bundle_path="$INTERNAL_CA_MOUNT_DIR/$INTERNAL_CA_FILENAME"
    ca_backend_env=$(cat <<EOF
    SSL_CERT_FILE: "$ca_bundle_path"
    REQUESTS_CA_BUNDLE: "$ca_bundle_path"
EOF
)
    ca_backend_mounts=$(cat <<EOF
  extraVolumes:
    - name: internal-ca
      secret:
        secretName: "$INTERNAL_CA_SECRET"
  extraVolumeMounts:
    - name: internal-ca
      mountPath: "$INTERNAL_CA_MOUNT_DIR"
      readOnly: true
EOF
)
    ca_frontend_mounts="$ca_backend_mounts"
  fi
  local db_egress=""
  if [[ -n "$DB_EGRESS_CIDR" ]]; then
    db_egress=$(printf '\n  # TODO: On-premise - the managed Postgres subnet. REQUIRED when\n  # postgres.enabled=false or the backend pool can never reach the DB.\n  dbEgress:\n    - cidr: "%s"\n      ports: [5432]' "$DB_EGRESS_CIDR")
  fi
  local embedding_egress=""
  if [[ -n "$EMBEDDING_EGRESS_CIDR" && "$EMBEDDING_EGRESS_CIDR" != "$LLM_EGRESS_CIDR" ]]; then
    embedding_egress=$(printf '\n  # TODO: On-premise - use the exact embedding gateway CIDRs and ports.\n  embeddingEgress:\n    - cidr: "%s"\n      ports: [443]' "$EMBEDDING_EGRESS_CIDR")
  fi
  local ldap_egress=""
  if [[ -n "$LDAP_EGRESS_CIDR" ]]; then
    ldap_egress=$(printf '\n  # TODO: On-premise - use the exact AD/LDAP controller CIDR and port (636 ldaps / 389 ldap).\n  ldapEgress:\n    - cidr: "%s"\n      ports: [%s]' "$LDAP_EGRESS_CIDR" "$LDAP_EGRESS_PORT")
  fi
  cat >"$VALUES_OUT" <<EOF
# Generated by scripts/airgap_migrate.sh.
# TODO: On-premise - review every value before installing.
global:
  imageRegistry: "$REGISTRY"
  imagePullSecrets:
    - name: "$PULL_SECRET"

backend:
  image:
    repository: "$BACKEND_REPOSITORY"
    tag: "$IMAGE_TAG"
    pullPolicy: IfNotPresent
  env:
    # TODO: On-premise - point these at your OpenAI-compatible internal LLM gateway.
    CODE_AGENT_BASE_URL: "$LLM_BASE_URL"
    GENERALIST_AGENT_BASE_URL: "$LLM_BASE_URL"
    # TODO: On-premise - set to a model id your internal gateway actually serves (gpt-5 is a placeholder). LiteLLM forwards this id verbatim to CODE_AGENT_BASE_URL.
    CODE_AGENT_MODEL: "$CODE_AGENT_MODEL"
    GENERALIST_AGENT_MODEL: "$GENERALIST_AGENT_MODEL"
    # TAGGER_ASSIST_BASE_URL / TAGGER_ASSIST_MODEL are optional overrides for
    # the tagging assist; left unset it reuses the generalist pair above.
    # Explore search backend: lexical (vanilla Postgres, default) | bm25 (needs
    # the pg_search extension) | semantic (needs pgvector + the embedding gateway
    # below). Only "semantic" makes the migrate Job run CREATE EXTENSION vector.
    SEARCH_BACKEND: "$SEARCH_BACKEND"
    # Only used when SEARCH_BACKEND=semantic:
    EMBEDDINGS_BASE_URL: "$EMBEDDING_BASE_URL"
    EMBEDDINGS_MODEL: "$EMBEDDING_MODEL"
    # Air-gap: use the bundled litellm cost map; do not fetch from GitHub on import.
    LITELLM_LOCAL_MODEL_COST_MAP: "True"
$ca_backend_env
    # TODO: On-premise - set to the public frontend route.
    FRONTEND_URL: "https://$FRONTEND_HOST"
    # TODO: On-premise - list every browser origin allowed to call the backend.
    ALLOWED_ORIGINS: "https://$FRONTEND_HOST"
    ADMIN_GROUPS: "$AUTH_ADMIN_GROUPS"
    ADMIN_USERNAMES: "$AUTH_ADMINS"
    # Optional: enables the LDAP-backed admin user-search API
    # (GET /admin/users/search; no UI consumes it). Leave empty to keep the
    # NullDirectoryClient fallback (DB-known usernames only). See AIRGAP.html
    # "Internal LDAP / Active Directory User Search" for the full env contract.
    AD_LDAP_URL: ""
    AD_LDAP_BIND_DN: ""
    AD_LDAP_SEARCH_BASE: ""
    AD_LDAP_USER_FILTER: ""
    AD_LDAP_USERNAME_ATTR: ""
  secrets:
    # TODO: On-premise - must contain OPENAI_API_KEY and BACKEND_AUTH_SECRET.
    # OPENAI_API_KEY is also reused for the embedding API unless you add
    # EMBEDDINGS_API_KEY to this Secret.
    # For an unauthenticated internal gateway, set OPENAI_API_KEY to any non-empty placeholder (e.g. "not-needed").
    # Add AD_LDAP_BIND_PASSWORD when AD_LDAP_URL is set.
    existingSecret: "$BACKEND_SECRET"
$ca_backend_mounts

frontend:
  image:
    repository: "$FRONTEND_REPOSITORY"
    tag: "$IMAGE_TAG"
    pullPolicy: IfNotPresent
  env:
    API_URL: "https://$BACKEND_HOST"
    # TODO: On-premise - point at your internal OIDC issuer.
    AUTH_SSO_ISSUER: "$OIDC_ISSUER"
    AUTH_SSO_CLIENT_ID: "$OIDC_CLIENT_ID"
    AUTH_SSO_SCOPE: "$OIDC_SCOPE"
    AUTH_GROUP_CLAIM: "$AUTH_GROUP_CLAIM"
    AUTH_ADMIN_GROUPS: "$AUTH_ADMIN_GROUPS"
    AUTH_ADMINS: "$AUTH_ADMINS"
    # TODO: On-premise - set when your IdP uses a private CA mounted in the pod.
    NODE_EXTRA_CA_CERTS: "$ca_bundle_path"
  secrets:
    # TODO: On-premise - must contain AUTH_SECRET, AUTH_SSO_CLIENT_SECRET, and BACKEND_AUTH_SECRET.
    existingSecret: "$FRONTEND_SECRET"
$ca_frontend_mounts

externalDatabase:
  enabled: true
  host: "$EXTERNAL_DB_HOST"
  port: 5432
  database: skynet
  user: skynet
  existingSecret: "$EXTERNAL_DB_SECRET"
  existingSecretKey: password
  composeUrl: true
  sslmode: require

postgres:
  enabled: false
  image:
    repository: "$POSTGRES_REPOSITORY"
    tag: pg16

openshift:
  routes:
    enabled: true
    backend:
      host: "$BACKEND_HOST"
    frontend:
      host: "$FRONTEND_HOST"

networkPolicy:
  enabled: true$db_egress
  # TODO: On-premise - use the exact IdP / internal service CIDRs for your cluster.
  egressCidrs:
    - "$IDP_EGRESS_CIDR"
  # TODO: On-premise - use the exact LLM gateway CIDRs and ports.
  llmEgress:
    - cidr: "$LLM_EGRESS_CIDR"
      ports: [443]$embedding_egress$ldap_egress

migration:
  enabled: true
  command: ["alembic", "upgrade", "head"]
EOF
  echo "wrote $VALUES_OUT"
}

cmd_render() {
  [[ -f "$VALUES_OUT" ]] || cmd_values
  helm lint "$CHART_DIR" -f "$VALUES_OUT"
  helm template "$RELEASE" "$CHART_DIR" -n "$NAMESPACE" -f "$VALUES_OUT" >/tmp/skynet-airgap-rendered.yaml
  echo "rendered manifest: /tmp/skynet-airgap-rendered.yaml"
}

cmd_install() {
  need helm
  [[ -f "$VALUES_OUT" ]] || cmd_values
  helm upgrade --install "$RELEASE" "$CHART_DIR" \
    --namespace "$NAMESPACE" \
    --create-namespace \
    -f "$VALUES_OUT"
}

cmd_status() {
  cat <<EOF
Run:
  kubectl -n $NAMESPACE rollout status deploy/$RELEASE-skynet-backend
  kubectl -n $NAMESPACE rollout status deploy/$RELEASE-skynet-frontend
  kubectl -n $NAMESPACE logs job/$RELEASE-skynet-migrate

Smoke test:
  curl -k https://$FRONTEND_HOST/
  curl -k https://$BACKEND_HOST/health

ADFS / OIDC callback URL:
  https://$FRONTEND_HOST/api/auth/callback/adfs
EOF
}

case "${1:-}" in
  configure) cmd_configure ;;
  todos) cmd_todos ;;
  check) cmd_check ;;
  validate-migrations) cmd_validate_migrations ;;
  build-images) cmd_build_images ;;
  push-images) cmd_push_images ;;
  values) cmd_values ;;
  render) cmd_render ;;
  install) cmd_install ;;
  status) cmd_status ;;
  all)
    echo "Note: 'all' does NOT build or push images — run build-images + push-images" >&2
    echo "(after 'docker login $REGISTRY') first, or install fails on ImagePullBackOff." >&2
    cmd_check
    cmd_validate_migrations
    cmd_values
    cmd_render
    cmd_install
    cmd_status
    ;;
  -h|--help|help|"") usage ;;
  *) usage; exit 2 ;;
esac
