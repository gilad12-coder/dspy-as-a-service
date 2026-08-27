#Requires -Version 5.1
<#
    Windows / PowerShell port of scripts/airgap_migrate.sh.

    Same subcommands, same env-var contract, same exit codes. The companion
    airgap_migrate.bat is a thin shim that forwards arguments to this script
    so operators can invoke either.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [string]$Command = '',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# ---------------------------------------------------------------------------
# Resolve paths from the script's own location, the same way the .sh does
# with `dirname "${BASH_SOURCE[0]}"/..`. Works no matter where the operator
# is cd'd to.
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$ChartDir  = Join-Path $RootDir 'deploy\helm\skynet'

function Get-OrDefault {
    param([string]$Name, [string]$Default)
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($v)) { return $Default }
    return $v
}

# ---------------------------------------------------------------------------
# Defaults — mirror the bash script line-for-line. Each one is a process-env
# read so values exported in cmd.exe (`set FOO=bar`) or PowerShell
# (`$env:FOO='bar'`) before invoking propagate through.
# ---------------------------------------------------------------------------
$RELEASE                = Get-OrDefault 'RELEASE'                'skynet'
$NAMESPACE              = Get-OrDefault 'NAMESPACE'              'skynet'
$VALUES_OUT             = Get-OrDefault 'VALUES_OUT'             (Join-Path $RootDir 'deploy\helm\skynet\values-airgap.generated.yaml')
$REGISTRY               = Get-OrDefault 'REGISTRY'               'artifactory.your-company.com/skynet'
$IMAGE_TAG              = Get-OrDefault 'IMAGE_TAG'              '0.1.0'
$PULL_SECRET            = Get-OrDefault 'PULL_SECRET'            'artifactory-pull-secret'
$BACKEND_REPOSITORY     = Get-OrDefault 'BACKEND_REPOSITORY'     'skynet/backend'
$FRONTEND_REPOSITORY    = Get-OrDefault 'FRONTEND_REPOSITORY'    'skynet/frontend'
$POSTGRES_REPOSITORY    = Get-OrDefault 'POSTGRES_REPOSITORY'    'pgvector/pgvector'
$EXTERNAL_DB_HOST       = Get-OrDefault 'EXTERNAL_DB_HOST'       'pgvector.internal'
$EXTERNAL_DB_SECRET     = Get-OrDefault 'EXTERNAL_DB_SECRET'     'skynet-db-password'
$BACKEND_SECRET         = Get-OrDefault 'BACKEND_SECRET'         'skynet-backend-secrets'
$FRONTEND_SECRET        = Get-OrDefault 'FRONTEND_SECRET'        'skynet-frontend-secrets'
$INTERNAL_CA_SECRET     = Get-OrDefault 'INTERNAL_CA_SECRET'     ''
$INTERNAL_CA_FILENAME   = Get-OrDefault 'INTERNAL_CA_FILENAME'   'ca-bundle.pem'
$INTERNAL_CA_MOUNT_DIR  = Get-OrDefault 'INTERNAL_CA_MOUNT_DIR'  '/etc/skynet/ca'
$LLM_BASE_URL           = Get-OrDefault 'LLM_BASE_URL'           'https://llm-gateway.internal/v1'
$CODE_AGENT_MODEL       = Get-OrDefault 'CODE_AGENT_MODEL'       'gpt-5'
$GENERALIST_AGENT_MODEL = Get-OrDefault 'GENERALIST_AGENT_MODEL' 'gpt-5'
$SEARCH_BACKEND        = Get-OrDefault 'SEARCH_BACKEND'         'lexical'
$EMBEDDING_BASE_URL     = Get-OrDefault 'EMBEDDING_BASE_URL'     $LLM_BASE_URL
$EMBEDDING_MODEL        = Get-OrDefault 'EMBEDDING_MODEL'        'jina-embeddings-v4'
$OIDC_ISSUER            = Get-OrDefault 'OIDC_ISSUER'            'https://idp.internal/realms/skynet'
$OIDC_CLIENT_ID         = Get-OrDefault 'OIDC_CLIENT_ID'         'skynet'
$OIDC_SCOPE             = Get-OrDefault 'OIDC_SCOPE'             'openid profile email groups'
$AUTH_GROUP_CLAIM       = Get-OrDefault 'AUTH_GROUP_CLAIM'       'groups'
$AUTH_ADMIN_GROUPS      = Get-OrDefault 'AUTH_ADMIN_GROUPS'      ''
$AUTH_ADMINS            = Get-OrDefault 'AUTH_ADMINS'            ''
$FRONTEND_HOST          = Get-OrDefault 'FRONTEND_HOST'          'skynet.apps.internal'
$BACKEND_HOST           = Get-OrDefault 'BACKEND_HOST'           'skynet-api.apps.internal'
$LLM_EGRESS_CIDR        = Get-OrDefault 'LLM_EGRESS_CIDR'        '10.0.5.0/24'
$EMBEDDING_EGRESS_CIDR  = Get-OrDefault 'EMBEDDING_EGRESS_CIDR'  $LLM_EGRESS_CIDR
$IDP_EGRESS_CIDR        = Get-OrDefault 'IDP_EGRESS_CIDR'        '10.0.6.0/24'
$DB_EGRESS_CIDR         = Get-OrDefault 'DB_EGRESS_CIDR'         '10.0.4.0/24'
$LDAP_EGRESS_CIDR       = Get-OrDefault 'LDAP_EGRESS_CIDR'       ''
$LDAP_EGRESS_PORT       = Get-OrDefault 'LDAP_EGRESS_PORT'       '636'

function Write-StdErr {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Write-Usage {
    $usage = @'
Usage: scripts\airgap_migrate.bat <command>

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
  values               Generate deploy\helm\skynet\values-airgap.generated.yaml.
  render               Run helm lint + helm template with the generated values.
  install              Run helm upgrade --install; the Helm migration hook runs first.
  status               Print rollout status commands for backend/frontend.
  all                  Run check, validate-migrations, values, render, install, status.
                       NOTE: `all` does NOT build or push images — run build-images +
                       push-images (after `docker login <registry>`) first, or install
                       fails on ImagePullBackOff.

Common environment overrides (set with `set NAME=value` in cmd or `$env:NAME='value'` in PowerShell):
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
'@
    Write-Host $usage
}

function Test-Need {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-StdErr "missing required command: $Name"
        exit 1
    }
}

function Invoke-NativeProbe {
    # Windows PowerShell 5.1 turns redirected native stderr into a terminating
    # NativeCommandError under ErrorActionPreference=Stop, so run availability
    # probes with it relaxed; callers read $LASTEXITCODE for the outcome.
    param([scriptblock]$Block)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try { & $Block | Out-Null }
    finally { $ErrorActionPreference = $previous }
}

function Read-Prompt {
    param(
        [string]$VarName,
        [string]$Label
    )
    $current = Get-Variable -Name $VarName -Scope Script -ValueOnly
    $answer = Read-Host "$Label [$current]"
    if (-not [string]::IsNullOrEmpty($answer)) {
        Set-Variable -Name $VarName -Value $answer -Scope Script
    }
}

function Read-Confirm {
    param([string]$Label)
    $answer = Read-Host "$Label [y/N]"
    return ($answer -in @('y', 'Y', 'yes', 'YES'))
}

function Invoke-Configure {
    if (-not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
        Write-StdErr 'configure requires an interactive terminal'
        exit 1
    }

    Write-Host @'
Skynet air-gap setup

Press Enter to accept a default. Secrets are referenced by Kubernetes Secret
name only; this script does not ask for secret values or write credentials.
'@

    Read-Prompt 'RELEASE'                'Helm release name'
    Read-Prompt 'NAMESPACE'              'Kubernetes namespace'
    Read-Prompt 'VALUES_OUT'             'Output values file'
    Read-Prompt 'REGISTRY'               'Internal image registry prefix'
    Read-Prompt 'IMAGE_TAG'              'Backend/frontend image tag'
    Read-Prompt 'PULL_SECRET'            'Image pull secret name'
    Read-Prompt 'BACKEND_REPOSITORY'     'Backend image repository under registry'
    Read-Prompt 'FRONTEND_REPOSITORY'    'Frontend image repository under registry'
    Read-Prompt 'POSTGRES_REPOSITORY'    'pgvector image repository under registry'
    Read-Prompt 'EXTERNAL_DB_HOST'       'External pgvector host'
    Read-Prompt 'EXTERNAL_DB_SECRET'     'DB password Secret name'
    Read-Prompt 'BACKEND_SECRET'         'Backend Secret name'
    Read-Prompt 'FRONTEND_SECRET'        'Frontend Secret name'
    Read-Prompt 'INTERNAL_CA_SECRET'     'Internal CA Secret name (blank to skip)'
    if (-not [string]::IsNullOrEmpty($script:INTERNAL_CA_SECRET)) {
        Read-Prompt 'INTERNAL_CA_FILENAME'   'Internal CA filename in Secret'
        Read-Prompt 'INTERNAL_CA_MOUNT_DIR'  'Internal CA mount directory'
    }
    Read-Prompt 'LLM_BASE_URL'           'Internal OpenAI-compatible LLM base URL'
    Read-Prompt 'CODE_AGENT_MODEL'       'Code-agent model id served by the gateway'
    Read-Prompt 'GENERALIST_AGENT_MODEL' 'Generalist-agent model id served by the gateway'
    Read-Prompt 'EMBEDDING_BASE_URL'     'Internal OpenAI-compatible embedding base URL'
    Read-Prompt 'EMBEDDING_MODEL'        'Embedding model id'
    Read-Prompt 'OIDC_ISSUER'            'Internal ADFS/OIDC issuer URL'
    Read-Prompt 'OIDC_CLIENT_ID'         'ADFS/OIDC client ID'
    Read-Prompt 'OIDC_SCOPE'             'ADFS/OIDC scopes'
    Read-Prompt 'AUTH_GROUP_CLAIM'       'ADFS/OIDC group claim name'
    Read-Prompt 'AUTH_ADMIN_GROUPS'      'Comma-separated admin ADFS/OIDC groups'
    Read-Prompt 'AUTH_ADMINS'            'Comma-separated break-glass admin users/emails'
    Read-Prompt 'FRONTEND_HOST'          'Frontend route host'
    Read-Prompt 'BACKEND_HOST'           'Backend route host'
    Read-Prompt 'LLM_EGRESS_CIDR'        'LLM gateway egress CIDR'
    Read-Prompt 'EMBEDDING_EGRESS_CIDR'  'Embedding gateway egress CIDR'
    Read-Prompt 'IDP_EGRESS_CIDR'        'IdP egress CIDR'
    Read-Prompt 'DB_EGRESS_CIDR'         'Managed Postgres egress CIDR (blank to omit)'
    Read-Prompt 'LDAP_EGRESS_CIDR'       'LDAP/AD controller egress CIDR (blank to omit)'
    if (-not [string]::IsNullOrEmpty($script:LDAP_EGRESS_CIDR)) {
        Read-Prompt 'LDAP_EGRESS_PORT'       'LDAP/AD egress port (636 ldaps / 389 ldap)'
    }

    Invoke-Values

    Write-Host ''
    Write-Host 'Next required secrets:'
    Write-Host "  $($script:PULL_SECRET)                  docker-registry pull secret"
    Write-Host "  $($script:EXTERNAL_DB_SECRET)           key: password"
    Write-Host "  $($script:BACKEND_SECRET)               keys: OPENAI_API_KEY, BACKEND_AUTH_SECRET"
    Write-Host "  $($script:FRONTEND_SECRET)              keys: AUTH_SECRET, AUTH_SSO_CLIENT_SECRET, BACKEND_AUTH_SECRET"
    if (-not [string]::IsNullOrEmpty($script:INTERNAL_CA_SECRET)) {
        Write-Host "  $($script:INTERNAL_CA_SECRET)              key: $($script:INTERNAL_CA_FILENAME)"
    }

    if (Read-Confirm 'Run artifact check now') {
        Invoke-Check
    }
    if (Read-Confirm 'Render Helm chart now') {
        Invoke-Render
    }
    if (Read-Confirm 'Install/upgrade now (runs Alembic migration hook)') {
        Invoke-Install
        Invoke-Status
    }
    else {
        Invoke-Status
    }
}

function Invoke-Check {
    Test-Need 'helm'
    $required = @(
        @{ Path = (Join-Path $RootDir 'backend\Dockerfile');           Type = 'File'; Label = 'backend/Dockerfile' },
        @{ Path = (Join-Path $RootDir 'frontend\Dockerfile');          Type = 'File'; Label = 'frontend/Dockerfile' },
        @{ Path = (Join-Path $RootDir 'frontend\package-lock.json');   Type = 'File'; Label = 'frontend/package-lock.json' },
        @{ Path = (Join-Path $RootDir 'backend\uv.lock');              Type = 'File'; Label = 'backend/uv.lock' },
        @{ Path = (Join-Path $RootDir 'backend\requirements.txt');     Type = 'File'; Label = 'backend/requirements.txt' },
        @{ Path = (Join-Path $RootDir 'backend\alembic\versions');     Type = 'Container'; Label = 'backend/alembic/versions' }
    )
    foreach ($r in $required) {
        if (-not (Test-Path -LiteralPath $r.Path -PathType $r.Type)) {
            Write-StdErr "missing $($r.Label)"
            exit 1
        }
    }
    Write-Host 'air-gap artifact check passed'
}

function Invoke-Todos {
    Write-Host 'TODO: On-premise markers - every place an operator must touch:'
    Write-Host ''
    $pattern = 'TODO: On-premise|TODO: On-prem'
    $hasGit = [bool](Get-Command git -ErrorAction SilentlyContinue)
    $isRepo = $false
    if ($hasGit) {
        Push-Location $RootDir
        try {
            Invoke-NativeProbe { git rev-parse 2>$null }
            $isRepo = ($LASTEXITCODE -eq 0)
        }
        finally {
            Pop-Location
        }
    }

    if ($hasGit -and $isRepo) {
        Push-Location $RootDir
        try {
            # `git grep` mirrors the .sh exactly: same pattern, same path
            # exclusions. A no-match exit code 1 is not a terminating error
            # in Windows PowerShell, so it needs no special handling.
            git grep -n 'TODO: On-premise\|TODO: On-prem' -- ':!*.lock' ':!*.lockb' ':!node_modules' ':!.venv'
        }
        finally {
            Pop-Location
        }
    }
    else {
        $includes = @('*.py', '*.ts', '*.tsx', '*.yaml', '*.yml', '*.md',
                      '*.toml', '*.json', '*.example', '*.sh', 'Dockerfile*')
        Get-ChildItem -Path $RootDir -Recurse -File -Include $includes -ErrorAction SilentlyContinue |
            Where-Object {
                $p = $_.FullName
                ($p -notmatch '[\\/]node_modules[\\/]') -and
                ($p -notmatch '[\\/]\.venv[\\/]')
            } |
            Select-String -Pattern $pattern -CaseSensitive |
            ForEach-Object { "$($_.Path):$($_.LineNumber):$($_.Line)" }
    }
    Write-Host ''
    Write-Host "Action: edit each line above before running 'values' / 'install'."
}

function Invoke-ValidateMigrations {
    $backendDir = Join-Path $RootDir 'backend'
    $defaultOut = Join-Path $RootDir 'migration.sql'
    $out = Get-OrDefault 'MIGRATION_SQL_OUT' $defaultOut

    if (-not (Test-Path -LiteralPath (Join-Path $backendDir 'alembic\versions') -PathType Container)) {
        Write-StdErr 'missing backend/alembic/versions'
        exit 1
    }

    # PKG_MANAGER toggle (uv by default, pip when set to 'pip'); pip is also the
    # fallback when uv is selected but absent.
    $pkgManager = Get-OrDefault 'PKG_MANAGER' 'uv'
    $hasUv      = [bool](Get-Command uv -ErrorAction SilentlyContinue)
    $useUv      = ($pkgManager -eq 'uv') -and $hasUv

    Push-Location $backendDir
    try {
        # env.py reads REMOTE_DB_URL but `--sql` only needs the dialect, so
        # supply a placeholder when the operator hasn't set one. A real value
        # is still honoured if exported.
        $dbUrl = Get-OrDefault 'REMOTE_DB_URL' 'postgresql+psycopg2://placeholder/skynet'
        $previousDbUrl = $env:REMOTE_DB_URL
        $env:REMOTE_DB_URL = $dbUrl
        try {
            if ($useUv) {
                # Materialize the lockfile-pinned env first. Without this we
                # fall through to whatever 'alembic' the system Python happens
                # to expose, which is how validate-migrations silently produced
                # an empty SQL file from a stale .venv. UV_PYTHON_DOWNLOADS=never
                # mirrors backend/Dockerfile so uv never reaches out to download
                # a Python build; if UV_INDEX_URL is set we pass it through so an
                # internal PyPI mirror satisfies the sync with no internet.
                $previousUvDownloads = $env:UV_PYTHON_DOWNLOADS
                $env:UV_PYTHON_DOWNLOADS = 'never'
                try {
                    & uv sync --frozen --quiet
                }
                finally {
                    $env:UV_PYTHON_DOWNLOADS = $previousUvDownloads
                }
                if ($LASTEXITCODE -ne 0) {
                    Write-StdErr 'uv sync --frozen failed; resolve dep conflicts before validate-migrations'
                    exit 1
                }
                & uv run alembic upgrade head --sql | Out-File -FilePath $out -Encoding utf8
            }
            else {
                # pip path: confirm migration deps import; if not, materialize
                # the pinned requirements.txt the same way backend/Dockerfile's
                # pip path does. alembic/env.py never imports ldap3, so we do not
                # require it — demanding it wrongly rejects otherwise-capable hosts.
                Test-Need 'python'
                Invoke-NativeProbe { python -c 'import alembic, sqlalchemy, pgvector' 2>$null }
                if ($LASTEXITCODE -ne 0) {
                    $pipArgs = @('-m', 'pip', 'install', '--quiet')
                    $pipIndexUrl    = Get-OrDefault 'PIP_INDEX_URL'    ''
                    $pipTrustedHost = Get-OrDefault 'PIP_TRUSTED_HOST' ''
                    if ($pipIndexUrl)    { $pipArgs += @('--index-url', $pipIndexUrl) }
                    if ($pipTrustedHost) { $pipArgs += @('--trusted-host', $pipTrustedHost) }
                    $pipArgs += @('-r', 'requirements.txt')
                    & python @pipArgs
                    if ($LASTEXITCODE -ne 0) {
                        Write-StdErr 'pip install -r requirements.txt failed; resolve dep conflicts before validate-migrations'
                        exit 1
                    }
                    Invoke-NativeProbe { python -c 'import alembic, sqlalchemy, pgvector' 2>$null }
                    if ($LASTEXITCODE -ne 0) {
                        Write-StdErr 'backend deps still missing after pip install; aborting'
                        exit 1
                    }
                }
                & python -m alembic upgrade head --sql | Out-File -FilePath $out -Encoding utf8
            }
        }
        finally {
            $env:REMOTE_DB_URL = $previousDbUrl
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $out) -or (Get-Item $out).Length -eq 0) {
        Write-StdErr "alembic produced an empty $out; aborting"
        exit 1
    }
    Write-Host "wrote offline migration SQL: $out"
    Write-Host 'review this file; the in-cluster migration Job will execute the same statements against REMOTE_DB_URL.'
}

function Invoke-BuildImages {
    $dockerBin = Get-OrDefault 'DOCKER' 'docker'
    Test-Need $dockerBin

    $registryPrefix  = Get-OrDefault 'REGISTRY_PREFIX'  'docker.io'
    $debianMirror    = Get-OrDefault 'DEBIAN_MIRROR'    ''
    $pipIndexUrl     = Get-OrDefault 'PIP_INDEX_URL'    ''
    $pipTrustedHost  = Get-OrDefault 'PIP_TRUSTED_HOST' ''
    $baseImage       = Get-OrDefault 'BASE_IMAGE'       ''
    $npmRegistry     = Get-OrDefault 'NPM_REGISTRY'     ''

    $backendArgs = @('build', (Join-Path $RootDir 'backend'),
                     '-t', "$REGISTRY/$BACKEND_REPOSITORY`:$IMAGE_TAG",
                     '--build-arg', "REGISTRY_PREFIX=$registryPrefix")
    if ($debianMirror)   { $backendArgs += @('--build-arg', "DEBIAN_MIRROR=$debianMirror") }
    if ($pipIndexUrl)    { $backendArgs += @('--build-arg', "PIP_INDEX_URL=$pipIndexUrl") }
    if ($pipTrustedHost) { $backendArgs += @('--build-arg', "PIP_TRUSTED_HOST=$pipTrustedHost") }

    $frontendArgs = @('build', (Join-Path $RootDir 'frontend'),
                      '-t', "$REGISTRY/$FRONTEND_REPOSITORY`:$IMAGE_TAG")
    if ($baseImage)   { $frontendArgs += @('--build-arg', "BASE_IMAGE=$baseImage") }
    if ($npmRegistry) { $frontendArgs += @('--build-arg', "NPM_REGISTRY=$npmRegistry") }

    $backendTag  = "$REGISTRY/$BACKEND_REPOSITORY`:$IMAGE_TAG"
    $frontendTag = "$REGISTRY/$FRONTEND_REPOSITORY`:$IMAGE_TAG"

    Write-Host "Building $backendTag"
    & $dockerBin @backendArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Building $frontendTag"
    & $dockerBin @frontendArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host 'Built:'
    Write-Host "  $backendTag"
    Write-Host "  $frontendTag"
}

function Invoke-PushImages {
    $dockerBin = Get-OrDefault 'DOCKER' 'docker'
    Test-Need $dockerBin
    $backendTag  = "$REGISTRY/$BACKEND_REPOSITORY`:$IMAGE_TAG"
    $frontendTag = "$REGISTRY/$FRONTEND_REPOSITORY`:$IMAGE_TAG"
    Write-StdErr "Reminder: '$dockerBin login $REGISTRY' (or 'podman login') must have succeeded or these pushes 401."
    & $dockerBin push $backendTag
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $dockerBin push $frontendTag
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host 'Pushed:'
    Write-Host "  $backendTag"
    Write-Host "  $frontendTag"
}

function Invoke-Values {
    $valuesDir = Split-Path -Parent $script:VALUES_OUT
    if (-not (Test-Path -LiteralPath $valuesDir)) {
        New-Item -ItemType Directory -Force -Path $valuesDir | Out-Null
    }

    $caBundlePath      = ''
    $caBackendEnv      = ''
    $caBackendMounts   = ''
    $caFrontendMounts  = ''
    $dbEgress          = ''
    $embeddingEgress   = ''
    $ldapEgress        = ''

    if (-not [string]::IsNullOrEmpty($script:INTERNAL_CA_SECRET)) {
        $caBundlePath = "$($script:INTERNAL_CA_MOUNT_DIR)/$($script:INTERNAL_CA_FILENAME)"
        $caBackendEnv = @"
    SSL_CERT_FILE: "$caBundlePath"
    REQUESTS_CA_BUNDLE: "$caBundlePath"
"@
        $caBackendMounts = @"
  extraVolumes:
    - name: internal-ca
      secret:
        secretName: "$($script:INTERNAL_CA_SECRET)"
  extraVolumeMounts:
    - name: internal-ca
      mountPath: "$($script:INTERNAL_CA_MOUNT_DIR)"
      readOnly: true
"@
        $caFrontendMounts = $caBackendMounts
    }

    if (-not [string]::IsNullOrEmpty($script:DB_EGRESS_CIDR)) {
        $dbEgress = "`n  # TODO: On-premise - the managed Postgres subnet. REQUIRED when`n  # postgres.enabled=false or the backend pool can never reach the DB.`n  dbEgress:`n    - cidr: `"$($script:DB_EGRESS_CIDR)`"`n      ports: [5432]"
    }

    if ((-not [string]::IsNullOrEmpty($script:EMBEDDING_EGRESS_CIDR)) -and
        ($script:EMBEDDING_EGRESS_CIDR -ne $script:LLM_EGRESS_CIDR)) {
        $embeddingEgress = "`n  # TODO: On-premise - use the exact embedding gateway CIDRs and ports.`n  embeddingEgress:`n    - cidr: `"$($script:EMBEDDING_EGRESS_CIDR)`"`n      ports: [443]"
    }

    if (-not [string]::IsNullOrEmpty($script:LDAP_EGRESS_CIDR)) {
        $ldapEgress = "`n  # TODO: On-premise - use the exact AD/LDAP controller CIDR and port (636 ldaps / 389 ldap).`n  ldapEgress:`n    - cidr: `"$($script:LDAP_EGRESS_CIDR)`"`n      ports: [$($script:LDAP_EGRESS_PORT)]"
    }

    $body = @"
# Generated by scripts/airgap_migrate.sh.
# TODO: On-premise - review every value before installing.
global:
  imageRegistry: "$($script:REGISTRY)"
  imagePullSecrets:
    - name: "$($script:PULL_SECRET)"

backend:
  image:
    repository: "$($script:BACKEND_REPOSITORY)"
    tag: "$($script:IMAGE_TAG)"
    pullPolicy: IfNotPresent
  env:
    # TODO: On-premise - point these at your OpenAI-compatible internal LLM gateway.
    CODE_AGENT_BASE_URL: "$($script:LLM_BASE_URL)"
    GENERALIST_AGENT_BASE_URL: "$($script:LLM_BASE_URL)"
    # TODO: On-premise - set to a model id your internal gateway actually serves (gpt-5 is a placeholder). LiteLLM forwards this id verbatim to CODE_AGENT_BASE_URL.
    CODE_AGENT_MODEL: "$($script:CODE_AGENT_MODEL)"
    GENERALIST_AGENT_MODEL: "$($script:GENERALIST_AGENT_MODEL)"
    # TAGGER_ASSIST_BASE_URL / TAGGER_ASSIST_MODEL are optional overrides for
    # the tagging assist; left unset it reuses the generalist pair above.
    # Explore search backend: lexical (vanilla Postgres, default) | bm25 (needs
    # the pg_search extension) | semantic (needs pgvector + the embedding gateway
    # below). Only "semantic" makes the migrate Job run CREATE EXTENSION vector.
    SEARCH_BACKEND: "$($script:SEARCH_BACKEND)"
    # Only used when SEARCH_BACKEND=semantic:
    EMBEDDINGS_BASE_URL: "$($script:EMBEDDING_BASE_URL)"
    EMBEDDINGS_MODEL: "$($script:EMBEDDING_MODEL)"
    # Air-gap: use the bundled litellm cost map; do not fetch from GitHub on import.
    LITELLM_LOCAL_MODEL_COST_MAP: "True"
$caBackendEnv
    # TODO: On-premise - set to the public frontend route.
    FRONTEND_URL: "https://$($script:FRONTEND_HOST)"
    # TODO: On-premise - list every browser origin allowed to call the backend.
    ALLOWED_ORIGINS: "https://$($script:FRONTEND_HOST)"
    ADMIN_GROUPS: "$($script:AUTH_ADMIN_GROUPS)"
    ADMIN_USERNAMES: "$($script:AUTH_ADMINS)"
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
    existingSecret: "$($script:BACKEND_SECRET)"
$caBackendMounts

frontend:
  image:
    repository: "$($script:FRONTEND_REPOSITORY)"
    tag: "$($script:IMAGE_TAG)"
    pullPolicy: IfNotPresent
  env:
    API_URL: "https://$($script:BACKEND_HOST)"
    # TODO: On-premise - point at your internal OIDC issuer.
    AUTH_SSO_ISSUER: "$($script:OIDC_ISSUER)"
    AUTH_SSO_CLIENT_ID: "$($script:OIDC_CLIENT_ID)"
    AUTH_SSO_SCOPE: "$($script:OIDC_SCOPE)"
    AUTH_GROUP_CLAIM: "$($script:AUTH_GROUP_CLAIM)"
    AUTH_ADMIN_GROUPS: "$($script:AUTH_ADMIN_GROUPS)"
    AUTH_ADMINS: "$($script:AUTH_ADMINS)"
    # TODO: On-premise - set when your IdP uses a private CA mounted in the pod.
    NODE_EXTRA_CA_CERTS: "$caBundlePath"
  secrets:
    # TODO: On-premise - must contain AUTH_SECRET, AUTH_SSO_CLIENT_SECRET, and BACKEND_AUTH_SECRET.
    existingSecret: "$($script:FRONTEND_SECRET)"
$caFrontendMounts

externalDatabase:
  enabled: true
  host: "$($script:EXTERNAL_DB_HOST)"
  port: 5432
  database: skynet
  user: skynet
  existingSecret: "$($script:EXTERNAL_DB_SECRET)"
  existingSecretKey: password
  composeUrl: true
  sslmode: require

postgres:
  enabled: false
  image:
    repository: "$($script:POSTGRES_REPOSITORY)"
    tag: pg16

openshift:
  routes:
    enabled: true
    backend:
      host: "$($script:BACKEND_HOST)"
    frontend:
      host: "$($script:FRONTEND_HOST)"

networkPolicy:
  enabled: true$dbEgress
  # TODO: On-premise - use the exact IdP / internal service CIDRs for your cluster.
  egressCidrs:
    - "$($script:IDP_EGRESS_CIDR)"
  # TODO: On-premise - use the exact LLM gateway CIDRs and ports.
  llmEgress:
    - cidr: "$($script:LLM_EGRESS_CIDR)"
      ports: [443]$embeddingEgress$ldapEgress

migration:
  enabled: true
  command: ["alembic", "upgrade", "head"]
"@

    # Write LF-only line endings so the values file diffs cleanly against the
    # bash-emitted version on operators who diff across both scripts.
    [System.IO.File]::WriteAllText($script:VALUES_OUT, ($body -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "wrote $($script:VALUES_OUT)"
}

function Invoke-Render {
    if (-not (Test-Path -LiteralPath $script:VALUES_OUT)) {
        Invoke-Values
    }
    & helm lint $ChartDir -f $script:VALUES_OUT
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $renderedPath = Join-Path $env:TEMP 'skynet-airgap-rendered.yaml'
    & helm template $script:RELEASE $ChartDir -n $script:NAMESPACE -f $script:VALUES_OUT |
        Out-File -FilePath $renderedPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "rendered manifest: $renderedPath"
}

function Invoke-Install {
    Test-Need 'helm'
    if (-not (Test-Path -LiteralPath $script:VALUES_OUT)) {
        Invoke-Values
    }
    & helm upgrade --install $script:RELEASE $ChartDir `
        --namespace $script:NAMESPACE `
        --create-namespace `
        -f $script:VALUES_OUT
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Status {
    Write-Host @"
Run:
  kubectl -n $($script:NAMESPACE) rollout status deploy/$($script:RELEASE)-skynet-backend
  kubectl -n $($script:NAMESPACE) rollout status deploy/$($script:RELEASE)-skynet-frontend
  kubectl -n $($script:NAMESPACE) logs job/$($script:RELEASE)-skynet-migrate

Smoke test:
  curl -k https://$($script:FRONTEND_HOST)/
  curl -k https://$($script:BACKEND_HOST)/health

ADFS / OIDC callback URL:
  https://$($script:FRONTEND_HOST)/api/auth/callback/adfs
"@
}

switch ($Command) {
    'configure'           { Invoke-Configure;          break }
    'todos'               { Invoke-Todos;              break }
    'check'               { Invoke-Check;              break }
    'validate-migrations' { Invoke-ValidateMigrations; break }
    'build-images'        { Invoke-BuildImages;        break }
    'push-images'         { Invoke-PushImages;         break }
    'values'              { Invoke-Values;             break }
    'render'              { Invoke-Render;             break }
    'install'             { Invoke-Install;            break }
    'status'              { Invoke-Status;             break }
    'all' {
        Write-StdErr "Note: 'all' does NOT build or push images — run build-images + push-images"
        Write-StdErr "(after 'docker login $REGISTRY') first, or install fails on ImagePullBackOff."
        Invoke-Check
        Invoke-ValidateMigrations
        Invoke-Values
        Invoke-Render
        Invoke-Install
        Invoke-Status
        break
    }
    { $_ -in @('-h', '--help', 'help', '') } {
        Write-Usage
        break
    }
    default {
        Write-Usage
        exit 2
    }
}
