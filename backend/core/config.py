"""Centralized configuration management via pydantic-settings.

Replaces scattered os.getenv() calls with a single Settings class that
validates environment variables at startup and provides typed access.
"""

from __future__ import annotations

import json
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Both agents (submit-wizard code agent + Cmd/Ctrl+J generalist) default
# to this one model id, so a single swap covers both; override per agent
# via CODE_AGENT_MODEL / GENERALIST_AGENT_MODEL.
#
# TODO: On-prem / air-gap — set this to whatever LiteLLM identifier your
# internal gateway exposes (e.g. "openai/<model>") and point
# CODE_AGENT_BASE_URL / GENERALIST_AGENT_BASE_URL at the gateway via env.
# The shipped default is OpenRouter's Auto Router Beta: OpenRouter is the
# platform's sole LLM provider, and the router picks a concrete model
# per request. Requires OPENROUTER_API_KEY in the env.
DEFAULT_AGENT_MODEL_ID = "openrouter/openrouter/auto-beta"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All settings have sensible defaults for local development.
    Production deployments should override via .env file or environment.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    remote_db_url: SecretStr | None = Field(default=None, description="PostgreSQL connection string for remote storage")

    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key for model access")
    openai_api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the OpenAI-compatible LLM gateway (LiteLLM reads the same "
            "env var for serving)."
        ),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Groq API key for dictation speech-to-text (Whisper large-v3-turbo "
            "on Groq LPUs) — the platform's sole STT provider. Unset disables "
            "dictation with a typed 503."
        ),
    )
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key for Claude models")

    stripe_secret_key: SecretStr | None = Field(
        default=None,
        alias="STRIPE_SECRET_KEY",
        description="Stripe secret API key (sk_test_… in test mode). Unset disables every billing mutation; reads still work.",
    )
    stripe_webhook_secret: SecretStr | None = Field(
        default=None,
        alias="STRIPE_WEBHOOK_SECRET",
        description="Stripe webhook signing secret (whsec_…) used to verify event payload authenticity.",
    )
    stripe_price_pack_starter: str = Field(
        default="", alias="STRIPE_PRICE_PACK_STARTER", description="Stripe price id for the 'starter' one-time credit pack."
    )
    stripe_price_pack_plus: str = Field(
        default="", alias="STRIPE_PRICE_PACK_PLUS", description="Stripe price id for the 'plus' one-time credit pack."
    )
    stripe_price_pack_pro: str = Field(
        default="", alias="STRIPE_PRICE_PACK_PRO", description="Stripe price id for the 'pro' one-time credit pack."
    )
    app_public_url: str = Field(
        default="http://localhost:3000",
        alias="APP_PUBLIC_URL",
        description="Public origin of the web app, used to build Stripe Checkout success/cancel return URLs.",
    )
    smtp_host: str | None = Field(
        default=None,
        alias="SMTP_HOST",
        description="SMTP relay host for outbound mail (email one-time sign-in codes). Unset disables email-based 2FA with a typed error.",
    )
    smtp_port: int = Field(default=587, alias="SMTP_PORT", description="SMTP relay port.")
    smtp_username: str | None = Field(
        default=None, alias="SMTP_USERNAME", description="SMTP auth username; unset sends unauthenticated."
    )
    smtp_password: SecretStr | None = Field(
        default=None, alias="SMTP_PASSWORD", description="SMTP auth password."
    )
    smtp_from: str | None = Field(
        default=None,
        alias="SMTP_FROM",
        description="From address for outbound mail; falls back to SMTP_USERNAME.",
    )
    smtp_starttls: bool = Field(
        default=True, alias="SMTP_STARTTLS", description="Upgrade the SMTP connection with STARTTLS."
    )
    webauthn_rp_id: str | None = Field(
        default=None,
        alias="WEBAUTHN_RP_ID",
        description="WebAuthn relying-party id (the site's registrable domain, e.g. 'skynet.example.com'). Unset derives it from APP_PUBLIC_URL's hostname.",
    )
    webauthn_origins: str = Field(
        default="",
        alias="WEBAUTHN_ORIGINS",
        description="Comma-separated browser origins accepted in passkey ceremonies. Unset allows APP_PUBLIC_URL plus the localhost dev origins.",
    )
    byok_vault_key: SecretStr | None = Field(
        default=None,
        alias="BYOK_VAULT_KEY",
        description="Fernet key (urlsafe base64, 32 bytes) that encrypts stored BYOK provider secrets at rest. Unset disables saving keys (the vault degrades to read-only); reads of already-stored masked metadata still work.",
    )
    litellm_proxy_url: str | None = Field(
        default=None,
        alias="LITELLM_PROXY_URL",
        description="Base URL of the self-hosted LiteLLM proxy that fronts managed inference (e.g. https://proxy.internal/v1). Unset (the default) routes managed runs directly to providers via process env keys; set it to flow all managed traffic through the metered gateway. BYOK runs always bypass the proxy and reach their provider directly.",
    )
    litellm_proxy_api_key: SecretStr | None = Field(
        default=None,
        alias="LITELLM_PROXY_API_KEY",
        description="Virtual key the backend presents to the LiteLLM proxy for managed runs. Only consulted when LITELLM_PROXY_URL is set.",
    )
    openrouter_provisioning_key: SecretStr | None = Field(
        default=None,
        alias="OPENROUTER_PROVISIONING_KEY",
        description="OpenRouter key-management (provisioning) API key. When set, managed runs authenticate with a per-user OpenRouter runtime key whose spend limit is synced to the account's credit balance before each dispatch, capping upstream spend at the provider itself. Unset (the default) sends managed runs through the shared gateway key. Requires BYOK_VAULT_KEY to encrypt the minted secrets at rest.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        description="OpenRouter master-account (inference) API key. Read-only here: the float monitor uses it to read the shared prepaid balance (GET /api/v1/credits) so it can warn when the OpenRouter float runs thin against outstanding credit liability. Managed inference itself routes through the LiteLLM proxy or per-user provisioned keys, not this field.",
    )
    openrouter_balance_floor_credits: int = Field(
        default=1000,
        alias="OPENROUTER_BALANCE_FLOOR_CREDITS",
        description="Low-water mark for the OpenRouter master-account balance, in credits (1 credit = 1 cent; default 1000 = $10). When the balance falls below this floor the float monitor logs a WARNING — native Auto Top-Up may have failed or demand is outrunning refills. 0 disables the monitor. Only consulted when OPENROUTER_API_KEY is set.",
    )
    worker_enabled: bool = Field(
        default=True,
        alias="WORKER_ENABLED",
        description=(
            "Start the in-process job worker alongside the API. Set false on "
            "API-only pods when job execution runs in dedicated worker "
            "replicas (worker_main.py)."
        ),
    )
    worker_threads: int = Field(
        default=4, ge=1, le=32, description="Number of concurrent worker threads", alias="WORKER_CONCURRENCY"
    )
    job_admission_max_memory_fraction: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Container memory usage fraction above which idle workers defer "
            "claiming new jobs (running jobs are unaffected); the job waits in "
            "the Postgres queue instead of OOM-killing the pod. 0 disables; "
            "also inert where no cgroup limit is readable (e.g. bare-metal dev)."
        ),
        alias="JOB_ADMISSION_MAX_MEMORY_FRACTION",
    )
    worker_poll_interval: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Seconds between queue polling cycles"
    )
    worker_stale_threshold: float = Field(
        default=600.0, ge=60.0, description="Seconds of inactivity before worker flagged as stuck"
    )
    job_max_attempts: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum orphan-recovery attempts before a job is marked failed",
        alias="JOB_MAX_ATTEMPTS",
    )
    orphan_sweep_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Seconds between periodic orphan-recovery sweeps (advisory-lock-gated)",
        alias="ORPHAN_SWEEP_INTERVAL",
    )
    stale_conversation_threshold_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        description="Days of inactivity before an unpinned agent conversation is auto-purged",
        alias="STALE_CONVERSATION_THRESHOLD_DAYS",
    )
    stale_conversation_sweep_interval_seconds: float = Field(
        default=86400.0,
        ge=300.0,
        description="Seconds between stale-conversation purge sweeps (advisory-lock-gated)",
        alias="STALE_CONVERSATION_SWEEP_INTERVAL",
    )
    staged_dataset_ttl_seconds: float = Field(
        default=600.0,
        ge=60.0,
        description="Seconds past a wizard staged-dataset's creation before it is auto-purged",
        alias="STAGED_DATASET_TTL_SECONDS",
    )
    staged_dataset_sweep_interval_seconds: float = Field(
        default=300.0,
        ge=60.0,
        description="Seconds between staged-dataset TTL purge sweeps (advisory-lock-gated)",
        alias="STAGED_DATASET_SWEEP_INTERVAL",
    )
    event_loop_lag_monitor_enabled: bool = Field(
        default=False,
        description=(
            "Enable an asyncio task that measures event-loop scheduling lag. "
            "Lag > threshold means a handler is doing sync CPU work and "
            "blocking the loop — diagnostic only, leave off in prod."
        ),
        alias="EVENT_LOOP_LAG_MONITOR",
    )
    telemetry_enabled: bool = Field(
        default=True,
        description=(
            "Accept first-party product-telemetry events at POST /telemetry/events. "
            "When off, ingestion is a silent no-op (events are dropped, never "
            "stored) — an ops kill switch for incident response or write-volume "
            "control. The read endpoints are unaffected."
        ),
        alias="TELEMETRY_ENABLED",
    )
    posthog_project_api_key: SecretStr | None = Field(
        default=None,
        alias="POSTHOG_PROJECT_API_KEY",
        description=(
            "PostHog project key for privacy-preserving export of accepted first-party telemetry. "
            "Unset keeps analytics entirely in Postgres."
        ),
    )
    posthog_host: str = Field(
        default="https://eu.i.posthog.com",
        alias="POSTHOG_HOST",
        description="PostHog event-ingestion origin; defaults to the EU cloud endpoint.",
    )
    sentry_dsn: SecretStr | None = Field(
        default=None,
        alias="SENTRY_DSN",
        description="Sentry DSN for backend and worker error reporting. Unset disables the SDK.",
    )
    sentry_environment: str = Field(
        default="development",
        alias="SENTRY_ENVIRONMENT",
        description="Deployment environment attached to Sentry events.",
    )
    sentry_traces_sample_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        alias="SENTRY_TRACES_SAMPLE_RATE",
        description="Fraction of backend performance traces exported to Sentry.",
    )
    event_loop_lag_threshold_ms: float = Field(
        default=100.0,
        ge=10.0,
        le=10000.0,
        description="Warn when event-loop lag exceeds this many milliseconds.",
        alias="EVENT_LOOP_LAG_THRESHOLD_MS",
    )
    progress_events_per_job_cap: int = Field(
        default=5000,
        ge=1,
        description="Maximum stored progress events per optimization job before old events are evicted",
    )
    log_entries_per_job_cap: int = Field(
        default=5000,
        ge=1,
        description="Maximum stored log entries per optimization job before old entries are evicted",
    )
    dataset_max_file_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        description="Per-file cap on compressed dataset bytes saved to a user's library",
    )
    user_storage_quota_bytes: int = Field(
        default=250 * 1024 * 1024,
        ge=1,
        description="Per-user unified storage budget in bytes across all of their Skynet data",
    )
    cancel_poll_interval: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Seconds between cancel signal checks"
    )
    lm_request_timeout_seconds: float = Field(
        default=600.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Per-request timeout (seconds) applied to every dspy.LM call so a stalled "
            "provider response can't wedge a run forever on a socket read. Overridable "
            "per-model via ModelConfig.extra['timeout']."
        ),
        alias="LM_REQUEST_TIMEOUT",
    )
    agent_request_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        description=(
            "Per-request LM timeout (seconds) for interactive agent turns (chat). "
            "Separate from lm_request_timeout_seconds, which is sized for batch "
            "optimization runs — a stalled provider should fail a chat turn in "
            "minutes, not tens of minutes."
        ),
        alias="AGENT_REQUEST_TIMEOUT",
    )
    job_stall_timeout_seconds: float = Field(
        default=1800.0,
        ge=0.0,
        description=(
            "Watchdog: fail a running optimization whose subprocess emits no "
            "progress/log/result event for this many seconds. Backstops a wedged child "
            "that the lease heartbeat would otherwise keep alive indefinitely. Keep it "
            "comfortably above the *retried* LM-call ceiling — build_language_model caps "
            "num_retries so (retries + 1) * lm_request_timeout_seconds stays under this "
            "with margin, so a hung call times out first; 0 disables the watchdog."
        ),
        alias="JOB_STALL_TIMEOUT",
    )
    gepa_eval_num_threads: int = Field(
        default=8,
        ge=1,
        le=64,
        description=(
            "Eval/rollout thread count injected into GEPA when the submission "
            "doesn't set num_threads. GEPA's own default is sequential "
            "candidate evaluation, and runs are LM-latency-bound, so parallel "
            "eval cuts wall-clock roughly linearly. An explicit user-supplied "
            "num_threads always wins; job_lm_max_concurrency bounds the "
            "multiplied total either way."
        ),
        alias="GEPA_EVAL_NUM_THREADS",
    )
    gepa_pxn_parents: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "GEPA PxN batched sampling — parent count (p). Each reflective "
            "iteration mutates p distinct parent candidates, drawing "
            "gepa_pxn_proposals (n) proposals from each, and evaluates all p*n "
            "as one batch. The default of 1 reproduces GEPA's classic "
            "single-mutation sampling; raising p (or n) above 1 switches GEPA to "
            "PxNSampling(p, n) — better wall-clock and generalization at higher "
            "LM cost, bounded per job by job_lm_max_concurrency. A submission "
            "that supplies its own gepa_kwargs.sampling_strategy always wins."
        ),
        alias="GEPA_PXN_PARENTS",
    )
    gepa_pxn_proposals: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "GEPA PxN batched sampling — proposals per parent (n); see "
            "gepa_pxn_parents. Raising n (or p) above 1 activates "
            "PxNSampling(p, n). Defaults to 1 (classic single-mutation GEPA)."
        ),
        alias="GEPA_PXN_PROPOSALS",
    )
    react_native_tool_calling: bool = Field(
        default=False,
        description=(
            "Route ReActV2 tool calls through the provider's native "
            "function-calling API instead of DSPy's text tool protocol. When "
            "on, the process installs a global ChatAdapter with "
            "use_native_function_calling=True; tools ride the provider's "
            "tools= parameter and turns come back as structured tool_calls "
            "(provider-default parallel calling included). The adapter is a "
            "no-op for signatures without a ToolCalls field, so only ReAct "
            "programs are affected. Off (default) keeps the text tool protocol "
            "that every model — including the flaky MiniMax student — parses "
            "reliably; only enable it for a deployment whose models all "
            "support native function calling."
        ),
        alias="REACT_NATIVE_TOOL_CALLING",
    )
    job_lm_max_concurrency: int = Field(
        default=16,
        ge=0,
        le=256,
        description=(
            "Per-job-child ceiling on concurrent LM calls. Grid pair threads "
            "times GEPA eval threads multiply, and a user can pass any "
            "num_threads in optimizer kwargs; this gate keeps one job from "
            "spraying the provider with enough parallel calls to trip rate "
            "limits and fail the run. 0 disables the gate."
        ),
        alias="JOB_LM_MAX_CONCURRENCY",
    )
    grid_pair_max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description=(
            "Concurrent (generation, reflection) pairs per grid-search job "
            "child. Total LM concurrency in the child is still capped by "
            "job_lm_max_concurrency regardless of this value."
        ),
        alias="GRID_PAIR_MAX_WORKERS",
    )
    grid_distributed_pairs: bool = Field(
        default=True,
        description=(
            "Fan a multi-pair grid search out as one claimable job row per "
            "pair so pairs spread across the whole worker fleet instead of "
            "sharing one child process. Off = classic all-pairs-in-one-child "
            "execution; flipping the flag never re-shapes a grid already in "
            "flight."
        ),
        alias="GRID_DISTRIBUTED_PAIRS",
    )
    job_run_start_method: Literal["fork", "spawn", "forkserver"] = Field(
        default="fork", description="Multiprocessing start method for job execution"
    )
    # Ceiling (pool_size + max_overflow = 20 + 20 = 40) is aligned with
    # Starlette's anyio threadpool width (40) so a burst of concurrent sync DB
    # handlers can't exhaust the pool and stall on pool_timeout. Tune via
    # DB_POOL_SIZE to keep total connections (× pod count) under the Postgres
    # max_connections budget.
    db_pool_size: int = Field(default=20, ge=1, le=200, description="SQLAlchemy pool size", alias="DB_POOL_SIZE")
    db_pool_max_overflow: int = Field(
        default=20,
        ge=0,
        le=200,
        description="SQLAlchemy max overflow connections",
        alias="DB_POOL_MAX_OVERFLOW",
    )
    db_pool_recycle_seconds: int = Field(
        default=3600,
        ge=60,
        description="Seconds before SQLAlchemy recycles a pooled DB connection",
        alias="DB_POOL_RECYCLE",
    )
    db_pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds SQLAlchemy waits for a free pool connection before erroring",
        alias="DB_POOL_TIMEOUT",
    )
    db_pgbouncer_transaction_mode: bool = Field(
        default=False,
        description="Disable driver features incompatible with PgBouncer transaction pooling",
        alias="DB_PGBOUNCER_TRANSACTION_MODE",
    )
    skynet_code_version: str = Field(
        default="",
        description="Build git SHA used to keep queued jobs on compatible workers",
        alias="SKYNET_CODE_VERSION",
    )
    require_code_version: bool = Field(
        default=False,
        description="Refuse to start if code_version can't be resolved (production safety)",
        alias="REQUIRE_CODE_VERSION",
    )

    artifacts_dir: str = Field(default="artifacts", description="Directory for storing optimized program artifacts")
    logs_dir: str = Field(default="logs", description="Directory for job execution logs")

    default_timeout: float = Field(default=30.0, ge=1.0, description="Standard request timeout in seconds")

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload on code changes (dev only)")

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        description="Comma-separated list of allowed CORS origins",
        alias="ALLOWED_ORIGINS",
    )

    # Air-gap deploys legitimately probe internal LiteLLM gateways on RFC1918
    # ranges, but the default has to fail closed: an unauthenticated caller
    # otherwise gets a /v1/models scan of the deploy network (incl. cloud
    # metadata services). Operators flip this on once their gateway address
    # range is well-defined.
    discover_allow_private: bool = Field(
        default=False,
        description=(
            "Allow POST /models/discover to probe link-local / RFC1918 / "
            "ULA / reserved / multicast addresses. Default off blocks "
            "SSRF-style scans of the deploy network. Loopback (127.0.0.0/8, "
            "::1) is always allowed so Ollama-on-localhost works without a "
            "flag. Cloud metadata service IPs (169.254.169.254 / "
            "fd00:ec2::254) are blocked unconditionally regardless."
        ),
    )
    program_cache_max_entries: int = Field(
        # 32 (was 128): each entry is a full deserialized program (MBs), the
        # cache lives in the fork-parent API process, and beta traffic serves
        # far fewer distinct programs than 32 at a time.
        default=32,
        ge=1,
        description=(
            "Maximum number of compiled DSPy programs held in the in-process "
            "LRU cache. Each entry is roughly the size of one optimized "
            "program; set to bound the API process resident set."
        ),
    )
    model_catalog_ttl_seconds: float = Field(
        default=1800.0,
        ge=0.0,
        description=(
            "How long the curated model catalog stays cached before "
            "re-evaluating provider key availability. 0 disables caching."
        ),
    )

    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    alert_webhook_url: str = Field(
        default="",
        alias="ALERT_WEBHOOK_URL",
        description=(
            "Incoming chat webhook (Slack-compatible {\"text\": …} payload; also "
            "accepted by Mattermost and Google Chat) that receives operational "
            "alerts — unhandled 500s, a dead worker, and code paths that call "
            "send_alert() directly. Unset (the default) disables outbound "
            "alerting: records still reach the logs, they just aren't forwarded."
        ),
    )
    alert_min_level: str = Field(
        default="ERROR",
        alias="ALERT_MIN_LEVEL",
        description=(
            "Minimum log level forwarded to ALERT_WEBHOOK_URL by the log handler "
            "(DEBUG/INFO/WARNING/ERROR/CRITICAL). Effective only at or above "
            "LOG_LEVEL, which gates the root logger first. Direct send_alert() "
            "calls ignore this threshold."
        ),
    )
    alert_environment: str = Field(
        default="",
        alias="ALERT_ENVIRONMENT",
        description=(
            "Short label prefixed to every alert (e.g. 'prod', 'staging') so a "
            "shared channel can attribute an alert to its source. Empty omits "
            "the prefix."
        ),
    )
    alert_throttle_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="ALERT_THROTTLE_SECONDS",
        description=(
            "Cooldown during which an identical alert (same level, title and "
            "body) is forwarded at most once, so an error loop can't flood the "
            "webhook. 0 disables throttling."
        ),
    )

    @field_validator("alert_min_level")
    @classmethod
    def _validate_alert_min_level(cls, v: str) -> str:
        """Normalise ALERT_MIN_LEVEL to a canonical upper-case logging level name.

        Args:
            v: Raw ALERT_MIN_LEVEL value from the environment.

        Returns:
            The upper-cased level name.

        Raises:
            ValueError: When the value is not a standard logging level name.
        """
        name = v.strip().upper()
        if name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("ALERT_MIN_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return name

    code_agent_model: str = Field(
        default=DEFAULT_AGENT_MODEL_ID,
        description=(
            "LiteLLM model id used by the submit-wizard code agent. "
            "Defaults to DEFAULT_AGENT_MODEL_ID (OpenRouter's Auto Router); "
            "override via CODE_AGENT_MODEL for on-prem deployments."
        ),
    )
    # TODO: On-prem / air-gap — set CODE_AGENT_BASE_URL to your internal
    # OpenAI-compatible gateway (e.g. https://llm.your-company.com/v1) so the
    # agent stops trying to reach the public OpenRouter endpoint.
    code_agent_base_url: str = Field(
        default="",
        description="Optional custom base URL for the code agent LM (e.g. internal OpenAI-compatible gateway)",
    )

    generalist_agent_mcp_url: str = Field(
        default="",
        description=(
            "URL of the MCP server the generalist agent connects to (usually the "
            "same app's /mcp mount). When unset it is derived from HOST/PORT so "
            "changing the API port doesn't strand the agent on a dead :8000."
        ),
    )
    generalist_agent_model: str = Field(
        default=DEFAULT_AGENT_MODEL_ID,
        description=(
            "LiteLLM model id used by the generalist agent (Cmd/Ctrl+J "
            "panel). Defaults to DEFAULT_AGENT_MODEL_ID (OpenRouter's Auto "
            "Router, which picks a concrete model per request)."
        ),
    )
    # TODO: On-prem / air-gap — set GENERALIST_AGENT_BASE_URL to your internal
    # OpenAI-compatible gateway. The default empty string lets LiteLLM choose
    # the public OpenRouter endpoint, which an air-gapped host cannot reach.
    generalist_agent_base_url: str = Field(
        default="",
        description="Optional custom base URL for the generalist agent LM (e.g. internal OpenAI-compatible gateway)",
    )

    tagger_assist_model: str = Field(
        default="",
        description=(
            "LiteLLM model id used by the tagger's AI co-tagging assist "
            "(interview, calibration predictions, auto-tagging). Empty falls "
            "back to generalist_agent_model."
        ),
    )
    tagger_assist_base_url: str = Field(
        default="",
        description=(
            "Optional custom base URL for the tagging-assist LM. Empty falls "
            "back to generalist_agent_base_url."
        ),
    )
    # TODO: On-prem / air-gap — point this at an internal OpenAI-compatible
    # embeddings endpoint (usually the same gateway family as CODE_AGENT_BASE_URL).
    # The backend sends POST {base_url}/embeddings with {model, input}; no model
    # weights are bundled in this repo.
    embeddings_base_url: str = Field(
        default="",
        description="Internal OpenAI-compatible embedding API base URL, e.g. https://llm.internal/v1",
    )
    embeddings_model: str = Field(
        default="jina-embeddings-v4",
        description=(
            "Embedding model id exposed by embeddings_base_url. "
            "The model must return at least embeddings_dim values. "
            "Jina v4 is multilingual (89 languages incl. Hebrew) and supports "
            "asymmetric retrieval LoRA adapters via the request ``task`` field — "
            "the gateway passes ``retrieval.query`` for searches and "
            "``retrieval.passage`` for indexed summaries so Hebrew↔English and "
            "same-language pairs all score correctly."
        ),
    )
    embeddings_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Optional bearer token for the embedding API. Falls back to OPENAI_API_KEY "
            "when unset so a shared internal gateway secret can be reused."
        ),
    )
    embeddings_dim: int = Field(
        default=512,
        ge=64,
        le=2048,
        description=(
            "Head-truncated dimension stored in job_embeddings.embedding_summary. "
            "Must match the schema; changing requires a migration."
        ),
    )
    # TODO: On-prem / air-gap — leave EMBEDDINGS_SUMMARY_MODEL empty so the
    # pipeline reuses code_agent_model (which already points at your internal
    # gateway). Setting this to a public-provider id would route summarisation
    # calls outside the air-gap.
    embeddings_summary_model: str = Field(
        default="",
        description=(
            "LiteLLM model id used to summarise a finished job before embedding. "
            "Falls back to code_agent_model when empty."
        ),
    )
    embedding_index_sweep_interval_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=3600.0,
        description=(
            "Seconds between bounded repair passes for missing or stale Explore "
            "embeddings."
        ),
        alias="EMBEDDING_INDEX_SWEEP_INTERVAL",
    )
    embedding_index_sweep_batch_size: int = Field(
        default=25,
        ge=1,
        le=500,
        description=(
            "Maximum number of successful jobs re-indexed during one Explore "
            "embedding repair pass."
        ),
        alias="EMBEDDING_INDEX_SWEEP_BATCH_SIZE",
    )
    search_backend: Literal["lexical", "bm25", "semantic"] = Field(
        default="lexical",
        alias="SEARCH_BACKEND",
        description=(
            "Explore search engine — the single knob that also decides which "
            "Postgres extension (if any) the deployment requires:\n"
            "  lexical  (default): vanilla Postgres, no extension. ILIKE substring "
            "search. The safe choice for an air-gapped database that can't add "
            "extensions — neither the migrate Job nor the app runs CREATE EXTENSION.\n"
            "  bm25: requires the pg_search extension. Ranks lexical search with "
            "BM25 relevance; degrades to ILIKE when pg_search is absent.\n"
            "  semantic: requires the pgvector extension. Embedding (vector) "
            "search; also enables the embedding pipeline and the job_embeddings "
            "migration schema, so only this profile runs CREATE EXTENSION vector. "
            "Set EMBEDDINGS_BASE_URL/MODEL alongside it.\n"
            "Accepts the synonyms vanilla/ilike->lexical, pg_search/paradedb->bm25, "
            "embeddings/vector/pgvector->semantic."
        ),
    )
    # Derived from search_backend by _resolve_search_backend below — SEARCH_BACKEND
    # is the only switch operators set. Kept as plain fields (not properties) so the
    # test suite can patch them per-case; any EMBEDDINGS_ENABLED / SEARCH_BM25_ENABLED
    # left in the environment is overridden by the value derived from SEARCH_BACKEND.
    embeddings_enabled: bool = Field(default=False)
    search_bm25_enabled: bool = Field(default=False)

    @field_validator("search_backend", mode="before")
    @classmethod
    def _normalize_search_backend(cls, value: object) -> object:
        """Trim, lower-case and map synonyms for SEARCH_BACKEND before validation.

        Lets operators write 'vanilla' / 'pgvector' / ' BM25 ' and still land on
        one of the canonical lexical/bm25/semantic values the Literal accepts.

        Args:
            value: Raw SEARCH_BACKEND input (string from env, or anything else).

        Returns:
            The canonical backend name when a string synonym matches, otherwise
            the value unchanged for the Literal validator to accept or reject.
        """
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        synonyms = {
            "": "lexical",
            "vanilla": "lexical",
            "ilike": "lexical",
            "none": "lexical",
            "off": "lexical",
            "pg_search": "bm25",
            "pgsearch": "bm25",
            "paradedb": "bm25",
            "embeddings": "semantic",
            "embedding": "semantic",
            "vector": "semantic",
            "pgvector": "semantic",
        }
        return synonyms.get(normalized, normalized)

    @model_validator(mode="after")
    def _resolve_search_backend(self) -> Settings:
        """Derive the embedding / BM25 flags from the single SEARCH_BACKEND knob.

        SEARCH_BACKEND is authoritative so the three profiles stay mutually
        exclusive: ``embeddings_enabled`` (pgvector + embedding pipeline + the
        job_embeddings migration schema) is on only for 'semantic', and
        ``search_bm25_enabled`` (pg_search ranking) only for 'bm25'. 'lexical'
        leaves both off, so neither the migrate Job nor the running app issues a
        CREATE EXTENSION against a vanilla Postgres.

        Returns:
            This settings instance with the derived flags applied.
        """
        self.embeddings_enabled = self.search_backend == "semantic"
        self.search_bm25_enabled = self.search_backend == "bm25"
        return self

    max_jobs_per_user: int = Field(default=100, ge=1, description="Default per-user job cap")
    max_total_users: int = Field(
        default=0,
        ge=0,
        description=(
            "Hard cap on total registered accounts; sign-ups are refused at or above it "
            "to bound platform cost. 0 disables the cap."
        ),
    )
    max_monthly_active_users: int = Field(
        default=0,
        ge=0,
        description=(
            "Hard cap on distinct non-admin identities admitted during each UTC calendar "
            "month. Users already admitted that month continue normally; new identities are "
            "refused once the cap is reached. 0 disables the cap."
        ),
    )
    max_concurrent_jobs_per_user: int = Field(
        default=5,
        ge=0,
        description=(
            "Cap on a single user's concurrently active runs (pending/validating/running/"
            "paused); further submissions are refused until one finishes. 0 disables the cap."
        ),
    )
    global_daily_spend_ceiling_credits: int = Field(
        default=0,
        ge=0,
        description=(
            "Platform-wide credit-spend backstop over a trailing 24h; new submissions are "
            "refused once reached. 0 disables the kill-switch (per-user credit gate still applies)."
        ),
    )
    submissions_paused: bool = Field(
        default=False,
        description=(
            "Operator kill-switch: when true, all new run/grid-search submissions are refused "
            "(in-flight runs continue). An instant emergency brake independent of spend."
        ),
    )
    redis_url: str | None = Field(
        default=None,
        description=(
            "Redis connection URL backing the cross-replica rate limiter and login lockout. "
            "When unset the limiters fail open (allow) and the login lockout falls back to "
            "per-process in-memory state — correct for a single backend instance."
        ),
    )
    rate_limit_submissions_per_minute: int = Field(
        default=30,
        ge=0,
        description=(
            "Per-account cap on run/grid-search submissions per rolling minute, enforced across "
            "replicas via Redis. Bounds a runaway script or abusive key. 0 disables the cap."
        ),
    )
    rate_limit_account_requests_per_hour: int = Field(
        default=20,
        ge=0,
        description=(
            "Per-email cap on account-action requests (register, password-reset, email-verify) "
            "per rolling hour, enforced across replicas via Redis. 0 disables the cap."
        ),
    )
    backend_auth_secret: SecretStr | None = Field(
        default=None,
        description="Shared HS256 secret used by the frontend to sign backend API tokens",
    )
    admin_usernames: str = Field(
        default="",
        description="Break-glass comma-separated usernames that grant backend admin access",
    )
    admin_groups: str = Field(
        default="",
        description="Comma-separated IdP groups that grant backend admin access",
    )
    quota_overrides_json: str = Field(
        default="{}",
        description='Per-user quota overrides as JSON, e.g. \'{"power_user": 500, "researcher": null}\'',
        alias="QUOTA_OVERRIDES",
    )

    @field_validator("quota_overrides_json")
    @classmethod
    def _validate_quota_overrides_json(cls, v: str) -> str:
        """Validate that QUOTA_OVERRIDES is a JSON object of {username: int|null}.

        Args:
            v: Raw env value (a JSON string).

        Returns:
            The validated JSON string with lowercase keys, normalised to ``"{}"``
            when blank. Lower-casing here keeps the wire-level representation
            stable across reads of ``quota_overrides_json`` and makes lookups
            in ``get_user_quota`` cheap.

        Raises:
            ValueError: When the JSON is malformed, not an object, or contains
                values that are not ``int`` or ``null``.
        """
        if not v.strip():
            return "{}"
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"QUOTA_OVERRIDES is not valid JSON: {exc}") from exc
        # Pydantic field_validator surfaces only ValueError as a ValidationError,
        # so keep the type-check failures as ValueError despite TRY004 preferring
        # TypeError for type errors.
        if not isinstance(parsed, dict):
            raise ValueError(  # noqa: TRY004
                "QUOTA_OVERRIDES must be a JSON object mapping usernames to int|null"
            )
        normalised: dict[str, int | None] = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                raise ValueError(  # noqa: TRY004
                    f"QUOTA_OVERRIDES keys must be strings, got {type(key).__name__}"
                )
            # bool is an int subclass, so reject it explicitly to avoid silently
            # treating ``true``/``false`` as quota 1/0.
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"QUOTA_OVERRIDES['{key}'] must be int or null, got {type(value).__name__}")
            normalised[key.strip().lower()] = value
        return json.dumps(normalised)

    @model_validator(mode="after")
    def _derive_generalist_mcp_url(self) -> Settings:
        """Fill an unset generalist MCP URL from the server's own HOST/PORT.

        Hardcoding ``http://localhost:8000/mcp/`` strands the agent on a dead
        port whenever the API runs on a non-default PORT, surfacing a raw network
        error instead of a config diagnostic. Deriving the default keeps the
        agent pointed at this app's own ``/mcp`` mount; a bind-all ``0.0.0.0``
        host is dialled as ``127.0.0.1`` — the IPv4 literal, not ``localhost``,
        because some container runtimes (e.g. Railway) resolve ``localhost`` to
        ``::1`` while Uvicorn listens on IPv4 only, turning the self-dial into
        a raw ``ConnectError``.

        Returns:
            The settings instance with ``generalist_agent_mcp_url`` populated.
        """
        if not self.generalist_agent_mcp_url:
            host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
            self.generalist_agent_mcp_url = f"http://{host}:{self.port}/mcp/"
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Return ``cors_origins`` parsed into a list of trimmed, non-empty origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def admin_usernames_set(self) -> frozenset[str]:
        """Return break-glass admin usernames as a lowercase frozenset."""
        return frozenset(s.strip().lower() for s in self.admin_usernames.split(",") if s.strip())

    @property
    def admin_groups_set(self) -> frozenset[str]:
        """Return admin IdP groups as a lowercase frozenset."""
        return frozenset(s.strip().lower() for s in self.admin_groups.split(",") if s.strip())

    @property
    def is_stripe_configured(self) -> bool:
        """Return whether a Stripe secret key is present (billing mutations enabled)."""
        return self.stripe_secret_key is not None

    @property
    def is_byok_vault_configured(self) -> bool:
        """Return whether a BYOK vault key is present (saving provider keys enabled)."""
        return self.byok_vault_key is not None

    @property
    def stripe_pack_price_ids(self) -> dict[str, str]:
        """Return the one-time credit-pack Stripe price ids keyed by pack id."""
        return {
            "starter": self.stripe_price_pack_starter,
            "plus": self.stripe_price_pack_plus,
            "pro": self.stripe_price_pack_pro,
        }

    @cached_property
    def quota_overrides(self) -> dict[str, int | None]:
        """Return parsed quota overrides keyed by lowercase username.

        Cached because every job-submission goes through ``get_user_quota`` and
        re-parsing JSON on each call is wasteful; the validator already
        normalises the JSON to lowercase keys.
        """
        return json.loads(self.quota_overrides_json)

    @cached_property
    def code_version(self) -> str:
        """Return the build version used for job-claim compatibility checks.

        Resolution order: ``SKYNET_CODE_VERSION`` env, then a local git lookup,
        then the literal ``"unknown"``. With ``REQUIRE_CODE_VERSION=true``,
        falling through to ``"unknown"`` raises — production builds must bake
        the SHA into the image so staged rollouts don't have multiple pods
        claiming each other's ``"unknown"``-tagged work.

        Raises:
            RuntimeError: When the fallback resolves to ``"unknown"`` and
                ``require_code_version`` is enabled.
        """
        configured = self.skynet_code_version.strip()
        if configured:
            return configured[:40]
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
            resolved = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            resolved = ""
        if resolved:
            return resolved[:40]
        if self.require_code_version:
            raise RuntimeError(
                "SKYNET_CODE_VERSION is unset and no git checkout is available — "
                "set REQUIRE_CODE_VERSION=false for local dev, or bake the SHA "
                "into the image (Dockerfile passes GIT_SHA build arg).",
            )
        return "unknown"

    def get_user_quota(self, username: str) -> int | None:
        """Return the effective job quota for a user.

        Users with a ``null`` override receive ``None`` (unlimited). Per-user
        overrides in ``quota_overrides_json`` take precedence over
        ``max_jobs_per_user``. Lookup is case-insensitive to match how override
        keys are stored.

        Args:
            username: The username to look up.

        Returns:
            Maximum number of allowed jobs, or ``None`` for unlimited access.
        """
        normalised = (username or "").strip().lower()
        if normalised in self.quota_overrides:
            return self.quota_overrides[normalised]
        return self.max_jobs_per_user


settings = Settings()


def embeddings_schema_enabled() -> bool:
    """Report whether migrations should manage the pgvector embedding schema.

    True only for the semantic search backend. The lexical and bm25 backends run
    on a vanilla Postgres with no pgvector extension, so the baseline and every
    downstream embedding migration skip the ``job_embeddings`` table, its Vector
    columns and the HNSW indexes — which is what keeps the migrate Job from
    issuing ``CREATE EXTENSION vector`` on a database that doesn't have it. The
    migrate Job inherits SEARCH_BACKEND from the backend ConfigMap, so this reads
    the same value the application pods do.

    Returns:
        True when SEARCH_BACKEND selects semantic search, else False.
    """
    return settings.embeddings_enabled
