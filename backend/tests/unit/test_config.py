"""Tests for core.config.Settings and its helpers."""

# pydantic-settings BaseSettings.__init__ accepts ``_env_file`` (and other
# leading-underscore kwargs) that mypy can't see without the dedicated plugin,
# so disable ``call-arg`` for this file.
# mypy: disable-error-code="call-arg"

from __future__ import annotations

import pytest

from core.config import DEFAULT_AGENT_MODEL_ID, Settings

_SETTINGS_ENV_VARS = (
    "REMOTE_DB_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "WORKER_CONCURRENCY",
    "WORKER_POLL_INTERVAL",
    "WORKER_STALE_THRESHOLD",
    "JOB_MAX_ATTEMPTS",
    "EMBEDDING_INDEX_SWEEP_INTERVAL",
    "EMBEDDING_INDEX_SWEEP_BATCH_SIZE",
    "PROGRESS_EVENTS_PER_JOB_CAP",
    "LOG_ENTRIES_PER_JOB_CAP",
    "CANCEL_POLL_INTERVAL",
    "JOB_RUN_START_METHOD",
    "DB_POOL_SIZE",
    "DB_POOL_MAX_OVERFLOW",
    "DB_POOL_RECYCLE",
    "DB_PGBOUNCER_TRANSACTION_MODE",
    "SKYNET_CODE_VERSION",
    "ARTIFACTS_DIR",
    "LOGS_DIR",
    "DEFAULT_TIMEOUT",
    "HOST",
    "PORT",
    "RELOAD",
    "ALLOWED_ORIGINS",
    "LOG_LEVEL",
    "ADMIN_USERNAMES",
)


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear Settings-relevant env vars so every test starts from pure defaults.

    pydantic-settings reads ``os.environ`` even when ``_env_file=None``, so any
    value exported in the developer shell (or loaded from ``backend/.env``
    earlier in the process) would otherwise leak into default-value tests.
    """
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)


def test_settings_defaults_worker_threads() -> None:
    """Default ``worker_threads`` is 4 when no env var is set."""
    s = Settings(_env_file=None)

    assert s.worker_threads == 4


def test_settings_defaults_worker_poll_interval() -> None:
    """Default ``worker_poll_interval`` is 1.0 second."""
    s = Settings(_env_file=None)

    assert s.worker_poll_interval == 1.0


def test_settings_defaults_worker_stale_threshold() -> None:
    """Default ``worker_stale_threshold`` is 600.0 seconds."""
    s = Settings(_env_file=None)

    assert s.worker_stale_threshold == 600.0


def test_settings_defaults_storage_retention_caps() -> None:
    """Default storage retention caps are 5000 rows per job."""
    s = Settings(_env_file=None)

    assert s.progress_events_per_job_cap == 5000
    assert s.log_entries_per_job_cap == 5000


def test_settings_defaults_job_max_attempts() -> None:
    """Default job retry cap is 3 attempts."""
    s = Settings(_env_file=None)

    assert s.job_max_attempts == 3


def test_settings_defaults_embedding_index_repair() -> None:
    """Embedding repair defaults to a one-minute interval and 25-row batch."""
    s = Settings(_env_file=None)

    assert s.embedding_index_sweep_interval_seconds == 60.0
    assert s.embedding_index_sweep_batch_size == 25


def test_settings_defaults_cancel_poll_interval() -> None:
    """Default ``cancel_poll_interval`` is 1.0 second."""
    s = Settings(_env_file=None)

    assert s.cancel_poll_interval == 1.0


def test_settings_defaults_job_run_start_method() -> None:
    """Default ``job_run_start_method`` is ``"fork"``."""
    s = Settings(_env_file=None)

    assert s.job_run_start_method == "fork"


def test_settings_defaults_db_pool_config() -> None:
    """Default DB pool settings are production-safe single-pod values."""
    s = Settings(_env_file=None)

    assert s.db_pool_size == 20
    assert s.db_pool_max_overflow == 20
    assert s.db_pool_recycle_seconds == 3600
    assert s.db_pgbouncer_transaction_mode is False


def test_settings_defaults_artifacts_dir() -> None:
    """Default ``artifacts_dir`` is ``"artifacts"``."""
    s = Settings(_env_file=None)

    assert s.artifacts_dir == "artifacts"


def test_settings_defaults_logs_dir() -> None:
    """Default ``logs_dir`` is ``"logs"``."""
    s = Settings(_env_file=None)

    assert s.logs_dir == "logs"


def test_settings_defaults_host() -> None:
    """Default ``host`` is ``"0.0.0.0"``."""
    s = Settings(_env_file=None)

    assert s.host == "0.0.0.0"


def test_settings_defaults_port() -> None:
    """Default ``port`` is 8000."""
    s = Settings(_env_file=None)

    assert s.port == 8000


def test_settings_defaults_reload() -> None:
    """Default ``reload`` is ``False``."""
    s = Settings(_env_file=None)

    assert s.reload is False


def test_settings_defaults_log_level() -> None:
    """Default ``log_level`` is ``"INFO"``."""
    s = Settings(_env_file=None)

    assert s.log_level == "INFO"


def test_settings_defaults_api_keys_are_none() -> None:
    """API key fields default to ``None`` when no env vars are exported."""
    s = Settings(_env_file=None)

    assert s.openai_api_key is None
    assert s.anthropic_api_key is None
    assert s.remote_db_url is None


def test_settings_env_override_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WORKER_CONCURRENCY`` env var overrides ``worker_threads``."""
    monkeypatch.setenv("WORKER_CONCURRENCY", "8")

    s = Settings(_env_file=None)

    assert s.worker_threads == 8


def test_settings_env_override_storage_retention_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage retention caps are configurable via environment."""
    monkeypatch.setenv("PROGRESS_EVENTS_PER_JOB_CAP", "123")
    monkeypatch.setenv("LOG_ENTRIES_PER_JOB_CAP", "456")

    s = Settings(_env_file=None)

    assert s.progress_events_per_job_cap == 123
    assert s.log_entries_per_job_cap == 456


def test_settings_env_override_job_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``JOB_MAX_ATTEMPTS`` env var overrides the retry cap."""
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "5")

    s = Settings(_env_file=None)

    assert s.job_max_attempts == 5


def test_settings_env_override_db_pool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """DB pool settings are configurable via environment."""
    monkeypatch.setenv("DB_POOL_SIZE", "12")
    monkeypatch.setenv("DB_POOL_MAX_OVERFLOW", "8")
    monkeypatch.setenv("DB_POOL_RECYCLE", "900")
    monkeypatch.setenv("DB_PGBOUNCER_TRANSACTION_MODE", "true")

    s = Settings(_env_file=None)

    assert s.db_pool_size == 12
    assert s.db_pool_max_overflow == 8
    assert s.db_pool_recycle_seconds == 900
    assert s.db_pgbouncer_transaction_mode is True


def test_settings_code_version_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SKYNET_CODE_VERSION`` provides the cached build version."""
    monkeypatch.setenv("SKYNET_CODE_VERSION", "abcdef123456")

    s = Settings(_env_file=None)

    assert s.code_version == "abcdef123456"


def test_settings_env_override_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PORT`` env var overrides ``port``."""
    monkeypatch.setenv("PORT", "9090")

    s = Settings(_env_file=None)

    assert s.port == 9090


def test_settings_env_override_reload_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RELOAD=true`` parses to ``reload is True``."""
    monkeypatch.setenv("RELOAD", "true")

    s = Settings(_env_file=None)

    assert s.reload is True


def test_settings_env_override_reload_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RELOAD=false`` parses to ``reload is False``."""
    monkeypatch.setenv("RELOAD", "false")

    s = Settings(_env_file=None)

    assert s.reload is False


def test_settings_env_override_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOG_LEVEL`` env var overrides ``log_level``."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    s = Settings(_env_file=None)

    assert s.log_level == "DEBUG"


def test_settings_env_override_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ALLOWED_ORIGINS`` env var populates ``cors_origins`` verbatim."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://example.com,http://other.com")

    s = Settings(_env_file=None)

    assert s.cors_origins == "http://example.com,http://other.com"


def test_settings_env_override_admin_usernames(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ADMIN_USERNAMES`` env var populates ``admin_usernames`` verbatim."""
    monkeypatch.setenv("ADMIN_USERNAMES", "alice,bob")

    s = Settings(_env_file=None)

    assert s.admin_usernames == "alice,bob"


def test_settings_env_override_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lowercase env var names still resolve thanks to ``case_sensitive=False``."""
    monkeypatch.setenv("port", "7777")

    s = Settings(_env_file=None)

    assert s.port == 7777


def test_cors_origins_list_parses_defaults() -> None:
    """``cors_origins_list`` parses the default CSV into the two dev origins."""
    s = Settings(_env_file=None)

    result = s.cors_origins_list

    assert result == ["http://localhost:3000", "http://localhost:3001"]


def test_cors_origins_list_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cors_origins_list`` trims surrounding whitespace from each entry."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "  http://a.com ,  http://b.com  ")

    s = Settings(_env_file=None)

    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_cors_origins_list_skips_empty_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cors_origins_list`` drops empty CSV entries."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com,,http://b.com,")

    s = Settings(_env_file=None)

    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_cors_origins_list_single_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cors_origins_list`` correctly handles a single-origin CSV."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://only.com")

    s = Settings(_env_file=None)

    assert s.cors_origins_list == ["http://only.com"]


def test_cors_origins_list_empty_string_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``ALLOWED_ORIGINS`` env var yields an empty list."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "")

    s = Settings(_env_file=None)

    assert s.cors_origins_list == []


def test_admin_usernames_set_empty_by_default() -> None:
    """``admin_usernames_set`` is empty when no env var is exported."""
    s = Settings(_env_file=None)

    assert s.admin_usernames_set == frozenset()


def test_admin_usernames_set_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """``admin_usernames_set`` parses a CSV into a frozenset."""
    monkeypatch.setenv("ADMIN_USERNAMES", "alice,bob,carol")

    s = Settings(_env_file=None)

    assert s.admin_usernames_set == frozenset({"alice", "bob", "carol"})


def test_admin_usernames_set_lowercases(monkeypatch: pytest.MonkeyPatch) -> None:
    """``admin_usernames_set`` lower-cases each entry."""
    monkeypatch.setenv("ADMIN_USERNAMES", "Alice,BOB")

    s = Settings(_env_file=None)

    assert s.admin_usernames_set == frozenset({"alice", "bob"})


def test_admin_usernames_set_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``admin_usernames_set`` trims whitespace around each entry."""
    monkeypatch.setenv("ADMIN_USERNAMES", " alice , bob ")

    s = Settings(_env_file=None)

    assert s.admin_usernames_set == frozenset({"alice", "bob"})


def test_admin_usernames_set_skips_empty_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """``admin_usernames_set`` drops empty CSV entries."""
    monkeypatch.setenv("ADMIN_USERNAMES", "alice,,bob,")

    s = Settings(_env_file=None)

    assert s.admin_usernames_set == frozenset({"alice", "bob"})


def test_settings_coerces_string_int_for_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PORT`` env var is coerced from string to int."""
    monkeypatch.setenv("PORT", "8080")

    s = Settings(_env_file=None)

    assert isinstance(s.port, int)
    assert s.port == 8080


def test_settings_coerces_string_bool_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RELOAD=1`` is coerced to ``bool``-typed ``True``."""
    monkeypatch.setenv("RELOAD", "1")

    s = Settings(_env_file=None)

    assert isinstance(s.reload, bool)
    assert s.reload is True


def test_settings_coerces_string_float_poll_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WORKER_POLL_INTERVAL`` is coerced from string to float."""
    monkeypatch.setenv("WORKER_POLL_INTERVAL", "2.5")

    s = Settings(_env_file=None)

    assert isinstance(s.worker_poll_interval, float)
    assert s.worker_poll_interval == 2.5


def test_settings_default_agent_models_use_shared_constant() -> None:
    """Both agents default to ``DEFAULT_AGENT_MODEL_ID`` so a single swap covers both."""
    s = Settings(_env_file=None)

    assert s.code_agent_model == DEFAULT_AGENT_MODEL_ID
    assert s.generalist_agent_model == DEFAULT_AGENT_MODEL_ID
