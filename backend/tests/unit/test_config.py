"""Tests for core.config.Settings and its helpers."""

# pydantic-settings BaseSettings.__init__ accepts ``_env_file`` (and other
# leading-underscore kwargs) that mypy can't see without the dedicated plugin,
# so disable ``call-arg`` for this file.
# mypy: disable-error-code="call-arg"

from __future__ import annotations

import json

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
    "MAX_JOBS_PER_USER",
    "ADMIN_USERNAMES",
    "QUOTA_OVERRIDES",
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


def test_settings_defaults_max_jobs_per_user() -> None:
    """Default ``max_jobs_per_user`` is 100."""
    s = Settings(_env_file=None)

    assert s.max_jobs_per_user == 100


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


def test_settings_env_override_max_jobs_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MAX_JOBS_PER_USER`` env var overrides ``max_jobs_per_user``."""
    monkeypatch.setenv("MAX_JOBS_PER_USER", "200")

    s = Settings(_env_file=None)

    assert s.max_jobs_per_user == 200


def test_settings_env_override_quota_overrides_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QUOTA_OVERRIDES`` env var populates ``quota_overrides_json``."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"power_user": 500}')

    s = Settings(_env_file=None)

    assert s.quota_overrides_json == '{"power_user": 500}'


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


def test_get_user_quota_unknown_user_returns_default() -> None:
    """An unknown username gets the default ``max_jobs_per_user`` quota."""
    s = Settings(_env_file=None)

    result = s.get_user_quota("unknown_user")

    assert result == s.max_jobs_per_user


def test_get_user_quota_admin_uses_default_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin usernames do not bypass quota enforcement."""
    monkeypatch.setenv("ADMIN_USERNAMES", "superadmin")

    s = Settings(_env_file=None)

    assert s.get_user_quota("superadmin") == 100


def test_get_user_quota_admin_case_does_not_change_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin username casing has no effect on quota resolution."""
    monkeypatch.setenv("ADMIN_USERNAMES", "Admin")

    s = Settings(_env_file=None)

    assert s.get_user_quota("admin") == 100


def test_get_user_quota_override_int_returns_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-user integer quota override wins over the default."""
    monkeypatch.setenv("QUOTA_OVERRIDES", json.dumps({"power_user": 500}))

    s = Settings(_env_file=None)

    assert s.get_user_quota("power_user") == 500


def test_get_user_quota_override_none_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-user ``null`` override means unlimited (returns ``None``)."""
    monkeypatch.setenv("QUOTA_OVERRIDES", json.dumps({"researcher": None}))

    s = Settings(_env_file=None)

    assert s.get_user_quota("researcher") is None


def test_get_user_quota_override_independent_from_admin_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-user quota overrides apply independently from admin authorization."""
    monkeypatch.setenv("ADMIN_USERNAMES", "alice")
    monkeypatch.setenv("QUOTA_OVERRIDES", json.dumps({"alice": 50}))

    s = Settings(_env_file=None)

    assert s.get_user_quota("alice") == 50


def test_get_user_quota_non_admin_user_with_no_override_returns_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-admin users without an override get the configured ``max_jobs_per_user``."""
    monkeypatch.setenv("MAX_JOBS_PER_USER", "75")
    monkeypatch.setenv("QUOTA_OVERRIDES", json.dumps({"other_user": 200}))

    s = Settings(_env_file=None)

    assert s.get_user_quota("regular_user") == 75


@pytest.mark.parametrize(
    ("username", "quota_json", "admin_csv", "expected"),
    [
        ("regular", "{}", "", 100),
        ("admin1", "{}", "admin1", 100),
        ("power", '{"power": 500}', "", 500),
        ("unlim", '{"unlim": null}', "", None),
        ("admin1", '{"admin1": 50}', "admin1", 50),
    ],
    ids=["regular_default", "admin_default", "override_int", "override_none", "admin_with_override"],
)
def test_get_user_quota_parametrized(
    monkeypatch: pytest.MonkeyPatch,
    username: str,
    quota_json: str,
    admin_csv: str,
    expected: int | None,
) -> None:
    """Parametrized check for override/default quota precedence."""
    monkeypatch.setenv("QUOTA_OVERRIDES", quota_json)
    monkeypatch.setenv("ADMIN_USERNAMES", admin_csv)

    s = Settings(_env_file=None)

    assert s.get_user_quota(username) == expected


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


def test_settings_valid_quota_overrides_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid ``QUOTA_OVERRIDES`` JSON is parsed and applied to ``get_user_quota``."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"power_user": 500, "researcher": null}')

    s = Settings(_env_file=None)

    assert s.get_user_quota("power_user") == 500
    assert s.get_user_quota("researcher") is None


def test_settings_malformed_quota_overrides_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed ``QUOTA_OVERRIDES`` JSON raises a validation error."""
    monkeypatch.setenv("QUOTA_OVERRIDES", "{not valid json")

    with pytest.raises(ValueError):  # pydantic.ValidationError inherits from ValueError
        Settings(_env_file=None)


def test_settings_empty_quota_overrides_json_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``QUOTA_OVERRIDES`` env var is normalised to ``"{}"``."""
    monkeypatch.setenv("QUOTA_OVERRIDES", "")

    s = Settings(_env_file=None)

    assert s.get_user_quota("any_user") == s.max_jobs_per_user


def test_settings_quota_overrides_rejects_array_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array is rejected — overrides must be an object."""
    monkeypatch.setenv("QUOTA_OVERRIDES", "[1, 2, 3]")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_settings_quota_overrides_rejects_string_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-int, non-null value is rejected by the validator."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"alice": "unlimited"}')

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_settings_quota_overrides_rejects_bool_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bool value is rejected (``True``/``False`` would silently coerce to 1/0)."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"alice": true}')

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_get_user_quota_lookup_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override lookup matches regardless of casing in either env or query."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"PowerUser": 500}')

    s = Settings(_env_file=None)

    assert s.get_user_quota("PowerUser") == 500
    assert s.get_user_quota("poweruser") == 500
    assert s.get_user_quota("POWERUSER") == 500


def test_quota_overrides_property_parses_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``quota_overrides`` is cached: repeated reads return the same dict instance."""
    monkeypatch.setenv("QUOTA_OVERRIDES", '{"alice": 50}')

    s = Settings(_env_file=None)

    assert s.quota_overrides is s.quota_overrides


def test_settings_default_agent_models_use_shared_constant() -> None:
    """Both agents default to ``DEFAULT_AGENT_MODEL_ID`` so a single swap covers both."""
    s = Settings(_env_file=None)

    assert s.code_agent_model == DEFAULT_AGENT_MODEL_ID
    assert s.generalist_agent_model == DEFAULT_AGENT_MODEL_ID
