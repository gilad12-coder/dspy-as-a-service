"""Direct Postgres inspector for invariants the API does not expose.

Some assertions — exactly-once job creation under idempotent retries,
orphan recovery after a pod kill — are easiest to verify by reading the
underlying table directly. This module wraps the small read-only surface
the scenarios need.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
import psycopg2.extras


def _dsn() -> str:
    """Return the load-test database DSN.

    Returns:
        The value of ``LOAD_TEST_DB_URL`` when set, otherwise
        ``REMOTE_DB_URL`` (used by the running backend). The harness
        normally connects directly to Postgres rather than via PgBouncer
        because session-scoped statements (idempotency lookups) are
        clearer to assert against the source of truth.

    Raises:
        RuntimeError: When neither variable is set.
    """
    dsn = os.getenv("LOAD_TEST_DB_URL") or os.getenv("REMOTE_DB_URL")
    if not dsn:
        raise RuntimeError("LOAD_TEST_DB_URL or REMOTE_DB_URL must be set for DB inspection.")
    return dsn


@contextmanager
def cursor() -> Iterator[psycopg2.extensions.cursor]:
    """Yield a short-lived psycopg2 cursor and close on exit.

    Yields:
        A psycopg2 :class:`DictCursor` with the load-test DSN connection
        managed for the caller.
    """
    conn = psycopg2.connect(_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    finally:
        conn.close()


def count_jobs_by_idempotency_key(username: str, idempotency_key: str) -> int:
    """Return the row count for ``(username, idempotency_key)`` in ``jobs``.

    Used by the idempotency invariants: exactly one job must exist for any
    given ``(username, idempotency_key)`` pair, regardless of how many
    retries the client sent.

    Args:
        username: Submitter scope.
        idempotency_key: Client-supplied dedup key.

    Returns:
        Number of matching rows.
    """
    with cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE username = %s AND idempotency_key = %s",
            (username, idempotency_key),
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0


def list_jobs_by_username(username: str, limit: int = 1000) -> list[dict]:
    """Return the most recent jobs owned by ``username``.

    Args:
        username: Submitter to filter on.
        limit: Maximum number of rows to return.

    Returns:
        A list of dicts with ``optimization_id``, ``status``,
        ``claimed_by``, ``notified_at``, and ``created_at``.
    """
    with cursor() as cur:
        cur.execute(
            "SELECT optimization_id, status, claimed_by, notified_at, created_at "
            "FROM jobs WHERE username = %s ORDER BY created_at DESC LIMIT %s",
            (username, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def count_orphaned_jobs() -> int:
    """Return the number of jobs whose lease has expired but status is not terminal.

    Used by the failure-injection scenario to detect orphans before the
    sweeper runs.

    Returns:
        The count of jobs still claimed by an expired lease.
    """
    with cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM jobs "
            "WHERE status IN ('running', 'validating') "
            "AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < NOW()"
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0


def count_job_statuses(usernames: list[str]) -> dict[str, int]:
    """Return job counts grouped by status for the supplied test users.

    Args:
        usernames: Test owners whose jobs belong to the current scenario.

    Returns:
        A status-to-count mapping, or an empty dict when no usernames are given.
    """
    if not usernames:
        return {}
    with cursor() as cur:
        cur.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE username = ANY(%s) GROUP BY status",
            (usernames,),
        )
        return {str(row["status"]): int(row["n"]) for row in cur.fetchall()}


def fund_test_users(usernames: list[str], credits: int = 100_000) -> int:
    """Give synthetic load-test users isolated, non-Stripe credit balances.

    The production submit gate correctly refuses unfunded identities. Load tests
    point every model at the free mock provider, so these rows exist only to let
    the request traverse the same billing gate; no Stripe or provider account is
    touched.

    Args:
        usernames: Synthetic identities to fund in the dedicated test database.
        credits: Paid-credit balance assigned to each identity.

    Returns:
        Number of user rows inserted or updated.
    """
    if not usernames:
        return 0
    with cursor() as cur:
        cur.execute(
            "INSERT INTO billing_customers "
            "(username, stripe_customer_id, credit_balance, grant_remaining, created_at, updated_at) "
            "SELECT username, 'loadtest_' || md5(username), %s, 0, NOW(), NOW() "
            "FROM unnest(%s::text[]) AS username "
            "ON CONFLICT (username) DO UPDATE SET "
            "credit_balance = EXCLUDED.credit_balance, grant_remaining = 0, updated_at = NOW()",
            (credits, usernames),
        )
        cur.connection.commit()
        return cur.rowcount or 0


def truncate_test_users(usernames: list[str]) -> int:
    """Delete every job row owned by the listed test usernames.

    Used between scenario runs so each scenario sees a clean baseline.
    Only touches rows belonging to the load-test usernames the caller
    supplies; never wipes the whole table.

    Args:
        usernames: Test usernames to scrub.

    Returns:
        The number of rows deleted across all listed users.
    """
    if not usernames:
        return 0
    with cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE username = ANY(%s)", (usernames,))
        cur.connection.commit()
        return cur.rowcount or 0
