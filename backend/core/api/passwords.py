"""Password hashing for Skynet-native (email/password) accounts.

Uses the standard library's scrypt KDF — a salted, memory-hard hash — so local
accounts need no third-party crypto dependency. Each stored value is
self-describing (``scrypt$<n>$<r>$<p>$<salt_b64>$<key_b64>``): the cost
parameters travel with the hash, so they can be raised later without a schema
migration and old hashes still verify against their original parameters.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32
# scrypt's working set is 128 * N * r bytes (~16 MiB here); keep maxmem above it
# or the stdlib call raises. A generous cap leaves room to raise N later.
_MAXMEM = 64 * 1024 * 1024


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    """Run scrypt over a password and salt with explicit cost parameters.

    Args:
        password: Plaintext password.
        salt: Per-password random salt.
        n: scrypt CPU/memory cost (a power of two).
        r: scrypt block size.
        p: scrypt parallelization factor.

    Returns:
        The derived key bytes.
    """
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=_KEY_BYTES
    )


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash safe to persist.

    Args:
        password: Plaintext password to hash.

    Returns:
        A ``scrypt$n$r$p$salt$key`` string.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    key = _derive(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(key).decode("ascii"),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    """Check a plaintext password against a stored :func:`hash_password` value.

    Args:
        password: Plaintext password supplied at login.
        encoded: The stored ``scrypt$…`` hash.

    Returns:
        True when the password matches; False on mismatch or a malformed hash.
    """
    try:
        scheme, n_s, r_s, p_s, salt_b64, key_b64 = encoded.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(key_b64)
        candidate = _derive(password, salt, int(n_s), int(r_s), int(p_s))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, expected)
