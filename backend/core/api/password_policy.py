"""Password acceptance policy for Skynet-native accounts.

Follows NIST SP 800-63B §5.1.1: length-first, no composition rules, no
rotation. The checks that actually blunt online guessing are a minimum
length, a blocklist of commonly used passwords (the SecLists
xato-net-10-million corpus, top 100k filtered to entries long enough to
pass the length rule), and context values an attacker tries first — the
account email and the service's own name.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .errors import DomainError

MIN_LENGTH = 8
# scrypt accepts any input length, so the ceiling only bounds KDF work per
# attempt while leaving real passphrases room (800-63B asks for 64+).
MAX_LENGTH = 128

_SERVICE_NAME = "skynet"
# Local parts shorter than this are too generic to treat as a signal
# (e.g. "il" for il@example.com matching inside "nailbiter").
_MIN_LOCAL_PART = 4

_BLOCKLIST_PATH = Path(__file__).parent / "data" / "common_passwords.txt"


@cache
def _blocklist() -> frozenset[str]:
    """Load the common-password corpus (one lowercased entry per line)."""
    return frozenset(_BLOCKLIST_PATH.read_text(encoding="utf-8").split("\n")) - {""}


def validate_password(password: str, email: str) -> None:
    """Reject a candidate password the verifier must not accept.

    Args:
        password: The candidate plaintext password.
        email: The normalized account email, for context matching.

    Raises:
        DomainError: 422 with a semantic code naming the failed rule
            (``accounts.weak_password``, ``accounts.password_too_long``,
            ``accounts.password_contains_email``, ``accounts.password_common``).
    """
    if len(password) < MIN_LENGTH:
        raise DomainError("accounts.weak_password", status=422)
    if len(password) > MAX_LENGTH:
        raise DomainError("accounts.password_too_long", status=422)
    lowered = password.lower()
    local_part = email.split("@", 1)[0]
    if len(local_part) >= _MIN_LOCAL_PART and local_part in lowered:
        raise DomainError("accounts.password_contains_email", status=422)
    if _SERVICE_NAME in lowered or lowered in _blocklist():
        raise DomainError("accounts.password_common", status=422)
