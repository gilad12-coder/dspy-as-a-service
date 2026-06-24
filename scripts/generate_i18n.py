"""Generate typed i18n constants from the shared catalog.

The source of truth is ``i18n/locales/he.json``. This script emits:

* ``frontend/src/shared/lib/generated/i18n-catalog.ts`` for TypeScript callers.
* ``backend/core/i18n_keys.py`` for Python callers.
* ``backend/core/i18n_locales/he.json`` — in-package copy so wheel installs
  ship the catalog inside ``core`` without depending on the repo-root file.

It intentionally does not extract or translate copy. It only turns the shared
catalog into stable, typo-resistant constants for both runtimes.

Pass ``--check`` to run in dry-run mode: the artefacts are regenerated in
memory and compared against what is on disk; the script exits non-zero (1)
if any artefact is out of sync, without touching the working tree.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "i18n" / "locales" / "he.json"
EN_CATALOG_PATH = ROOT / "i18n" / "locales" / "en.json"
SCHEMA_PATH = ROOT / "i18n" / "schema.json"
TS_OUT = ROOT / "frontend" / "src" / "shared" / "lib" / "generated" / "i18n-catalog.ts"
PY_OUT = ROOT / "backend" / "core" / "i18n_keys.py"
PY_CATALOG_OUT = ROOT / "backend" / "core" / "i18n_locales" / "he.json"

MESSAGE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def _load_catalog() -> dict[str, Any]:
    """Read and validate the catalog file against ``i18n/schema.json``.

    When the ``jsonschema`` package is installed, the full schema is enforced.
    Otherwise a structural fallback verifies that ``terms`` and ``messages``
    are dict sections so downstream emitters cannot crash on the wrong shape.

    Returns:
        Parsed catalog mapping with ``terms`` and ``messages`` confirmed to
        exist as dicts.

    Raises:
        ValueError: When the catalog fails JSON Schema validation.
        TypeError: When ``terms`` or ``messages`` is missing or not a dict
            (structural fallback used when ``jsonschema`` is unavailable).
    """
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if jsonschema is not None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(catalog, schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"{CATALOG_PATH} fails schema {SCHEMA_PATH}: {exc.message}") from exc
    else:
        for section in ("terms", "messages"):
            if not isinstance(catalog.get(section), dict):
                raise TypeError(f"{CATALOG_PATH} must contain object section {section!r}")
    # Enforce the naming convention even when jsonschema is unavailable so a
    # malformed key never sneaks through.
    bad_keys = [k for k in catalog["messages"] if not MESSAGE_KEY_PATTERN.fullmatch(k)]
    if bad_keys:
        raise ValueError(
            f"messages keys must match {MESSAGE_KEY_PATTERN.pattern!r}; offenders: {sorted(bad_keys)}"
        )
    return catalog


def _load_en_overlay() -> dict[str, dict[str, str]]:
    """Load the optional English overlay catalog.

    English is an additive overlay: ``i18n/locales/en.json`` mirrors the Hebrew
    catalog's ``terms`` and ``messages`` sections but may translate only a
    subset — missing keys fall back to Hebrew at runtime, so a partial file is
    valid. The file is optional; absent, both sections render empty.

    Returns:
        Mapping with ``terms`` and ``messages`` dicts (each possibly empty).

    Raises:
        TypeError: When a present section is not a JSON object.
    """
    if not EN_CATALOG_PATH.exists():
        return {"terms": {}, "messages": {}}
    overlay = json.loads(EN_CATALOG_PATH.read_text(encoding="utf-8"))
    for section in ("terms", "messages"):
        value = overlay.get(section, {})
        if not isinstance(value, dict):
            raise TypeError(f"{EN_CATALOG_PATH} section {section!r} must be an object")
        overlay[section] = value
    return overlay


def _validate_overlay(catalog: dict[str, Any], en_overlay: dict[str, dict[str, str]]) -> None:
    """Reject English overlay keys absent from the Hebrew catalog.

    The generated ``*_EN`` constants are typed against the Hebrew-derived key
    unions (``TermKey`` / ``I18nMessageKey``), so an English-only key would emit
    TypeScript that fails to compile. Fail fast here with a clear message.

    Raises:
        ValueError: When the overlay references unknown term or message keys.
    """
    for section in ("terms", "messages"):
        unknown = sorted(set(en_overlay[section]) - set(catalog[section]))
        if unknown:
            raise ValueError(
                f"{EN_CATALOG_PATH} {section} has keys not present in {CATALOG_PATH.name}: {unknown}"
            )


def _enum_name(key: str) -> str:
    """Convert a catalog key to a SCREAMING_SNAKE_CASE Python enum identifier.

    Args:
        key: Catalog key (camelCase or dotted, e.g. ``"jobs.notFound"``).

    Returns:
        Identifier suitable for a ``StrEnum`` member (always at least ``KEY``).
    """
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    name = re.sub(r"[^0-9A-Za-z]+", "_", key).strip("_").upper()
    return name or "KEY"


def _enum_map(keys: list[str], section: str) -> dict[str, str]:
    """Build a deterministic ``{ENUM_NAME: catalog_key}`` mapping.

    Two distinct catalog keys can normalise to the same enum identifier
    (e.g. ``foo.bar`` and ``foo_bar``). That would silently overwrite an
    entry in the generated output, so the collision is rejected here.

    Args:
        keys: Catalog keys (already sorted by caller).
        section: ``"messages"`` or ``"terms"`` — used in error messages.

    Returns:
        Dict mapping each enum identifier to its source catalog key.

    Raises:
        ValueError: When two keys collapse to the same enum identifier.
    """
    mapping: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for key in keys:
        name = _enum_name(key)
        if name in mapping:
            collisions.setdefault(name, [mapping[name]]).append(key)
        else:
            mapping[name] = key
    if collisions:
        details = "; ".join(f"{name} <- {sorted(srcs)}" for name, srcs in sorted(collisions.items()))
        raise ValueError(f"{section} keys collapse to identical enum names: {details}")
    return mapping


def _ts_object(values: dict[str, str]) -> str:
    """Render a flat string mapping as a TypeScript object literal.

    Args:
        values: Mapping rendered as ``{ key: "value", ... }``.

    Returns:
        Multi-line TS source for an object literal.
    """
    lines = ["{"]
    for key, value in values.items():
        prop = key if re.fullmatch(r"[A-Za-z_$][0-9A-Za-z_$]*", key) else json.dumps(key)
        lines.append(f"  {prop}: {json.dumps(value, ensure_ascii=False)},")
    lines.append("}")
    return "\n".join(lines)


def _render_ts(catalog: dict[str, Any], en_overlay: dict[str, dict[str, str]]) -> str:
    """Build the frontend TS catalog source string.

    Sorts every section deterministically by key so a re-ordered source catalog
    produces an identical artefact. Emits the Hebrew catalog plus the partial
    English overlay (``TERMS_EN`` / ``I18N_MESSAGES_EN``), both typed against the
    Hebrew-derived key unions.

    Args:
        catalog: Parsed Hebrew catalog.
        en_overlay: Parsed English overlay (``terms`` / ``messages`` dicts).

    Returns:
        Full TS file contents (including trailing newline).
    """
    terms_sorted = {k: catalog["terms"][k] for k in sorted(catalog["terms"])}
    messages_sorted = {k: catalog["messages"][k] for k in sorted(catalog["messages"])}
    en_terms_sorted = {k: en_overlay["terms"][k] for k in sorted(en_overlay["terms"])}
    en_messages_sorted = {k: en_overlay["messages"][k] for k in sorted(en_overlay["messages"])}
    terms_ts = _ts_object(terms_sorted)
    messages_ts = _ts_object(messages_sorted)
    en_terms_ts = _ts_object(en_terms_sorted)
    en_messages_ts = _ts_object(en_messages_sorted)
    message_key_ts = _ts_object(_enum_map(sorted(catalog["messages"]), "messages"))
    term_key_ts = _ts_object(_enum_map(sorted(catalog["terms"]), "terms"))
    return "\n".join(
        [
            "// Generated by scripts/generate_i18n.py. Do not edit by hand.",
            "",
            f"export const TERMS = {terms_ts} as const;",
            "",
            f"export const I18N_MESSAGES = {messages_ts} as const;",
            "",
            "export type TermKey = keyof typeof TERMS;",
            "export type I18nMessageKey = keyof typeof I18N_MESSAGES;",
            "export type ErrorCode = I18nMessageKey;",
            "",
            "// English overlay — partial; runtime falls back to Hebrew for absent keys.",
            f"export const TERMS_EN: Partial<Record<TermKey, string>> = {en_terms_ts};",
            "",
            f"export const I18N_MESSAGES_EN: Partial<Record<I18nMessageKey, string>> = {en_messages_ts};",
            "",
            f"export const I18N_KEY = {message_key_ts} as const;",
            "",
            f"export const TERM_KEY = {term_key_ts} as const;",
            "",
        ]
    )


def _render_py(catalog: dict[str, Any]) -> str:
    """Build the backend Python keys module source string.

    Args:
        catalog: Parsed catalog with ``messages`` and ``terms`` sections.

    Returns:
        Full Python source for ``backend/core/i18n_keys.py``.
    """
    message_map = _enum_map(sorted(catalog["messages"]), "messages")
    term_map = _enum_map(sorted(catalog["terms"]), "terms")
    message_lines = [f"    {name} = {key!r}" for name, key in message_map.items()] or ["    pass"]
    term_lines = [f"    {name} = {key!r}" for name, key in term_map.items()] or ["    pass"]
    return "\n".join(
        [
            '"""Generated i18n key constants. Do not edit by hand.',
            "",
            "Run ``python scripts/generate_i18n.py`` to regenerate after editing",
            "``i18n/locales/he.json``.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "from enum import StrEnum",
            "",
            "",
            "class I18nKey(StrEnum):",
            '    """Stable identifiers for catalog ``messages`` entries (formatted via ``t()``)."""',
            "",
            *message_lines,
            "",
            "",
            "class TermKey(StrEnum):",
            '    """Stable identifiers for catalog ``terms`` entries (resolved via ``term()``)."""',
            "",
            *term_lines,
            "",
        ]
    )


def _write(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, creating parent directories as needed.

    Args:
        path: Destination file.
        content: UTF-8 text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _check_drift(targets: list[tuple[Path, str]]) -> int:
    """Compare each rendered artefact against the file on disk.

    Args:
        targets: Pairs of (output path, expected content).

    Returns:
        ``0`` when every artefact matches disk, ``1`` otherwise. Drifting
        files are listed on stderr.
    """
    drifted: list[Path] = []
    for path, expected in targets:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != expected:
            drifted.append(path)
    if not drifted:
        return 0
    print("i18n drift detected:", file=sys.stderr)
    for path in drifted:
        print(f"  - {path.relative_to(ROOT)}", file=sys.stderr)
    print("Run 'python scripts/generate_i18n.py' to regenerate.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Regenerate (or audit) the TS, Python, and JSON artefacts.

    Args:
        argv: Optional CLI argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: ``0`` on success, ``1`` when ``--check`` finds
        drift between the catalog and the generated artefacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: exit non-zero if any artefact differs from disk; do not write.",
    )
    args = parser.parse_args(argv)

    catalog = _load_catalog()
    en_overlay = _load_en_overlay()
    _validate_overlay(catalog, en_overlay)
    ts_content = _render_ts(catalog, en_overlay)
    py_content = _render_py(catalog)
    py_catalog_content = CATALOG_PATH.read_text(encoding="utf-8")

    if args.check:
        return _check_drift(
            [
                (TS_OUT, ts_content),
                (PY_OUT, py_content),
                (PY_CATALOG_OUT, py_catalog_content),
            ]
        )

    _write(TS_OUT, ts_content)
    _write(PY_OUT, py_content)
    PY_CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CATALOG_PATH, PY_CATALOG_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
