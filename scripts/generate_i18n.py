"""Generate typed i18n constants from the shared catalog.

The source of truth is ``i18n/locales/he.json``. This script emits:

* ``frontend/src/shared/lib/generated/i18n-catalog.ts`` for TypeScript callers.
* ``frontend/src/shared/lib/generated/ui-catalog.ts`` — the consolidated UI
  string catalogs (``i18n/locales/ui/<locale>.json``) as a per-locale registry.
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
# Hebrew is the complete backend base; every other i18n/locales/<locale>.json is
# an additive overlay merged at runtime via the registry fallback chain.
BACKEND_BASE_LOCALE = "he"
SCHEMA_PATH = ROOT / "i18n" / "schema.json"
TS_OUT = ROOT / "frontend" / "src" / "shared" / "lib" / "generated" / "i18n-catalog.ts"
PY_OUT = ROOT / "backend" / "core" / "i18n_keys.py"
PY_CATALOG_OUT = ROOT / "backend" / "core" / "i18n_locales" / "he.json"

# Frontend UI strings live in a separate per-locale catalog tree so backend
# wheels never ship UI copy. ``he.json`` is the complete base; every other
# ``<locale>.json`` is an overlay merged at runtime via the registry fallback
# chain.
UI_CATALOG_DIR = ROOT / "i18n" / "locales" / "ui"
UI_TS_OUT = ROOT / "frontend" / "src" / "shared" / "lib" / "generated" / "ui-catalog.ts"
UI_BASE_LOCALE = "he"

MESSAGE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
# UI keys include auto-extracted, digit-leading segments (``...literal.16``),
# so they use a looser dotted-identifier rule than the backend message keys.
UI_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+$")


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


def _load_backend_overlays() -> dict[str, dict[str, dict[str, str]]]:
    """Load every per-locale backend overlay from ``i18n/locales/<locale>.json``.

    Globs all ``*.json`` except the Hebrew base (``he.json``). Each overlay
    mirrors the base catalog's ``terms`` and ``messages`` sections but may
    translate only a subset — missing keys fall back through the registry chain
    at runtime, so a partial file (e.g. a regional-variant delta) is valid. The
    ``ui`` subdirectory is a separate tree and is not globbed here.

    Returns:
        Mapping of locale tag to ``{"terms": {...}, "messages": {...}}`` (each
        section possibly empty).

    Raises:
        TypeError: When a present section is not a JSON object.
    """
    overlays: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted((ROOT / "i18n" / "locales").glob("*.json")):
        if path.stem == BACKEND_BASE_LOCALE:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sections: dict[str, dict[str, str]] = {}
        for section in ("terms", "messages"):
            value = data.get(section, {})
            if not isinstance(value, dict):
                raise TypeError(f"{path} section {section!r} must be an object")
            sections[section] = {k: str(v) for k, v in value.items()}
        overlays[path.stem] = sections
    return overlays


def _validate_backend_overlays(
    catalog: dict[str, Any], overlays: dict[str, dict[str, dict[str, str]]]
) -> None:
    """Reject overlay keys absent from the Hebrew base, and require English to
    translate every term.

    Every overlay's ``terms`` / ``messages`` consts are typed against the
    Hebrew-derived key unions (``TermKey`` / ``I18nMessageKey``), so an unknown
    key would emit TypeScript that fails to compile. Fail fast here.

    English additionally must translate every term: ``en`` falls back to ``he``,
    so a term missing from ``en.json`` leaks Hebrew into the English UI. Other
    locales fall back through English first (e.g. ``fr -> en -> he``), so a
    partial overlay degrades to English, never Hebrew — partial is fine for them.

    Raises:
        ValueError: When an overlay references unknown keys, or when a Hebrew
            term has no English translation.
    """
    he_sections = {"terms": set(catalog["terms"]), "messages": set(catalog["messages"])}
    for locale, overlay in overlays.items():
        for section in ("terms", "messages"):
            unknown = sorted(set(overlay[section]) - he_sections[section])
            if unknown:
                raise ValueError(
                    f"i18n/locales/{locale}.json {section} has keys not present in "
                    f"{CATALOG_PATH.name}: {unknown}"
                )

    en_terms = set(overlays.get("en", {}).get("terms", {}))
    untranslated_terms = sorted(he_sections["terms"] - en_terms)
    if untranslated_terms:
        raise ValueError(
            f"{EN_CATALOG_PATH} is missing English translations for these terms: "
            f"{untranslated_terms}. Every term must be translated — an untranslated "
            f"term falls back to Hebrew and leaks into the English UI. Add the English "
            f'value(s) under "terms" in {EN_CATALOG_PATH.name}.'
        )


def _be_ident(prefix: str, locale: str) -> str:
    """Map a locale tag to a backend overlay const identifier (``pt-BR`` ->
    ``terms_pt_BR``)."""
    return f"{prefix}_" + re.sub(r"[^A-Za-z0-9]", "_", locale)


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


def _render_ts(catalog: dict[str, Any], overlays: dict[str, dict[str, dict[str, str]]]) -> str:
    """Build the frontend TS catalog source string.

    Sorts every section deterministically by key so a re-ordered source catalog
    produces an identical artefact. Emits the Hebrew base, each per-locale
    overlay as a ``Partial`` const, and ``TERMS_BY_LOCALE`` /
    ``I18N_MESSAGES_BY_LOCALE`` registries keyed by locale tag (which ``i18n.ts``
    walks via the registry fallback chain). English keeps its historical
    ``TERMS_EN`` / ``I18N_MESSAGES_EN`` exports for backward compatibility.

    Args:
        catalog: Parsed Hebrew catalog.
        overlays: Per-locale overlays from ``_load_backend_overlays``.

    Returns:
        Full TS file contents (including trailing newline).
    """
    terms_sorted = {k: catalog["terms"][k] for k in sorted(catalog["terms"])}
    messages_sorted = {k: catalog["messages"][k] for k in sorted(catalog["messages"])}
    terms_ts = _ts_object(terms_sorted)
    messages_ts = _ts_object(messages_sorted)
    message_key_ts = _ts_object(_enum_map(sorted(catalog["messages"]), "messages"))
    term_key_ts = _ts_object(_enum_map(sorted(catalog["terms"]), "terms"))
    overlay_locales = sorted(overlays)

    lines = [
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
        "// Per-locale overlays — partial; runtime walks the fallback chain over",
        "// the registries below, so an absent key degrades to the next locale.",
    ]
    term_idents: dict[str, str] = {}
    msg_idents: dict[str, str] = {}
    for locale in overlay_locales:
        overlay = overlays[locale]
        terms_obj = _ts_object({k: overlay["terms"][k] for k in sorted(overlay["terms"])})
        msgs_obj = _ts_object({k: overlay["messages"][k] for k in sorted(overlay["messages"])})
        if locale == "en":
            t_ident, m_ident, keyword = "TERMS_EN", "I18N_MESSAGES_EN", "export const"
        else:
            t_ident, m_ident, keyword = _be_ident("terms", locale), _be_ident("i18n", locale), "const"
        term_idents[locale] = t_ident
        msg_idents[locale] = m_ident
        lines.append(f"{keyword} {t_ident}: Partial<Record<TermKey, string>> = {terms_obj};")
        lines.append("")
        lines.append(f"{keyword} {m_ident}: Partial<Record<I18nMessageKey, string>> = {msgs_obj};")
        lines.append("")

    def _registry(name: str, base_ident: str, idents: dict[str, str]) -> str:
        rows = [f"export const {name}: Record<string, Record<string, string>> = {{"]
        rows.append(f"  he: {base_ident} as Record<string, string>,")
        for locale in overlay_locales:
            prop = locale if re.fullmatch(r"[A-Za-z_$][0-9A-Za-z_$]*", locale) else json.dumps(locale)
            rows.append(f"  {prop}: {idents[locale]} as Record<string, string>,")
        rows.append("};")
        return "\n".join(rows)

    lines.append(_registry("TERMS_BY_LOCALE", "TERMS", term_idents))
    lines.append("")
    lines.append(_registry("I18N_MESSAGES_BY_LOCALE", "I18N_MESSAGES", msg_idents))
    lines.append("")
    lines.append(f"export const I18N_KEY = {message_key_ts} as const;")
    lines.append("")
    lines.append(f"export const TERM_KEY = {term_key_ts} as const;")
    lines.append("")
    return "\n".join(lines)


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


def _load_ui_catalogs() -> dict[str, dict[str, str]]:
    """Load the per-locale UI string catalogs from ``i18n/locales/ui``.

    ``he.json`` is the complete base whose key set defines ``MessageKey``; every
    other ``<locale>.json`` is an overlay that may translate any subset and
    inherits the rest through the registry fallback chain at runtime. Overlay
    keys absent from the base are rejected — they would type-error against the
    generated key union.

    Returns:
        Mapping of locale tag to its flat ``{key: value}`` catalog, always
        including the base locale.

    Raises:
        FileNotFoundError: When the base ``he.json`` is missing.
        TypeError: When a catalog file is not a JSON object.
        ValueError: When a key violates ``UI_KEY_PATTERN`` or an overlay
            references a key absent from the base.
    """
    base_path = UI_CATALOG_DIR / f"{UI_BASE_LOCALE}.json"
    if not base_path.exists():
        raise FileNotFoundError(f"missing UI base catalog {base_path}")
    catalogs: dict[str, dict[str, str]] = {}
    for path in sorted(UI_CATALOG_DIR.glob("*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict):
            raise TypeError(f"{path} must be a JSON object of key -> string")
        bad = [k for k in catalog if not UI_KEY_PATTERN.fullmatch(k)]
        if bad:
            raise ValueError(f"{path} has malformed keys: {sorted(bad)[:10]}")
        catalogs[path.stem] = {k: str(v) for k, v in catalog.items()}
    base_keys = set(catalogs[UI_BASE_LOCALE])
    for locale, catalog in catalogs.items():
        if locale == UI_BASE_LOCALE:
            continue
        unknown = sorted(set(catalog) - base_keys)
        if unknown:
            raise ValueError(
                f"{UI_CATALOG_DIR / f'{locale}.json'} has keys absent from "
                f"{UI_BASE_LOCALE}.json: {unknown[:10]}"
            )
    return catalogs


def _ui_ident(locale: str) -> str:
    """Map a locale tag to a TS const identifier (``pt-BR`` -> ``ui_pt_BR``)."""
    return "ui_" + re.sub(r"[^A-Za-z0-9]", "_", locale)


def _ts_union(keys: list[str]) -> str:
    """Render a sorted key list as a TS string-literal union type body."""
    return "\n".join(f"  | {json.dumps(k, ensure_ascii=False)}" for k in keys)


def _render_ui_ts(catalogs: dict[str, dict[str, str]]) -> str:
    """Build the frontend UI catalog TS source.

    Emits the ``MessageKey`` union (from the base locale's keys), the complete
    base catalog as ``UI_MESSAGES``, each overlay locale as a ``Partial`` const,
    and a ``UI_CATALOGS`` registry keyed by locale tag. The server-only loader
    merges a request's fallback chain over this registry; the client never
    imports it, so no catalog ships in the browser bundle.

    Args:
        catalogs: Per-locale catalogs from ``_load_ui_catalogs``.

    Returns:
        Full TS file contents (including trailing newline).
    """
    base = catalogs[UI_BASE_LOCALE]
    base_keys = sorted(base)
    base_sorted = {k: base[k] for k in base_keys}
    overlay_locales = sorted(loc for loc in catalogs if loc != UI_BASE_LOCALE)

    lines = [
        "// Generated by scripts/generate_i18n.py. Do not edit by hand.",
        "",
        "export type MessageKey =",
        f"{_ts_union(base_keys)};",
        "",
        f"export const UI_MESSAGES: Record<MessageKey, string> = {_ts_object(base_sorted)} as Record<MessageKey, string>;",
        "",
    ]
    for locale in overlay_locales:
        overlay_sorted = {k: catalogs[locale][k] for k in sorted(catalogs[locale])}
        lines.append(
            f"const {_ui_ident(locale)}: Partial<Record<MessageKey, string>> = {_ts_object(overlay_sorted)};"
        )
        lines.append("")
    registry = ["export const UI_CATALOGS: Record<string, Partial<Record<MessageKey, string>>> = {"]
    registry.append(f"  {UI_BASE_LOCALE}: UI_MESSAGES,")
    for locale in overlay_locales:
        prop = locale if re.fullmatch(r"[A-Za-z_$][0-9A-Za-z_$]*", locale) else json.dumps(locale)
        registry.append(f"  {prop}: {_ui_ident(locale)},")
    registry.append("};")
    lines.append("\n".join(registry))
    lines.append("")
    return "\n".join(lines)


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
    overlays = _load_backend_overlays()
    _validate_backend_overlays(catalog, overlays)
    ts_content = _render_ts(catalog, overlays)
    py_content = _render_py(catalog)
    py_catalog_content = CATALOG_PATH.read_text(encoding="utf-8")
    ui_ts_content = _render_ui_ts(_load_ui_catalogs())

    if args.check:
        return _check_drift(
            [
                (TS_OUT, ts_content),
                (PY_OUT, py_content),
                (PY_CATALOG_OUT, py_catalog_content),
                (UI_TS_OUT, ui_ts_content),
            ]
        )

    _write(TS_OUT, ts_content)
    _write(PY_OUT, py_content)
    _write(UI_TS_OUT, ui_ts_content)
    PY_CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CATALOG_PATH, PY_CATALOG_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
