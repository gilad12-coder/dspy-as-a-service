"""Sync the translation overlays: extract missing keys, apply translations.

Companion to ``generate_i18n.py``. Two surfaces are localized, selected with
``--scope`` (default ``ui``): the per-locale UI string tree under
``i18n/locales/ui`` and the backend error/term catalog under ``i18n/locales``
(whose strings nest under a ``messages`` object). In both, Hebrew is the complete
base — every key exists in ``he.json`` — and each other ``<locale>.json`` is an
overlay that may translate a subset, inheriting the rest through the registry
fallback chain at runtime. Over time the *full* language overlays (the ones meant
to be complete — de, fr, ja, …) drift behind the base as new strings land only in
he/en. This script closes that gap in two deterministic phases that bracket a
translation step (the ``translate-i18n`` agent workflow):

    extract <workdir>  Compute, for every full locale, the base keys it is missing,
                       and write:
                         <workdir>/keys.json     union of missing keys -> English text
                         <workdir>/missing.json  per-locale missing-key lists
                         <workdir>/out/          (created empty; the workflow fills it)
                       and print ``ARGS_JSON=…`` — the small arg object to hand to
                       the translation workflow (key file path, out dir, locale meta).

    apply   <workdir>  Merge <workdir>/out/<locale>.json translations back into each
                       overlay, validating that placeholders are preserved and never
                       adding a non-base key. Existing entries are left untouched;
                       new keys are inserted verbatim so diffs stay reviewable.

Pass the same ``--scope`` to both phases. The intentionally-partial regional delta
overlays (en-GB, pt-BR, fr-CA, yue, …) are left alone: a locale is topped up only
when it is already mostly complete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Complete key set lives in the Hebrew base; English supplies the source text the
# translators work from (both carry every key).
BASE_LOCALE = "he"
SOURCE_LOCALE = "en"


@dataclass(frozen=True)
class Scope:
    """A translatable catalog surface: where its locale files live and their shape.

    Attributes:
        name: CLI identifier (``ui`` or ``backend``).
        directory: Folder holding the ``<locale>.json`` catalog files.
        max_missing: A locale is topped up only when it misses no more than this
            many base keys. The intentionally-partial regional deltas miss far
            more and are skipped, so they keep falling back to their parent.
        section: ``None`` when the file is a flat ``{key: string}`` map (UI); the
            name of the object the strings nest under (``"messages"``) for the
            backend catalog, whose files also carry ``terms`` / ``$schema``.
    """

    name: str
    directory: Path
    max_missing: int
    section: str | None


# The two surfaces the app localizes. ``ui`` is the per-locale UI string tree;
# ``backend`` is the error/term catalog the frontend reads to localize API codes.
# The thresholds differ because the bases differ in size (UI ~2.6k keys, backend
# ~200): in both, full locales miss a handful while regional deltas miss most.
SCOPES: dict[str, Scope] = {
    "ui": Scope("ui", ROOT / "i18n" / "locales" / "ui", 500, None),
    "backend": Scope("backend", ROOT / "i18n" / "locales", 50, "messages"),
}

# Endonym-free English names + writing direction, used to brief each translator.
# A locale absent here still works (the workflow infers the language from the tag).
LOCALE_META: dict[str, dict[str, str]] = {
    "ar": {"englishName": "Arabic", "dir": "rtl"},
    "de": {"englishName": "German", "dir": "ltr"},
    "es": {"englishName": "Spanish", "dir": "ltr"},
    "fa": {"englishName": "Persian", "dir": "rtl"},
    "fr": {"englishName": "French", "dir": "ltr"},
    "hi": {"englishName": "Hindi", "dir": "ltr"},
    "it": {"englishName": "Italian", "dir": "ltr"},
    "ja": {"englishName": "Japanese", "dir": "ltr"},
    "ko": {"englishName": "Korean", "dir": "ltr"},
    "pt": {"englishName": "Portuguese", "dir": "ltr"},
    "ru": {"englishName": "Russian", "dir": "ltr"},
    "tr": {"englishName": "Turkish", "dir": "ltr"},
    "uk": {"englishName": "Ukrainian", "dir": "ltr"},
    "zh-Hans": {"englishName": "Chinese (Simplified)", "dir": "ltr"},
}

# Every {…} run in a string is a placeholder (named params, {term.x} vocabulary,
# ICU plural blocks). A translation must carry exactly the same set, untranslated.
_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")


def _load(scope: Scope, locale: str) -> dict[str, str]:
    """Return the flat key->string map of translatable strings for a locale.

    Args:
        scope: Catalog surface being synced.
        locale: Locale tag whose file to read.

    Returns:
        The whole file for a flat scope, or its nested ``scope.section`` object
        for a sectioned one (``{}`` when that section is absent or mistyped).
    """
    data = json.loads((scope.directory / f"{locale}.json").read_text(encoding="utf-8"))
    if scope.section is None:
        return data
    section = data.get(scope.section, {})
    return section if isinstance(section, dict) else {}


def _placeholders(text: str) -> set[str]:
    """Return the set of ``{…}`` placeholder tokens in a string."""
    return set(_PLACEHOLDER_RE.findall(text))


def _full_locales(scope: Scope, base_keys: set[str]) -> list[str]:
    """Return the overlay locales eligible for top-up (mostly-complete, drifted).

    Args:
        scope: Catalog surface being synced.
        base_keys: The complete base key set.

    Returns:
        Locale tags (sorted) that are missing some — but not too many — base keys.
    """
    locales: list[str] = []
    for path in sorted(scope.directory.glob("*.json")):
        locale = path.stem
        if locale in (BASE_LOCALE, SOURCE_LOCALE):
            continue
        missing = base_keys - set(_load(scope, locale))
        if 0 < len(missing) <= scope.max_missing:
            locales.append(locale)
    return locales


def cmd_extract(scope: Scope, workdir: Path) -> int:
    """Write the translation work order for every drifted full locale.

    Args:
        scope: Catalog surface being synced.
        workdir: Scratch directory the workflow round-trips through.

    Returns:
        Process exit status (always 0).
    """
    base = _load(scope, BASE_LOCALE)
    source = _load(scope, SOURCE_LOCALE)
    base_keys = set(base)
    locales = _full_locales(scope, base_keys)

    union: dict[str, str] = {}
    per_locale: dict[str, list[str]] = {}
    for locale in locales:
        missing = sorted(base_keys - set(_load(scope, locale)))
        per_locale[locale] = missing
        for key in missing:
            union.setdefault(key, source.get(key, base[key]))

    out_dir = workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "keys.json").write_text(json.dumps(union, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "missing.json").write_text(
        json.dumps(per_locale, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    workflow_args = {
        "keysFile": str((workdir / "keys.json").resolve()),
        "outDir": str(out_dir.resolve()),
        "locales": [
            {"code": locale, **LOCALE_META.get(locale, {"englishName": locale, "dir": "ltr"})}
            for locale in locales
        ],
    }
    print(f"extract[{scope.name}]: {len(locales)} locales, {len(union)} unique missing keys")
    for locale in locales:
        print(f"  {locale}: {len(per_locale[locale])} missing")
    print("ARGS_JSON=" + json.dumps(workflow_args, ensure_ascii=False))
    return 0


def _merge_into(scope: Scope, locale: str, additions: dict[str, str]) -> None:
    """Insert new entries at the top of a locale overlay, preserving the rest.

    Splicing just after the target object's opening brace keeps every existing
    line byte-for-byte, so the diff is exactly the added keys. For a flat scope
    that object is the file root; for a sectioned scope it is the section object
    (e.g. ``"messages"``). The result is parsed before writing to guarantee valid
    JSON.

    Args:
        scope: Catalog surface being synced.
        locale: Overlay locale tag to update.
        additions: New key->translation pairs to insert.

    Raises:
        ValueError: If the spliced text is not valid JSON.
    """
    path = scope.directory / f"{locale}.json"
    text = path.read_text(encoding="utf-8")
    indent = "    " if scope.section else "  "
    block = "".join(
        f"{indent}{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)},\n"
        for k, v in additions.items()
    )
    anchor = 0 if scope.section is None else text.index(f'"{scope.section}"')
    brace = text.index("{", anchor)
    newline = text.index("\n", brace)
    merged = text[: newline + 1] + block + text[newline + 1 :]
    try:
        json.loads(merged)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{locale}: merge produced invalid JSON: {exc}") from exc
    path.write_text(merged, encoding="utf-8")


def cmd_apply(scope: Scope, workdir: Path) -> int:
    """Merge the workflow's translations back into the overlays, with validation.

    Args:
        scope: Catalog surface being synced.
        workdir: Scratch directory used by :func:`cmd_extract`.

    Returns:
        Process exit status: 0 when everything merged, 1 when any locale produced
        no usable output (so a re-run is signalled).
    """
    base = _load(scope, BASE_LOCALE)
    source = _load(scope, SOURCE_LOCALE)
    base_keys = set(base)
    out_dir = workdir / "out"
    per_locale = json.loads((workdir / "missing.json").read_text(encoding="utf-8"))

    total = 0
    problems: list[str] = []
    empty_locales: list[str] = []
    for locale, missing in per_locale.items():
        out_file = out_dir / f"{locale}.json"
        if not out_file.exists():
            empty_locales.append(locale)
            problems.append(f"{locale}: no output file")
            continue
        translations = json.loads(out_file.read_text(encoding="utf-8"))
        existing = _load(scope, locale)
        additions: dict[str, str] = {}
        for key in missing:
            if key in existing or key not in base_keys:
                continue
            value = translations.get(key)
            src = source.get(key, base.get(key, ""))
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{locale}/{key}: missing or empty")
                continue
            if _placeholders(value) != _placeholders(src):
                problems.append(f"{locale}/{key}: placeholder mismatch")
                continue
            additions[key] = value
        if additions:
            _merge_into(scope, locale, additions)
        else:
            empty_locales.append(locale)
        total += len(additions)
        print(f"  {locale}: +{len(additions)} merged")

    if problems:
        print(f"apply: {len(problems)} skipped entries:", file=sys.stderr)
        for problem in problems[:60]:
            print(f"  - {problem}", file=sys.stderr)
    print(f"apply[{scope.name}]: merged {total} translations across {len(per_locale)} locales")
    return 1 if empty_locales else 0


def main(argv: list[str] | None = None) -> int:
    """Run the requested phase (``extract`` or ``apply``).

    Args:
        argv: Optional CLI argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status from the chosen subcommand.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "apply"):
        p = sub.add_parser(name)
        p.add_argument("workdir", type=Path, help="Scratch directory for the round-trip.")
        p.add_argument(
            "--scope",
            choices=sorted(SCOPES),
            default="ui",
            help="Catalog surface to sync: 'ui' strings (default) or 'backend' messages.",
        )
    args = parser.parse_args(argv)
    scope = SCOPES[args.scope]
    if args.cmd == "extract":
        return cmd_extract(scope, args.workdir)
    return cmd_apply(scope, args.workdir)


if __name__ == "__main__":
    sys.exit(main())
