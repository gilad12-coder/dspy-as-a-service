# Contributing to Skynet

Thanks for considering a contribution! This guide covers the things that
aren't obvious from the code — the test layout, the i18n rules, and the
migration discipline that keeps deploys boring.

For deeper engineering conventions (commenting style, docstrings, import
rules), read [`AGENTS.md`](AGENTS.md) — it is the canonical reference and
applies to every file you touch.

## Dev setup

```bash
createdb skynet
cp backend/.env.example backend/.env          # set REMOTE_DB_URL + model keys
cp frontend/.env.example frontend/.env.local
cd deploy/litellm && docker compose up -d && cd ../..
just install
just backend    # :8000
just frontend   # :3000
```

## Tests — run the right suites

Backend tests live in **two** places; run both before opening a PR:

```bash
cd backend
.venv/bin/python -m pytest core/ -q      # unit suites next to each package
.venv/bin/python -m pytest tests/unit -q # cross-cutting policy suites
```

The policy suites enforce repo invariants (e.g. *no Hebrew literals outside
the i18n catalog*, OpenAPI surface checks) that the per-package suites won't
catch — skipping them is how green-locally-red-in-CI happens.

Frontend:

```bash
cd frontend
npm run typecheck
npm run lint
npm run test:unit
```

Or simply `just test` for the combined sweep.

## i18n rules

- Hebrew (`i18n/locales/ui/he.json`) is the **base catalog** — every key must
  exist there. English (`en.json`) is the source text translators work from.
  Add new UI strings to **both**, then run:

  ```bash
  python3 scripts/generate_i18n.py
  ```

  This regenerates the typed catalogs (unknown keys become compile errors).
- Never hardcode user-facing text in JSX — lint enforces it. Use `msg()` /
  `formatMsg()` with semantic keys (`feature.surface.name`), not the legacy
  `auto.*` extractor keys.
- The other 22 locales are topped up in batches via
  `scripts/i18n_sync.py extract → translate → apply`; shipping with only
  he+en populated is fine (fallback covers the rest).
- RTL: use logical CSS properties (`ms-*`/`me-*`, `text-start`), `dir="auto"`
  on user content, and `dir="ltr"` on code/model ids.

## Database migrations

- One linear Alembic head, always. Before adding a migration, find the
  current head and set it as your `down_revision`.
- **Pick a revision id that doesn't exist yet** — grep
  `backend/alembic/versions/` first. A duplicated id corrupts the revision
  graph into a cycle, and because migrations run at boot
  (`sync_migration_head`), that crashes every deployment at startup.
- Migrations must be safe on a live database (plain transactional DDL,
  `IF NOT EXISTS` guards).

## Pull requests

- Branch from `main`; never push to `main` directly.
- Conventional commit subjects (`feat(tagger): …`, `fix(db): …`) — the squash
  commit becomes the changelog line.
- Explain *why* in the PR body, list what you tested, and keep diffs
  surgical: every changed line should trace to the PR's purpose.

## License of contributions

Skynet is licensed under [AGPL-3.0](LICENSE). By contributing you agree that
your contributions are licensed under the same terms.
