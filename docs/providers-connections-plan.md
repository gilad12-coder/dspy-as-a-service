# Skynet — Provider Connections & Managed + BYOK Routing Plan

> **Derived from** the Providers-tab shaping session (2026-06-29) + two codebase scouts
> (frontend BYOK/catalog rails; backend routing/metering/vault).
> **Supersedes** `docs/monetization-implementation-plan.md` **Phase 7 (BYOK vault)** — its
> premise ("replace the in-memory stub") is already obsolete: the encrypt-at-rest vault
> is built. This plan is the next layer: generalize the vault into LiteLLM-style
> *connections*, wire it into the run path (the missing keystone), redesign the Providers
> tab, and add the managed gateway.
> **Status:** implemented (Phases A–E) on `feat/provider-connections`. Phase E's
> proxy is feature-flagged off by default and ships as config + deploy docs only
> (operator stands it up). See the phase sections for what landed.

## Locked decisions (this session)

| Decision | Choice |
|---|---|
| Key scope | **Per-user encrypted vault** (not operator-shared) |
| "By code" authoring | **Minimal JSON, behind an "Advanced" disclosure**; manual grid is the front door |
| Appetite | **Full managed + BYOK, gateway included** |
| Managed upstream | **OpenRouter master account** (0% inference markup, 5.5% deposit fee) |
| Routing/metering seam | **Self-hosted LiteLLM proxy** (spend caps, virtual keys, caching) |
| BYOK path | **Bypasses the gateway** — user key → provider direct |
| Connection cardinality | **Multiple per provider** — id-based PK; existing keys migrate as each provider's first connection |
| Managed cost basis | **Keep heuristic tiers** (mini/frontier); ledger debits real tokens — promote to per-model price only if margins demand it |
| OpenRouter fee | **Folded into the credit markup** — operator owns/funds one master account; 5.5% deposit fee baked into the ~1.3–1.5× markup |
| Proxy timing | **Last (Phase E)** — managed + BYOK ship on env keys + the vault bridge first; nothing waits on the proxy |

---

## Current state (from the scouts)

| Piece | State | Anchor |
|---|---|---|
| BYOK vault (encrypt-at-rest, verify-probe) | ✅ Built | `core/billing/byok_vault.py` — Fernet/AES from `settings.byok_vault_key`; real provider `/models` probe (`_probe:325`) |
| Vault routes | ✅ Built | `core/api/routers/billing.py:284-389` (list/save/verify/remove, keyed on `username`) |
| Managed runs (env keys → dspy/LiteLLM) | ✅ Works | `service_gateway/language_models.py::build_language_model:99` |
| Real per-run credit debit (ledger truth) | ✅ Built | `worker/engine.py::_debit_run_credits:893` → `billing/service.py::debit_run:604` (grant-first, then balance; `TOKENS_PER_CREDIT=1000`) |
| `token_source` enforced server-side | ✅ Built | `submissions.py::_enforce_credit_balance:323` (BYOK skips), `_enforce_frontier_lock:352` (BYOK exempt) |
| Rich provider shape (`env_var`, `default_base_url`) | ✅ Exists | `CatalogProvider`, frontend `shared/types/api.ts:445` |
| Per-connection form (base_url + api_key → discovery) | ✅ Exists, **ephemeral per-job** | `ModelConfig.extra`, only in submit wizard; api_key stripped before persistence (`_helpers.py::strip_api_key:173`) |
| Model catalog | ✅ Dynamic | `core/api/model_catalog.py` — generated from `litellm.model_cost` + `_PROVIDER_META` (16 providers); `available` iff backend env key present (`:558`) |
| **BYOK vault → run-path bridge** | ❌ **MISSING** | `reveal_secret:296` has **no production caller** — tests only |
| Self-hosted LiteLLM proxy / OpenRouter master | ❌ Doesn't exist | dspy's bundled LiteLLM calls providers directly; no gateway |
| Provider logos / brand-icon library | ❌ None | only `lucide-react`; provider rows render `label` as plain text |

### The keystone gap

The vault **stores and verifies** keys, but **nothing reads them at run time.**
`build_language_model` (`language_models.py:99`) only gets a key from the *client-supplied*
`config.extra["api_key"]` or *process env vars* — it never calls `vault.reveal_secret`.

**Consequence:** today the entire Providers tab is cosmetic. A user can save and verify an
OpenAI key and it will never run their jobs. **The bridge is the load-bearing change** —
everything else is UI or infra layered on top of it.

---

## Design brief — the new Providers tab

**Feature summary.** Replace the 5 hardcoded BYOK rows with a flexible, LiteLLM-shaped
**connections** model: a user can connect any provider LiteLLM supports, via a guided form
or a pasted JSON block, with real brand logos. All per-user, encrypted, reusing the
existing vault + verify probe.

**Primary action.** Connect a provider key that *actually runs the user's jobs* (post-bridge).

**Data model — generalize "provider key" → "connection".**
Today: `BillingProviderKeyModel` PK `(username, provider)` storing `secret_ciphertext`,
`last4`, `status`. Generalize to a connection:

```
{ provider, label?, api_key (encrypted), api_base?, params? (jsonb), status }
```

- Drive the provider list off the **live catalog** (`CatalogProvider` already carries
  `env_var` + `default_base_url`), not the 5 hardcoded slugs.
- Schema add on `BillingProviderKeyModel`: `api_base`, `params` (jsonb), `label`; relax the
  `(username, provider)` PK to an id-based key so a user can hold multiple/custom
  connections (e.g. two OpenAI-compatible endpoints).

**Layout — manual is the front door, code is disclosed.**
- **Curated grid** of catalog providers, each a **logo card** → tap to add.
- **Add form:** key field + optional base-URL (prefilled from `default_base_url`), env-var
  name shown for familiarity.
- **"+ Custom"** card → any LiteLLM `provider` / `base_url` for the long tail.
- **Saved connections list:** logo + masked tail + verified/invalid pill + replace/remove
  (today's rows, logo-led).
- **"Advanced → paste JSON"** disclosure: minimal connection schema, parsed into the same
  vault records. Collapsed by default — preserves the "boring is a feature" ethos.

**Logos.** Add `@lobehub/icons` — purpose-built AI-provider brand marks (OpenAI, Anthropic,
Gemini, Mistral, Groq, DeepSeek, xAI, Together, …), light-mode friendly. The "looks sick"
payoff, and honest (real providers, real connections).

**Key states.** Empty (curated grid only) · has-connections · adding · verifying · invalid
(re-probe affordance) · custom · signed-out (rows still render, ungated — vault reads work
without auth).

**Design guardrail.** The JSON editor is the one piece that pushes against
"one-sentence-explainable settings." Keep it behind the disclosure; never the front door.

---

## Phased roadmap

Sequenced so a **working managed + BYOK product exists after Phase B** — the gateway is
additive, never a blocker. BYOK never touches the gateway.

### Phase A — Vault → connections
- Schema: add `api_base`, `params` (jsonb), `label` to `BillingProviderKeyModel`; id-based
  PK to allow multiple/custom connections. Alembic migration.
- Generalize `byok_vault.py` save/list/verify/remove to the connection shape; extend the
  verify probe to honor a custom `api_base`.
- Frontend: replace `ByokProviderInfo` (`lib/byok.ts`) with a catalog-driven provider shape;
  extend `byok-api.ts` + `byok-provider.tsx` to carry `api_base`/`params`/`label`.
- **Exit:** a connection with a custom base-URL saves, verifies against that endpoint, and
  round-trips masked — no run wiring yet.

### Phase B — The bridge (keystone)
- In the run path, when `token_source == "byok"`, inject
  `ProviderKeyVault.reveal_secret(username, provider)` (+ `api_base`/`params`) into the
  `ModelConfig` before `build_language_model:99`. Single seam; callers at
  `optimization/core.py:425/426,729/734,1042-1047`, `optimizers.py:423`,
  `agents/generalist.py:86`, `agents/code.py:868`.
- Resolve provider from the model string (`provider/model`); fall back cleanly when the user
  has no connection for that provider (block with a clear message, not a silent managed run).
- Tests: BYOK run uses the vault key; missing connection blocks; managed unaffected.
- **Exit:** a saved BYOK key actually runs the user's job end-to-end; managed still works.

### Phase C — Tab UI redesign
- Curated logo grid + tap-to-add form + "+ Custom" + logo-led saved-connections list.
- `@lobehub/icons`; a logo slot on each row; light-mode treatment.
- **Exit:** the Providers tab is logo-led, catalog-driven, visually on-system; manual flow
  end-to-end.

### Phase D — Advanced JSON authoring
- "Advanced → paste JSON" disclosure; parse → validate → upsert into the same vault records;
  honest per-line errors.
- **Exit:** a pasted JSON block produces the same connections as the manual form.

### Phase E — Managed gateway (additive, last)
- Stand up a **self-hosted LiteLLM proxy** (container + deploy + monitoring), configured with
  the **OpenRouter master** key; per-user/per-job **virtual keys** + **spend caps**;
  optional prompt caching as a margin lever.
- Branch `build_language_model:99` on `token_source`: managed → proxy `base_url` + virtual
  key; byok → vault key direct (Phase B).
- **Reconcile** the proxy's usage with the existing credit ledger (ledger stays source of
  truth; proxy caps are a backstop, not the accounting).
- `[FG-2]` BYOK concurrency note already in the UI; managed now has real caps to message.
- **Exit:** managed runs flow through the proxy → OpenRouter; spend caps enforced; ledger and
  proxy agree; BYOK still bypasses.

---

## Cross-cutting

- **i18n:** every new string via `msg()`; keys added to `i18n/locales/ui/he.json` (base) +
  translated across all 17 full locales; regenerate `ui-catalog.ts`; `--check` clean.
  RTL-aware (logos and masked keys stay `dir="ltr"`).
- **Security:** plaintext secrets never logged, never persisted to the job overview (the
  `strip_api_key` invariant extends to the bridge); BYOK secrets never transit the proxy
  (decision: bypass). Vault stays Fernet-at-rest; `byok_vault_key` required for any reveal.
- **Tests:** vault connection round-trip + custom base_url probe; the bridge (key reaches the
  run, missing-connection block); proxy reconciliation vs ledger; mode gating unchanged.

## Resolved decisions (2026-06-29)

All four prior open questions are now locked (see the Locked-decisions table). Their
remaining *implementation* obligations:

- **Connection PK relax → multiple per provider.** Move off `(username, provider)` to an
  id-based PK. The migration is the one real risk: it must preserve the current 5-provider
  keys, re-homing each as that provider's first connection. Back-compat: existing
  `keyFor(slug)` / verify / remove paths keep working against "the connection for this slug"
  until the UI exposes multiples.
- **Pricing → keep heuristic tiers.** No catalog price field this round;
  `billing/model_access.py:18-32` tiers stay authoritative. Revisit only if managed margins
  need per-model precision.
- **OpenRouter master → fold fee into markup.** Operator owns/funds the account; no code owns
  the funding. Code treats OpenRouter as the single managed upstream in Phase E; the 5.5%
  deposit fee is a pricing constant in the markup, not a runtime concern.
- **Proxy → Phase E, last.** A new always-on service to deploy, secure, and monitor — the real
  weight of "gateway in scope." Sequenced last so a working managed + BYOK product exists after
  Phase B and nothing else waits on it.
