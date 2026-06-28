# Skynet — Monetization Implementation Plan

> **Derived from** `docs/monetization-offer.md` (the shaped offer + design brief).
> **Status:** plan only — no code written yet. Sequenced by leverage × dependency.
> Captured 2026-06-28.

## Sequencing principles

1. **Ledger truth first.** Almost everything (the guarantee, real estimates, caps,
   the proof moment) depends on credits actually being debited server-side. Today
   they aren't — the free grant always reads full and spend only meters to Stripe.
   This is the foundation; build it before any UI promises depend on it.
2. **Ship the offer's emotional core early.** The proof moment and the free first
   run are where the offer becomes real or doesn't. Get them in front of users
   before the polish.
3. **Extend the existing modules.** `features/billing/` (FE) and `core/billing/`
   (BE) are ~70% there. Complete and sharpen — no rip-and-replace.
4. **Friction guards are folded into the phase they belong to** (tagged `[FG-n]`),
   not bolted on at the end.

Friction guards (from peer review, see offer brief §11):
- `[FG-1]` DSPy estimate illusion → projected bracket + user-set Max Cost Ceiling.
- `[FG-2]` BYOK concurrency messaging → explain *why* the queue exists.
- `[FG-3]` Post-checkout balance sync → `syncing` chip state, no false zero.

---

## Phase 0 — Credit ledger truth (foundation)

The missing backbone. Without it the guarantee and estimates are cosmetic.

- **Debit on run completion.** In the worker's once-only completion claim
  (`core/worker/engine.py`, next to `_report_run_usage_best_effort`), write a
  signed `run` row to `credit_ledger` and decrement `billing_customers.credit_balance`.
- **Wallet reflects real spend.** `core/billing/service.py::get_wallet` stops
  reporting the free grant at a constant 200/200 and returns the real remaining
  figure from the ledger.
- **Free-grant reset — rolling, non-cumulative, lazy.** Add `grant_reset_at` (and a
  grant-remaining figure) to `BillingCustomerModel`. On wallet read: if
  `now > grant_reset_at`, top up to a flat 200 (leftover expires — no banking) and
  set `grant_reset_at = now + 30d`. No cron; resets scatter across the month.
  Paid users anchor to the Stripe subscription period instead. *(Decision §5.12)*
- **Enforce, don't just display.** Add a balance/grant gate at submit so a depleted
  account can't silently run on managed compute.
- **Tests:** extend `core/billing/tests/test_service.py` and
  `core/worker/tests/test_engine_metering.py` — debit math, rolling reset edges,
  non-cumulative expiry, the submit gate.

**Exit criteria:** a managed run visibly decrements the wallet; the grant resets on
a per-user rolling date; an empty account is blocked at submit.

---

## Phase 1 — The guarantee engine ("No lift, no charge")

Backend adjudication of the offer's load-bearing promise (offer brief §4).

- **Test-split comparison.** Score the *baseline* program and the *optimized*
  program on the held-out **test split** (already present — `splits.*` in
  `service_gateway/optimization/core.py`, `_test_split_indices` in
  `api/routers/share.py`). Lift = optimized > baseline on that split.
- **Graceful fallback.** Dataset too small for a real test split → fall back to the
  **valset** gain and label which basis was used (don't refuse the guarantee).
- **First-run-per-task tracking.** Persist which `(user, task)` pairs have spent
  their one guaranteed run. The first is covered; re-runs bill regardless.
- **Auto-refund.** No lift on a guaranteed run → write an offsetting `run` refund
  row for the **entire** run (eval + optimization + scoring) and restore the balance.
- **Managed vs BYOK.** Managed → full-run refund. BYOK → refund our platform fee
  only (provider tokens were never ours to refund).
- **Mode propagation.** The wizard's managed/BYOK mode is client-only today
  (`TokenSourceToggle`); thread it into the submit payload so frontier-locking and
  the guarantee are *enforced* server-side, not advisory.

**Exit criteria:** a no-lift guaranteed run lands a refund row and a restored
balance; a re-run on the same task bills; mode reaches the backend and gates frontier.

---

## Phase 2 — Pre-run bracket + cost ceiling `[FG-1]`

- Replace the tight pre-run estimate with a **projected credit bracket**.
- Add a user-set **Max Cost Ceiling**; run button reads `Run (cap: 540 credits)`.
- Backend hard-stops the job at the ceiling — the ceiling is simultaneously the
  honest-estimate fix *and* the per-job spend guardrail from the guardrails plan.
- Surfaces: `features/submit/.../ModelConfigModal.tsx`, `steps/ModelStep.tsx`.

**Exit criteria:** user sets a cap pre-run; the job cannot exceed it; the UI never
implies a false-precision single number.

---

## Phase 3 — The proof moment (frontend — highest emotional leverage)

- New result treatment: **lift** → "We beat your baseline +6.2% on held-out data —
  run billed"; **no lift** → "No lift — this run was free. 312 credits refunded."
- Billing becomes the evidence the product worked. Reuse existing result components;
  pull every string through `msg()`.

**Exit criteria:** both outcomes render with the billing-as-proof framing; refunds
show in the wallet ledger as legible rows ("no lift — refunded").

---

## Phase 4 — Onboarding first-run (Time-Delay collapse)

- **Pre-baked demo** before/after on a sample task (feel the magic in seconds).
- **Managed/BYOK** as a **one-tap, managed-preselected** choice (no first-win tax).
- Upload → **instant baseline score** on their data (creates the personal gap).
- "Optimize — your first run is free" → into the guaranteed first run.

**Exit criteria:** a new user reaches a real before/after (demo, then their own
baseline) without a payment wall, and lands in the free first run.

---

## Phase 5 — The Founder's Rate page

- Evolve `frontend/src/app/upgrade/page.tsx` + `UpgradeView.tsx`: hero promise
  (**guarantee on top**), credible-itemized stack, calm deadline line, `$20/mo
  locked` CTA → Stripe subscription. Credit packs demote to a "need more credits?"
  section.
- Backend: a **Founder's Rate** Stripe price + the 12-month **price-lock** + the
  **deadline gate** (offer unavailable after the close date).
- `[FG-3]` On return from checkout, the balance chip (`CreditBalanceChip`) enters a
  **`syncing`** state — a quiet shimmer over the prior balance — until the webhook
  lands, instead of flashing a stale/empty number.

**Exit criteria:** a user subscribes to the Founder's Rate, the lock + deadline are
honored, and the post-checkout return never shows a false zero.

---

## Phase 6 — Friction polish

- `[FG-2]` **BYOK concurrency messaging.** Where a BYOK run queues, state the reason
  (self-hosted LiteLLM concurrency limits) so it never reads as a pay-to-skip penalty.
- Low-balance **no-alarm** treatment (validated): operational metric, not flashing red.
- Empty/low/healthy and `syncing` chip states finalized across `CreditBalanceChip`
  and `WalletTab`.

---

## Phase 7 — BYOK vault (post-launch)

- Replace the in-memory stub (`byok-provider.tsx`, `lib/byok.ts`) with a real
  encrypt-at-rest vault + a verify probe + backend routes.
- Not launch-critical: managed is the default, so BYOK can follow the offer.

---

## Cross-cutting

- **i18n:** every money/credit string via `msg()`; new keys added to the catalog
  and translated across the 24 locales (RTL-aware).
- **Trust copy:** calm, factual microcopy — no aggressive/high-pressure language
  (it would shatter the "quiet instrument panel" ethos).
- **Tests:** backend ledger + guarantee unit tests; reset-cadence edges; submit gate.
- **One open input:** the Founder's Rate **close date** (the deadline gate in P5).
