# Skynet — Monetization & Offer Brief

> **Status:** Shaped, awaiting build. Captured 2026-06-28 via `/shape`.
> **Source of truth** for the launch offer, the credit/token-flow model, and the
> surfaces that carry them. Built on top of the existing `features/billing/`
> (frontend) and `core/billing/` (backend) — this completes and sharpens that
> system, it does not replace it.
> **Frameworks:** Alex Hormozi's *$100M Offers* (Value Equation, risk reversal,
> stacking, real scarcity) tempered by Skynet's trust-first ethos (transparent
> compute, no fake urgency).

---

## 1. The offer in one screen

> ## Better prompts, or you don't pay.
> *Skynet optimizes your DSPy program and proves the lift on data the optimizer
> never saw. No improvement? The run's free.*

**The Founder's Rate** — *$20/mo, locked for 12 months. Open until **[CLOSE DATE — TBD, ~late July]**.*

Founding membership includes:

- ✓ Setup-free managed optimization
- ✓ Test-split before/after proof
- ✓ **No lift, no charge** guarantee
- ✓ Hosted serving of every optimized program *(normally Premium)*
- ✓ Full version history
- ✓ 2× first-month credits
- ✓ Price locked 12 months · a direct line to the team

No fake countdown, no dollar-anchor theater — one honest deadline, a guarantee
almost no competitor can make, and a price that goes up later but never for you.

---

## 2. Positioning — the category of one

Skynet is **not** "DSPy hosting" and **not** "cheap tokens" (reselling tokens at
a markup is a race to the bottom; technical users will arbitrage it). It is:

> **The only place your prompt gets measurably better — or you don't pay for the run.**

Local DSPy/GEPA has the *same* dream outcome and the *same* likelihood. Skynet's
entire reason to exist is that it destroys the **denominator** of the Value
Equation — the time and effort half — which is the one thing running it yourself
can't fix.

---

## 3. The Value Equation (why it works)

`Value = (Dream Outcome × Perceived Likelihood) / (Time Delay × Effort & Sacrifice)`

| Lever | The move | Where it shows up in product |
|---|---|---|
| **Dream outcome ↑** | Ship a measurably better program *with the proof* (status: "I improved the metric, and here's the chart") | Hero promise, before/after chart |
| **Likelihood ↑** | Proof measured on a **test split the optimizer never saw** + the guarantee | Result/proof screen, "No lift, no charge" |
| **Time delay ↓** | Pre-baked demo magic in seconds → instant baseline score on *their* data → free first run | Onboarding |
| **Effort ↓** | Managed by default (no key juggling, no setup), **serving included** (done-for-you, not "here's an artifact, go deploy") | One-tap onboarding, hosted serving |

The strategic bet: **win the denominator.** That's the moat, and the offer leans
on it everywhere.

---

## 4. The guarantee — full spec ("No lift, no charge")

The load-bearing wall of the offer. Specific, measurable by the platform itself,
and structured so it can't be gamed.

- **Trigger basis:** baseline vs. optimized score on the **test split** — the
  slice the optimizer never sees (not the train/feedback set it mutates on, not
  the valset it *selects* against). A gain there is unbiased — no overfitting,
  no winner's-curse. Strongest possible "perceived likelihood" line: *"we beat
  your baseline on data the optimizer never saw."*
- **Threshold:** any real improvement on the test split counts as lift → the run
  is billed. (The test-split basis is what makes "any improvement" honest rather
  than a noise gotcha.)
- **Refund scope:** if there's no lift, the **entire run is free** — baseline
  eval + optimization + scoring, all refunded. One clean line, no deductions.
- **Coverage / anti-gaming:** the **first optimization per task** carries the
  guarantee. Re-runs on the same task bill normally (the badge honestly shows as
  off). Stops fishing for repeated free compute on an already-good program.
- **Graceful fallback:** when a dataset is too small to reserve a meaningful test
  split, fall back to the **valset** gain — clearly labeled which basis was used
  — rather than refusing the guarantee. Protects the tiny first-run / demo case.
  Power users may supply their own test set.
- **Managed vs BYOK:** **full-strength on managed** (we can refund the whole
  run). On **BYOK** we only refund our **platform fee** — the provider tokens are
  already spent, since we never touched them. This is an honest reason managed is
  the better deal, and reinforces "tokens flow through the platform by default."
- **Framing — billing as proof:** when there *is* lift, the charge is the
  receipt that it worked: *"We beat your baseline +6.2% on held-out data — that's
  why this run was billed."* When there isn't: *"No lift — this run was free,
  312 credits refunded."* A product failure becomes a trust win.

---

## 5. Locked decisions (decision log)

| # | Decision | Choice | Rationale (Hormozi lens) |
|---|---|---|---|
| 1 | Pricing architecture | **Keep what's built** + Founder's Rate on top | Ships fastest; ~70% already wired (Stripe, packs, gating) |
| 2 | Guarantee | **No lift, no charge** (full spec §4) | #1 conversion lever is reversing the deepest fear |
| 3 | Lift basis | **Test-split gain**, graceful fallback to valset | Unbiased → unimpeachable proof; "any improvement" stays honest |
| 4 | Guarantee coverage | **First run per task**; whole run refunded | Caps exposure without weakening the line |
| 5 | Onboarding | Managed/BYOK choice, **managed pre-selected, one-tap** | Keep trust (explicit cost) without taxing time-to-first-win |
| 6 | Token flow | **Managed by default**, BYOK optional | The single biggest effort-remover in the product |
| 7 | Value-stack tone | **Credible itemization** (no fake $ anchors) | Stacks value in a register engineers trust |
| 8 | Scarcity | **Deadline only** (close date TBD); drop the hard 100 | One clean, real scarcity line |
| 9 | Offer name | **The Founder's Rate** | Benefit-led; names what they're locking |
| 10 | Hero promise | **Guarantee on top** + outcome/proof subhead | The guarantee is the wedge no one else can copy |
| 11 | First win | **Both**: pre-baked demo → instant baseline on their data | Demo = "does it work?", baseline = "what's it worth to me?" |
| 12 | Free-grant reset | **Rolling 30-day per-user anchor, non-cumulative, lazy-eval** | Smooths infra load AND prevents hoarding — the two properties are separable |

---

## 6. Design brief — surfaces

Extend, don't reinvent. Reuse `SettingsRow` / `Tabs` / `Popover` / the shell
header slot. Gold accent (`#C8A882`) only on the **one** primary affordance per
view. All money/credit copy through `msg()` so it translates across 24 locales.
Warm, precise, premium — calm factual microcopy, never salesy.

1. **Onboarding first-run** *(new)*
   Pre-baked demo before/after → managed/BYOK as a **one-tap, managed-preselected**
   choice → upload → **instant baseline score** on their data → "Optimize — your
   first run is free." The time-delay collapse.

2. **The Founder's Rate page** *(evolve `/upgrade` + `UpgradeView`)*
   Hero promise (guarantee on top), credible-itemized stack, a calm deadline line,
   `$20/mo locked` CTA → Stripe subscription. Credit packs demote to a "need more
   credits?" section below — no longer the headline.

3. **Pre-run bracket + cost ceiling + guarantee badge** *(at `ModelConfigModal` / `ModelStep`)*
   A DSPy job's token use isn't linear (bootstrapping, compile steps, dataset size,
   validation loops), so don't promise a tight estimate — show a **projected
   bracket** and let the user set a **Max Cost Ceiling** before running. The run
   button carries it: `Run (cap: 540 credits)`. Plus the guarantee badge —
   "**first run on this task is guaranteed**"; re-runs show it honestly off.

4. **The proof moment** *(new, on the result)*
   The trust climax. Either *"We beat your baseline +6.2% on held-out data — run
   billed"* or *"No lift — this run was free. 312 credits refunded."* Billing
   becomes the evidence the product worked.

5. **Wallet / ledger** *(extend `WalletTab`)*
   Guarantee refunds appear as ledger rows ("no lift — refunded"); a Founder badge
   + "rate locked through [date + 1yr]."

---

## 7. Key states to design

free / never-run · ran-with-lift · ran-no-lift (refunded) · founder · low balance
· empty balance · managed mode · BYOK mode · frontier-locked · guarantee-eligible
(first run on task) vs re-run · deadline-active vs deadline-passed · **balance-syncing
(post-checkout)**.

---

## 8. Backend dependencies (what the offer requires to actually work)

Design is UX-first, but these are the real dependencies behind the surfaces:

- **Server-side credit debiting + `run` ledger rows.** Today the free grant always
  reads "full" (200/200) and spend only meters to Stripe — nothing debits the
  local ledger. The guarantee needs genuine debit + refund.
- **Guarantee adjudication.** Compare baseline vs optimized **test-split** score,
  auto-refund on no lift, track "first run per task."
- **Mode propagation + enforcement.** Managed/BYOK is client-only today; it must
  reach the backend so frontier-locking and the guarantee are *enforced*, not
  advisory.
- **The Founder's Rate.** A Stripe price + the 12-month price-lock + the deadline
  gate.
- **BYOK vault.** Real encrypt/verify (currently an in-memory stub) — later, not
  launch-critical because managed is the default.

---

## 9. Built pricing reference (do not drift from these without re-deciding)

- **1 credit = $0.01** (`CREDIT_USD_VALUE`).
- **Free grant:** 200 credits, **rolling 30-day per-user window, non-cumulative**
  (tops up to a flat 200; leftover expires — no banking), mini models only via gating.
  Reset is lazy-evaluated on wallet read (`now > grant_reset_at` → top up, advance).
- **Premium:** $20 / mo → frontier models + hosted serving + priority queue.
- **Credit packs (prepaid top-ups):** 500 / $5 · 2,200 / $20 (popular) · 6,500 / $50.
- **Model tiers:** `mini` (free) vs `frontier` (locked in managed until paid/Premium;
  BYOK never locks).
- **Markup** lives in the credit→cost mapping / Stripe per-unit price, re-priceable
  without touching the catalog. Compute stays near pass-through (trust); the
  subscription is where margin lives.

---

## 10. Open items

- [ ] **Close date** for the Founder's Rate (the one missing input).
- [ ] Exact wording of the noise-floor fallback label (when valset is used).
- [ ] "Direct line to the team" mechanism for founders (channel? email?).
- [ ] Post-deadline price (what "the rate goes up" actually becomes).

---

## 11. Friction guards (peer review, 2026-06-28)

Tracked as work items in `docs/monetization-implementation-plan.md`.

1. **DSPy estimate illusion.** Token use in an iterative optimizer isn't linear
   (bootstrapping, compile steps, dataset size, validation loops). Don't promise a
   tight number — show a **projected bracket** and let the user set a **Max Cost
   Ceiling** before running (`Run (cap: 50 credits)`). The ceiling is both the
   honest-estimate fix and a hard guardrail.
2. **BYOK concurrency messaging.** When a BYOK run queues, state *why* (self-hosted
   LiteLLM concurrency limits) so it never reads as an arbitrary penalty for not
   buying credits.
3. **Post-checkout balance sync.** Stripe webhooks aren't instant. On return from
   checkout, the balance chip enters a quiet **`syncing`** state (shimmer over the
   prior balance) rather than a stale/empty number — no false "zero balance" panic
   while the FastAPI→Postgres seam catches up.
