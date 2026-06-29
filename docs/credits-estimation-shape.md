# Design Brief — Credits Estimation Per Run

> Produced via `/shape` (discovery interview + codebase flow-map). Design-planning
> artifact — **no code written yet**. Hand off to `/impeccable craft` or any
> implementation flow. Source of truth for the feature's UX/flow decisions.

## 0. Decisions locked in discovery

| Decision | Choice |
|---|---|
| Estimate framing | **Low–high range + hard cap** (the range *is* the message; honest about variance) |
| Accuracy engine | **Per-model, per-token cost model** — "the best estimate we can give" (founder directive) |
| Charging basis | **Align charging to the same per-model micro-dollar basis** so estimate ↔ actual reconcile by construction (see §2 — key decision) |
| Balance gating | **Display + clamp cap** — gate on balance, clamp the cost ceiling, charge actuals; **no holds/reservations** |
| Reconciliation | **Prominent** estimate-vs-actual after the run (loud trust loop, not a buried line) |

---

## 1. Feature Summary
Replace the existing coarse cost *bracket* with a **model-aware, per-token credit estimate**: for the model(s) a run actually uses, project input/output token volume, price it against each model's real per-token cost, apply the platform markup, and present a low–high credit range with a hard cap. Shown wherever the run's cost is decided, mode-aware (managed vs BYOK), gating the balance without holds, and **prominently reconciled against the actual charge** after the run. Delivers the product's load-bearing principle — *show cost before commitment and after, no surprise burn* — for a workload whose token use is genuinely variable.

## 2. ⭐ Key Architectural Decision — per-model cost ledger drives estimate *and* charge
The current charge is **flat and model-blind**: `debit_run` → `credits_for_tokens(total_tokens)` = `ceil(total_tokens / 1000)` (`backend/core/billing/service.py`, `TOKENS_PER_CREDIT = 1000`). A frontier model and a mini that burn the same tokens cost the same credits; the platform absorbs the per-model price gap.

The founder directive ("per model and per tokens, the best estimate we can give") requires a per-model engine. For the estimate to be *honest* — i.e. to actually match the bill, which the **prominent reconciliation** will display side-by-side — the **charge must use the same per-model basis**. Otherwise we advertise our own inaccuracy.

**Adopted model (recommended):** a single per-model micro-dollar cost function is the source of truth for both paths —

```
cost_micro_usd(model, in_tokens, out_tokens)
    = in_tokens  * input_cost_per_token(model)
    + out_tokens * output_cost_per_token(model)       # + reasoning-token cost where applicable
credits = ceil( cost_micro_usd * MARKUP / CREDIT_USD_VALUE )   # CREDIT_USD_VALUE = $0.01
```

- **Estimate** = the function on *projected* low/high token volumes per model → low/high credit band.
- **Charge** = the same function on *actual* tokens from `total_tokens_from_history` → debited credits.
- They reconcile by construction: same formula, projected vs. actual inputs. The flat `TOKENS_PER_CREDIT = 1000` is retired in favour of `per-model $/token × markup`; markup stays the single re-priceable margin lever.

**Per-model price data is already in the process** — LiteLLM's `model_cost` registry (read at `backend/core/api/model_catalog.py:496`) carries `input_cost_per_token` / `output_cost_per_token` per model; the catalog build loop (`model_catalog.py:517`) reads `meta` but currently drops those fields. Surfacing them onto `CatalogModel` (and into a backend pricing helper) is the enabling change.

> **If the founder wants estimate-only and to keep flat charging:** then the estimate must be **token-volume-aware, not $/token-aware** (credits would still be flat per token), and reconciliation should soften its precision claims. This brief assumes the aligned path; flag at handoff if that's wrong.

## 3. The Estimate Engine (replaces `frontend/src/features/submit/lib/cost-bracket.ts`)
Today's `projectCostBracket` is model-blind: `AUTO_METRIC_CALLS {500/2000/8000} × 700–4500 tokens/call × rowFactor × reflection × pairs ÷ 1000`. Keep its *shape* (budget × per-call tokens × dataset factor × sweep, as a low/high band) but make every term per-model:

- **Per-call token split.** Project input vs. output tokens per metric call, not a blended count. Output bound informed by the model's `max_tokens` (user-set in `ModelConfig`) and typical verbosity; input informed by signature + (optionally) a **sampled** average row length from the dataset.
- **Reasoning multiplier.** `supports_thinking` (already on `CatalogModel`) is the single biggest volume lever — thinking tokens can dominate. Apply a per-tier multiplier to projected output tokens (calibrate from historical actuals).
- **Per-model price.** Multiply projected tokens by that model's real `input/output_cost_per_token`; sum task-model + reflection-model contributions; for grid search, sum across all `gen × refl` pairs.
- **Markup → credits.** Convert micro-dollars → credits via markup and `CREDIT_USD_VALUE`. Low/high band preserved (≈the honest spread; tighter than today because it's no longer averaging across all model tiers).
- **Pure & testable**, like the current module. Mirror the canonical math in a backend helper so estimate and charge can't drift.

## 4. Primary User Action
Before committing a run, the user understands **"what will this cost me, roughly, and what's the worst case"** with enough confidence to proceed, cap it, or change the model/budget — never blindsided by the actual charge.

## 5. Design Direction
Warm / precise / premium (`.impeccable.md`). The **range is the message** — honesty over false precision; calm factual microcopy ("estimate, not a quote"), never salesy urgency. Gold `#C8A882` spent only on the cap/headline affordance. Extend the established vocabulary — `CostCeilingCard`, `CreditBalanceChip`, `ProofMoment`, `RunCreditsChip` — so it reads native, not bolted on. Light theme, RTL-safe, LTR-islanded numerals, all copy through i18n across 24 locales.

## 6. Layout Strategy
- **Estimate lives where cost is decided** — keep it on the model step (`ModelStep.tsx` → `CostCeilingCard.tsx`), and **promote a compact recap to the review/summary step** (`SummaryStep.tsx`) so the last thing seen before launch is the number. Run button keeps echoing the cap (`SubmitNav.tsx:76`).
- **Card anatomy:** range headline + $-equivalent → **balance-relative line** ("≈ 180–320 of your 1,720 cr") → **hard-cap control** (seeded, editable) → mode/model sub-line.
- Balance framing makes headroom visible without a new widget — reuses the `CreditBalanceChip` number style. Live balance is real: `CreditProvider` fetches `/billing/wallet` (`credit-provider.tsx`), `STUB_WALLET` only seeds/falls back.

## 7. Key States
| State | What the user sees / feels |
|---|---|
| Config incomplete (no dataset) | Provisional estimate + "add data to sharpen" — never blank |
| **Managed** | Full credit range + cap; the default |
| **BYOK** *(newly shown)* | Small **platform-fee** range ("≈ 36–64 cr platform fee — your key pays the model"); previously hidden (`ModelStep.tsx:282`) |
| **Reasoning model** | Visibly wider/higher band + one-line "thinking models burn more tokens" note |
| Grid search | Range × (gen×refl) pairs, clearly "swept across N pairs" |
| Estimate ≤ balance | Calm, neutral |
| High-end > balance | Warn + auto-clamp cap to balance + upgrade nudge — **display + clamp, never block** |
| **Post-run reconciliation** | Prominent "estimated 180–320 → **actual 210**": *as estimated* / *came in under* / *capped at 400* |
| Loading catalog | Skeleton, no layout shift |

## 8. Interaction Model
- **Live recompute** (debounced, optimistic) as model, `autoLevel`/`max_full_evals`, dataset size, reflection toggle, or grid pairs change (`use-submit-wizard.ts` `costBracket` memo at :396).
- **Cap:** default seeded from `defaultCeilingForBracket` (high × 1.15), editable, clamped to balance via `cost_ceiling_budget` / `_cap_cost_ceiling_to_balance` (`submissions.py:361`).
- **Mode toggle** (`TokenSourceToggle.tsx`) flips the whole framing managed↔BYOK in place.
- **Post-run:** `ProofMoment` (`OptimizationDetailView.tsx:1149`) gains a dedicated estimate-vs-actual block; estimate must be **persisted on the run record** at submit so it's available to compare against `result.details.billing.credits`.

## 9. Content Requirements (all i18n, 24 locales)
New/changed keys: range label, $-equiv, **"estimate not a quote" disclaimer**, cap label + "stops at X", BYOK platform-fee framing, reasoning-token note, balance-headroom line, over-balance warning, and the three reconciliation verdicts (as-estimated / under / capped). RTL-safe; numerals LTR-islanded; reuse `formatCredits` / `creditsToUsd`. New money copy flows through the `i18n/locales/ui/<locale>.json` → `generate_i18n.py` pipeline.

## 10. Appetite & Boundaries
- **Appetite:** ~2 weeks (medium–large). It's now an *engine + charging-basis* change plus 3 surfaces plus i18n — bigger than the estimate-only framing, because per-model accuracy pulls the charge path in with it.
- **In scope:** per-model/per-token estimate engine (front + backend mirror), per-model charging on the same basis, surfacing LiteLLM `$/token` onto `CatalogModel`, range+cap UI, BYOK estimate, prominent reconciliation (with persisted estimate), balance-relative framing.
- **Rabbit holes to avoid:** full per-row tokenization of the dataset (sample, don't tokenize everything); a live upstream price-sync service (snapshot LiteLLM's registry, re-pull on deploy); holds/reservations (explicitly out).
- **No-gos:** model-blind estimates, blocking runs (we display + clamp), surprise charges, hiding BYOK cost, a precise estimate reconciled against a flat charge (the mismatch trap §2 guards against).

## 11. Open Questions (resolve in build)
1. **Markup value & per-model floor.** Per-model pricing replaces flat `TOKENS_PER_CREDIT`; confirm the markup multiplier and whether a per-run minimum credit floor still applies.
2. **Reasoning/thinking-token multipliers** need calibration — backfill defaults from historical actuals (`RunCreditsChip` / wallet ledger).
3. **Input-length sampling** — sample N dataset rows to estimate avg input tokens, or keep `rowFactor` heuristic? (Accuracy vs. submit-time latency.)
4. **Reconciliation home** — extend `ProofMoment` vs. a new sibling element; where the persisted estimate is stored on the run/result model.

## 12. Flow-map anchors (for the implementer)
- **Run wizard:** `frontend/src/app/submit/page.tsx`; hook `frontend/src/features/submit/hooks/use-submit-wizard.ts`; steps in `frontend/src/features/submit/components/steps/`; run button `SubmitNav.tsx`.
- **Estimate today:** `frontend/src/features/submit/lib/cost-bracket.ts` → `CostCeilingCard.tsx` (managed-only, `ModelStep.tsx:282`).
- **Submit API:** `backend/core/api/routers/submissions.py` (`submit_job` :447, `submit_grid_search` :603); models `backend/core/models/submissions.py` (`RunRequest`, `_OptimizationRequestBase`, `token_source`, `max_cost_credits`); credit gating `_enforce_credit_balance` :326, `_cap_cost_ceiling_to_balance` :361.
- **Billing primitives:** `backend/core/billing/service.py` (`credits_for_tokens` :135, `platform_fee_credits` :173, `cost_ceiling_budget` :194, `debit_run` :641, `report_run_usage` :1015); frontend mirror `frontend/src/features/billing/lib/credit.ts`.
- **Model catalog / pricing source:** `backend/core/api/model_catalog.py` (`litellm.model_cost` :496, build loop dropping price :517, `CatalogModel` :29); picker `frontend/src/features/submit/components/ModelPicker.tsx`.
- **Actual cost after run:** `total_tokens_from_history` (`backend/core/service_gateway/optimization/core.py`); worker `_debit_run_credits` / `_stamp_billing_outcome` (`backend/core/worker/engine.py`); UI `RunCreditsChip.tsx`, `ProofMoment.tsx`, `proof-banner.ts`.
- **Balance UI:** `CreditBalanceChip.tsx` (`app-shell.tsx:216`), `WalletTab.tsx`, `credit-provider.tsx` (live `/billing/wallet`).

## 13. Recommended impeccable references
`ux-writing.md` (the honesty microcopy — highest leverage), `interaction-design.md` (debounced live recompute, optimistic feedback), `spatial-design.md` (card rhythm + review-step recap), `motion-design.md` (gentle recompute transition + reconciliation reveal).
