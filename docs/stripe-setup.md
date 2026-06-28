# Stripe billing setup

This wires Skynet's managed-credit billing to a real Stripe account: prepaid
credit packs, a Premium subscription, the Stripe-hosted customer portal, and
webhook reconciliation into the `billing_customers` / `credit_ledger` tables.

You run the account steps (Stripe can't be driven on your behalf); the code is
already in place. Start in **test mode** — every step below uses test keys and
test cards. Nothing charges a real card until you switch to live keys (last
section).

## What's already built

| Piece | Where |
|---|---|
| Config (`STRIPE_*` env) | `backend/core/config.py`, `backend/.env.example` |
| DB tables (customer link, credit ledger, webhook idempotency) | `backend/core/storage/models.py`, migration `f0a1b2c3d4e5_add_billing_tables.py` |
| Stripe service (checkout, subscription, portal, webhook sync) | `backend/core/billing/service.py` |
| API (`/billing/wallet`, `/checkout`, `/subscribe`, `/portal`, `/webhook`) | `backend/core/api/routers/billing.py` |
| Frontend (real Buy/subscribe/manage, live wallet) | `frontend/src/features/billing/*` |
| Provisioning script | `backend/scripts/provision_stripe.py` |

If `STRIPE_SECRET_KEY` is unset the app still runs: the wallet reads as a free
tier and every purchase route returns `503`. So you can deploy first and turn
billing on later.

## 1. Create a Stripe account

1. Sign up at <https://dashboard.stripe.com/register>.
2. Confirm you're in **Test mode** — the toggle is top-right of the dashboard.
   (You don't need to "activate" the account or submit business details to use
   test mode.)

## 2. Get your test API keys

1. Dashboard → **Developers → API keys**.
2. Copy the **Secret key** (`sk_test_…`). The publishable key is **not** needed —
   the browser never talks to Stripe directly; it follows a URL the backend mints.
3. Put it in `backend/.env`:

   ```bash
   STRIPE_SECRET_KEY=sk_test_...
   APP_PUBLIC_URL=http://localhost:3000   # where Checkout returns the buyer
   ```

## 3. Provision products, prices, and the meter

From `backend/` with the venv active:

```bash
python scripts/provision_stripe.py
```

It creates (idempotently — safe to re-run) three one-time credit-pack prices
($5 / $20 / $50), the $20/mo Premium subscription price, and an optional
usage meter, then prints the env lines. Paste them into `backend/.env`:

```bash
STRIPE_PRICE_PACK_STARTER=price_...
STRIPE_PRICE_PACK_PLUS=price_...
STRIPE_PRICE_PACK_PRO=price_...
STRIPE_PRICE_PREMIUM=price_...
# STRIPE_PRICE_METERED=price_...        # only if the meter provisioned
# STRIPE_METER_EVENT_NAME=skynet_tokens
```

> The **credits** each pack grants (500 / 2200 / 6500) live in
> `core/billing/service.py::PACK_CREDITS`. Stripe only holds the dollar price, so
> you can re-price or change the markup without touching Stripe.

## 4. Wire the webhook

The webhook is how a completed payment actually credits the account. The
endpoint is `POST /billing/webhook` and it verifies Stripe's signature, so it
needs the signing secret.

### Local development — Stripe CLI

```bash
brew install stripe/stripe-cli/stripe        # or see stripe.com/docs/stripe-cli
stripe login
stripe listen --forward-to localhost:8000/billing/webhook
```

`stripe listen` prints a signing secret (`whsec_…`). Put it in `backend/.env`:

```bash
STRIPE_WEBHOOK_SECRET=whsec_...
```

Leave `stripe listen` running while you test — it forwards live test events to
your local backend.

### Production — dashboard endpoint

1. Dashboard → **Developers → Webhooks → Add endpoint**.
2. URL: `https://<your-host>/billing/webhook`.
3. Subscribe to these events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy the endpoint's **Signing secret** (`whsec_…`) into the deployment's
   `STRIPE_WEBHOOK_SECRET`.

## 5. Enable the customer portal

For the Settings → Billing **"Manage subscription"** button (`/billing/portal`)
to work, activate the portal once: Dashboard → **Settings → Billing → Customer
portal** → **Activate**. Allow plan cancellation and payment-method updates.

## 6. Run the migration

```bash
cd backend && python manage.py setup        # runs alembic upgrade head
# or directly:  alembic upgrade head
```

This creates `billing_customers`, `credit_ledger`, and `billing_webhook_events`.
(The app also builds them via `create_all` on boot, so a fresh DB needs no
manual step — the migration is for existing databases.)

## 7. Test it

1. Restart the backend so it picks up the new env.
2. In the app: **Upgrade** page → buy a pack, or **Settings → Billing → Go
   Premium**.
3. On the Stripe Checkout page use a test card: `4242 4242 4242 4242`, any
   future expiry, any CVC, any ZIP.
4. You're redirected back to `/upgrade?status=success`. Within a second or two
   `stripe listen` (or the dashboard endpoint) delivers
   `checkout.session.completed`, the webhook credits the ledger, and the wallet
   balance updates on the next fetch.

Watch events live in Dashboard → **Developers → Events**, or in the
`stripe listen` terminal.

## Metered overage

Usage-based overage is **wired end to end**. Every successful optimization now
reports its token usage to the Stripe meter:

- The worker captures per-run tokens from the LM call history
  (`total_tokens_from_history`) into the run result, alongside `num_lm_calls`.
- On success, `BackgroundWorker._report_run_usage_best_effort` calls
  `StripeBillingService.report_run_usage(username, tokens)` — on a daemon thread
  (so a slow Stripe call never stalls the worker) and guarded by the same
  once-only completion claim as the finished-job notification, so a re-run or
  redelivery is never double-billed.
- `report_run_usage` floors tokens to whole meter units
  (`METER_UNIT_TOKENS = 1000`, i.e. **1 unit = 1000 tokens**) and pushes a meter
  event for the account's Stripe customer. The dollar value per unit lives on the
  metered price (`scripts/provision_stripe.py`), so you re-price the markup in
  Stripe without code changes.

**To turn recorded usage into actual charges**, the metered price must be on the
customer's subscription — recording a meter event only bills a customer whose
subscription carries that metered price. Add the metered price as a second item
on the Premium subscription (Dashboard → the Premium product, or via the
subscription's items). For everyone else the meter events are recorded as usage
analytics and never charged — which is exactly what you want for the free tier.

What metering deliberately does **not** do:

- **No Stripe-customer sprawl.** Usage is metered only for accounts that already
  have a billing customer (bought a pack or subscribed). A free-tier user who
  never touched billing gets no Stripe customer created just to record usage.
- **No local credit debit.** Metered usage flows to Stripe; it does not decrement
  the prepaid `credit_balance` (packs) or the free grant. Those remain a separate
  prepaid concept. If you want runs to draw down purchased credits instead of (or
  before) metering, that's a follow-up: convert tokens to credits and debit the
  ledger in `report_run_usage`.

Caveats to know:

- Token capture reads the `usage` block dspy/LiteLLM records on each LM call. A
  provider that returns no usage yields `None` (the run meters nothing rather
  than billing zero). A resumed run only counts the tokens of its current
  process.
- **BYOK is not discriminated server-side.** The backend `ModelConfig` has no
  per-request key field today, so every successful run is treated as managed and
  metered. When a real BYOK/gateway-bypass path lands, skip metering for it.

Prepaid credit packs + Premium remain the primary billing paths; metered overage
sits on top for accounts that opt into it via the metered subscription item.

## Going live

1. Toggle the dashboard to **Live mode** and grab live keys (`sk_live_…`).
2. Re-run `python scripts/provision_stripe.py` against the live account (it
   creates live prices; paste the new `price_…` ids).
3. Add a **live** webhook endpoint (step 4) and use its live signing secret.
4. Swap `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_*` in the
   production environment. Done.
