# Stripe billing setup

This wires Skynet's managed-credit billing to a real Stripe account: prepaid
pay-as-you-go credit packs and webhook reconciliation into the
`billing_customers` / `credit_ledger` tables.

You run the account steps (Stripe can't be driven on your behalf); the code is
already in place. Start in **test mode** — every step below uses test keys and
test cards. Nothing charges a real card until you switch to live keys (last
section).

## What's already built

| Piece | Where |
|---|---|
| Config (`STRIPE_*` env) | `backend/core/config.py`, `backend/.env.example` |
| DB tables (customer link, credit ledger, webhook idempotency) | `backend/core/storage/models.py`, migration `f0a1b2c3d4e5_add_billing_tables.py` |
| Stripe service (pack checkout, webhook sync) | `backend/core/billing/service.py` |
| API (`/billing/wallet`, `/checkout`, `/webhook`) | `backend/core/api/routers/billing.py` |
| Frontend (real Buy, live wallet) | `frontend/src/features/billing/*` |
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

## 3. Provision products and prices

From `backend/` with the venv active:

```bash
python scripts/provision_stripe.py
```

It creates (idempotently — safe to re-run) three one-time credit-pack prices
($5 / $20 / $50), then prints the env lines. Paste them into `backend/.env`:

```bash
STRIPE_PRICE_PACK_STARTER=price_...
STRIPE_PRICE_PACK_PLUS=price_...
STRIPE_PRICE_PACK_PRO=price_...
```

> The **credits** each pack grants (500 / 2000 / 5000 — at par, one credit per
> cent) live in
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
3. Subscribe to this event:
   - `checkout.session.completed`
4. Copy the endpoint's **Signing secret** (`whsec_…`) into the deployment's
   `STRIPE_WEBHOOK_SECRET`.

## 5. Run the migration

```bash
cd backend && python manage.py setup        # runs alembic upgrade head
# or directly:  alembic upgrade head
```

This creates `billing_customers`, `credit_ledger`, and `billing_webhook_events`.
(The app also builds them via `create_all` on boot, so a fresh DB needs no
manual step — the migration is for existing databases.)

## 6. Test it

1. Restart the backend so it picks up the new env.
2. In the app: settings → **Billing** tab → pick a pack → buy.
3. On the Stripe Checkout page use a test card: `4242 4242 4242 4242`, any
   future expiry, any CVC, any ZIP.
4. You're redirected back to `/?billing=success`. Within a second or two
   `stripe listen` (or the dashboard endpoint) delivers
   `checkout.session.completed`, the webhook credits the ledger, and the wallet
   balance updates on the next fetch.

Watch events live in Dashboard → **Developers → Events**, or in the
`stripe listen` terminal.

## Going live

1. Toggle the dashboard to **Live mode** and grab live keys (`sk_live_…`).
2. Re-run `python scripts/provision_stripe.py` against the live account (it
   creates live prices; paste the new `price_…` ids).
3. Add a **live** webhook endpoint (step 4) and use its live signing secret.
4. Swap `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_*` in the
   production environment. Done.
