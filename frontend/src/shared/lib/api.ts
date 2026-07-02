import type {
  ColumnMapping,
  EvalExampleResult,
  GridSearchResult,
  GridSearchRequest,
  OptimizationDatasetResponse,
  OptimizationPayloadResponse,
  OptimizationSubmissionResponse,
  OptimizationStatusResponse,
  PaginatedJobsResponse,
  ProfileDatasetRequest,
  ProfileDatasetResponse,
  QueueStatusResponse,
  RunRequest,
  ServeInfoResponse,
  ServeResponse,
  ValidateCodeResponse,
  ValidateDatasetRequest,
  ValidateDatasetResponse,
} from "@/shared/types/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { I18N_KEY, tI18n } from "@/shared/lib/i18n";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { readNdjsonStream, readServerSentEvents, type ServerSentEvent } from "@/shared/lib/sse";

// Resolve the runtime API base lazily on every call. Capturing it once at
// module load races the injected `window.__SKYNET_ENV__` script: the framework
// chunks execute (and evaluate this module) before that inline `<head>` script
// runs, so a module-scope const freezes the build-time `localhost:8000`
// fallback and every request hits the wrong origin. Reading at call time always
// sees the injected env, since requests fire after hydration.
const apiBase = () => getRuntimeEnv().apiUrl;
const JOB_CACHE_MS = 1000;
const QUEUE_CACHE_MS = 5000;
const SIDEBAR_CACHE_MS = 3000;

const apiUrlAtLoad = apiBase();
if (
  typeof window !== "undefined" &&
  process.env.NODE_ENV === "production" &&
  apiUrlAtLoad.startsWith("http://") &&
  !apiUrlAtLoad.includes("localhost") &&
  !apiUrlAtLoad.includes("127.0.0.1")
) {
  console.error(
    "[Skynet] Production API URL uses HTTP — API keys and tokens will be transmitted in plaintext. " +
      "Set API_URL (or NEXT_PUBLIC_API_URL) to an https:// URL.",
  );
}

const _inflight = new Map<string, Promise<unknown>>();
const _cache = new Map<string, { data: unknown; ts: number }>();
const GET_CACHE_MS = 2000;
// Bumped by invalidateCache. A request that started before an invalidation
// must not write its (possibly-stale) result into _cache or complete a
// post-invalidation dedup — otherwise callers that fetch right after a
// mutation can get pre-mutation data via a still-resolving in-flight promise.
let _cacheGen = 0;
let _authToken: string | undefined;

export function setApiAuthToken(token: string | undefined) {
  _authToken = token;
}

let _authTokenRefresher: (() => Promise<string | undefined>) | undefined;

/**
 * Register an async refresher that mints a fresh backend bearer token (e.g.
 * by re-fetching the NextAuth session). Used to recover from a 401 when the
 * cached token expired while a tab sat idle during a long optimization run.
 */
export function setApiAuthTokenRefresher(
  refresher: (() => Promise<string | undefined>) | undefined,
) {
  _authTokenRefresher = refresher;
}

/**
 * Run a one-shot bearer-token refresh after a 401 and hand back a fresh
 * token to retry with. Returns `undefined` when there is no refresher, the
 * refresh failed, or the refreshed token is unchanged (so the caller should
 * surface the original 401 instead of looping).
 */
async function refreshAuthTokenOn401(): Promise<string | undefined> {
  if (!_authTokenRefresher) return undefined;
  let fresh: string | undefined;
  try {
    fresh = await _authTokenRefresher();
  } catch {
    return undefined;
  }
  if (!fresh || fresh === _authToken) return undefined;
  _authToken = fresh;
  return fresh;
}

/**
 * `fetch` for raw SSE/NDJSON callers that can't go through `request`.
 * Attaches the cached bearer token and, on a 401, transparently refreshes
 * the token and retries once — the in-memory token has a short TTL and goes
 * stale while a long run keeps the tab idle/backgrounded.
 */
export async function fetchWithAuthRetry(url: string, init: RequestInit): Promise<Response> {
  const send = (token: string | undefined) =>
    fetch(url, {
      ...init,
      headers: {
        ...(init.headers as Record<string, string> | undefined),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });
  const res = await send(_authToken);
  if (res.status !== 401) return res;
  const fresh = await refreshAuthTokenOn401();
  if (!fresh) return res;
  return send(fresh);
}

function cachedGet<T>(path: string, maxAge = GET_CACHE_MS): Promise<T> {
  const key = path;
  const startGen = _cacheGen;

  const cached = _cache.get(key);
  if (cached && Date.now() - cached.ts < maxAge) {
    return Promise.resolve(cached.data as T);
  }

  const existing = _inflight.get(key);
  if (existing) return existing as Promise<T>;

  const promise = request<T>(path)
    .then((data) => {
      if (startGen === _cacheGen) _cache.set(key, { data, ts: Date.now() });
      if (_inflight.get(key) === promise) _inflight.delete(key);
      return data;
    })
    .catch((err) => {
      if (_inflight.get(key) === promise) _inflight.delete(key);
      throw err;
    });
  _inflight.set(key, promise);
  return promise;
}

/**
 * Invalidate all cached GET responses whose path includes any of the
 * given substrings. Call after mutations (delete, cancel, rename, pin)
 * so the next fetch hits the server.
 */
export function invalidateCache(...pathSubstrings: string[]) {
  _cacheGen++;
  const matches = (key: string) =>
    pathSubstrings.length === 0 ||
    pathSubstrings.some((s) => key === s || key.startsWith(`${s}/`) || key.startsWith(`${s}?`));
  for (const [key] of _cache) if (matches(key)) _cache.delete(key);
  for (const [key] of _inflight) if (matches(key)) _inflight.delete(key);
}

// Listeners that keep the GET cache honest when mutations fire from
// anywhere in the app (UI buttons, bulk dialogs, or agent MCP tool calls).
// Without these, the 2s GET cache can serve pre-mutation data to sidebar
// and dashboard re-fetches that happen immediately after an event.
if (typeof window !== "undefined") {
  window.addEventListener("optimizations-changed", () =>
    invalidateCache("/optimizations", "/analytics"),
  );
}

/** Backend error code for the unified storage budget being exceeded (HTTP 409). */
export const STORAGE_QUOTA_CODE = I18N_KEY.USER_STORAGE_QUOTA_EXCEEDED;

/** Browser event the central error path fires when a write hits the storage budget. */
export const STORAGE_QUOTA_EVENT = "storage-quota-exceeded";

/** Backend error code for a managed run blocked by an empty credit balance (HTTP 402). */
export const INSUFFICIENT_CREDITS_CODE = I18N_KEY.BILLING_INSUFFICIENT_CREDITS;

/** Browser event the central error path fires when a submit hits the credit gate. */
export const INSUFFICIENT_CREDITS_EVENT = "billing-insufficient-credits";

/** Browser event fired after a storage-freeing delete so the meter re-reads usage. */
export const STORAGE_CHANGED_EVENT = "storage-changed";

/**
 * Error carrying the backend's structured envelope so callers can branch on the
 * machine-readable ``code`` (not just the rendered message). Subclasses ``Error``
 * so existing ``err instanceof Error`` / ``err.message`` handling keeps working.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly params?: Record<string, unknown>;

  constructor(
    message: string,
    opts: { status: number; code?: string; params?: Record<string, unknown> },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status;
    this.code = opts.code;
    this.params = opts.params;
  }
}

/** Narrow a caught value to the storage-budget 409 so its toast can be suppressed. */
export function isStorageQuotaError(err: unknown): err is ApiError {
  return err instanceof ApiError && err.code === STORAGE_QUOTA_CODE;
}

/** Narrow a caught value to the credit-gate 402 so its toast can be suppressed. */
export function isInsufficientCreditsError(err: unknown): err is ApiError {
  return err instanceof ApiError && err.code === INSUFFICIENT_CREDITS_CODE;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const send = (token: string | undefined) =>
    fetch(`${apiBase()}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init?.headers,
      },
    });
  let res: Response;
  try {
    res = await send(_authToken);
    if (res.status === 401) {
      const fresh = await refreshAuthTokenOn401();
      if (fresh) res = await send(fresh);
    }
  } catch (err) {
    throw new Error(msg("auto.shared.lib.api.literal.1"), { cause: err });
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const parsed = parseError(text);
    // The storage budget is account-wide, so any blocked write opens one shared
    // modal regardless of which producer flow tripped it. The 409 still throws
    // so the caller's success path halts; producers suppress their own toast via
    // isStorageQuotaError so the modal is the single surface.
    if (parsed.code === STORAGE_QUOTA_CODE && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent(STORAGE_QUOTA_EVENT, { detail: parsed.params }));
    }
    // The credit gate is account-wide like the storage budget: any blocked submit
    // opens the one paywall modal, and producers suppress their own toast via
    // isInsufficientCreditsError so the modal is the single surface.
    if (parsed.code === INSUFFICIENT_CREDITS_CODE && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent(INSUFFICIENT_CREDITS_EVENT));
    }
    throw new ApiError(
      parsed.message ?? formatMsg("auto.shared.lib.api.template.1", { p1: res.status }),
      { status: res.status, code: parsed.code, params: parsed.params },
    );
  }
  return res.json();
}

/**
 * Parse an error response body into its rendered message and structured fields.
 *
 * Message preference order:
 *   1. ``body.code`` + ``body.params`` — re-rendered client-side via
 *      :func:`tI18n` so UI copy comes from the local catalog.
 *   2. ``body.detail`` — the rendered Hebrew string from the server.
 *   3. ``body.error`` — legacy envelope fallback.
 *
 * ``message`` is ``undefined`` when the body is not JSON (e.g. an HTML error
 * page) so the caller can fall back to a status-code template; ``code`` /
 * ``params`` are passed through for callers that branch on the error code.
 */
function parseError(text: string): {
  message?: string;
  code?: string;
  params?: Record<string, unknown>;
} {
  try {
    const body = JSON.parse(text) as {
      code?: string;
      params?: Record<string, unknown>;
      detail?: unknown;
      error?: unknown;
    };
    const code = typeof body.code === "string" ? body.code : undefined;
    if (code) {
      const translated = tI18n(code, body.params);
      if (translated !== code) return { message: translated, code, params: body.params };
    }
    const raw = body.detail ?? body.error;
    if (typeof raw === "string") return { message: raw, code, params: body.params };
    if (raw != null) return { message: JSON.stringify(raw), code, params: body.params };
    return { code, params: body.params };
  } catch {
    /* response was not JSON (e.g. HTML error page) */
  }
  return {};
}

/** Rendered message from an error body, or ``undefined`` for non-JSON bodies. */
function parseErrorMessage(text: string): string | undefined {
  return parseError(text).message;
}

/**
 * Fire-and-forget sender for the telemetry SDK. Best-effort by contract: it
 * never throws, never blocks, and never surfaces an error — telemetry must not
 * affect the product. `keepalive` lets a flush survive the page navigation that
 * often triggers it. The bearer token is attached when present so events are
 * attributed to the signed-in user; an anonymous batch (no token) is still
 * accepted by the public ingest route. Falls back to a headerless `sendBeacon`
 * if a `keepalive` fetch can't be issued (e.g. payload over the ~64KB cap).
 */
export function postTelemetry(body: unknown): void {
  if (typeof window === "undefined") return;
  const url = `${apiBase()}/telemetry/events`;
  const payload = JSON.stringify(body);
  try {
    void fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(_authToken ? { Authorization: `Bearer ${_authToken}` } : {}),
      },
      body: payload,
      keepalive: true,
    }).catch(() => {
      /* lossy by design — swallow network failures */
    });
  } catch {
    try {
      navigator.sendBeacon?.(url, new Blob([payload], { type: "application/json" }));
    } catch {
      /* give up: telemetry never raises to the caller */
    }
  }
}

export function submitRun(payload: RunRequest) {
  return request<OptimizationSubmissionResponse>("/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function submitGridSearch(payload: GridSearchRequest) {
  return request<OptimizationSubmissionResponse>("/grid-search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listJobs(params?: {
  status?: string;
  username?: string;
  optimization_type?: string;
  limit?: number;
  offset?: number;
  include_shared?: boolean;
}) {
  const q = new URLSearchParams();
  if (params?.status) q.set("status", params.status);
  if (params?.username) q.set("username", params.username);
  if (params?.optimization_type) q.set("optimization_type", params.optimization_type);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  if (params?.include_shared) q.set("include_shared", "true");
  const qs = q.toString();
  return cachedGet<PaginatedJobsResponse>(`/optimizations${qs ? `?${qs}` : ""}`);
}

export interface OptimizationCounts {
  total: number;
  pending: number;
  validating: number;
  running: number;
  success: number;
  failed: number;
  cancelled: number;
  /** Runs shared with the caller (set only when include_shared is requested). */
  shared?: number;
}

export function getOptimizationCounts(username?: string, includeShared?: boolean) {
  const q = new URLSearchParams();
  if (username) q.set("username", username);
  if (includeShared) q.set("include_shared", "true");
  const qs = q.toString();
  return cachedGet<OptimizationCounts>(`/optimizations/counts${qs ? `?${qs}` : ""}`);
}

export interface StorageQuotaOverride {
  username: string;
  /** Per-user byte ceiling that replaces the default; null when no override. */
  quota_bytes: number | null;
  updated_at?: string | null;
  updated_by?: string | null;
  /** Effective budget after override resolution (override or default). */
  effective_bytes: number;
  /** The user's current storage footprint in bytes. */
  used_bytes: number;
}

export interface StorageQuotaOverridesResponse {
  default_bytes: number;
  overrides: StorageQuotaOverride[];
}

export function getStorageQuotaOverrides() {
  return request<StorageQuotaOverridesResponse>("/admin/storage-quotas");
}

export function setStorageQuotaOverride(username: string, quotaBytes: number) {
  return request<StorageQuotaOverride>("/admin/storage-quotas", {
    method: "PUT",
    body: JSON.stringify({ username, quota_bytes: quotaBytes }),
  });
}

export function deleteStorageQuotaOverride(username: string) {
  return request<StorageQuotaOverride>(`/admin/storage-quotas/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
}

export interface ApiTokenInfo {
  last4: string;
  created_at: string;
  last_used_at: string | null;
}

export interface ApiTokenCreated {
  token: string;
  last4: string;
  created_at: string;
}

/** Fetch metadata for the caller's active API token, or ``null`` if none exists. */
export function getApiToken() {
  return request<ApiTokenInfo | null>("/settings/api-token");
}

/** Generate (or rotate) the caller's API token; the plaintext is returned once. */
export function generateApiToken() {
  return request<ApiTokenCreated>("/settings/api-token", { method: "POST" });
}

/** Revoke the caller's active API token. Idempotent; the route returns 204. */
export async function revokeApiToken(): Promise<void> {
  const res = await fetchWithAuthRetry(`${apiBase()}/settings/api-token`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      parseErrorMessage(text) ?? formatMsg("auto.shared.lib.api.template.1", { p1: res.status }),
    );
  }
}

export interface BillingFreeGrant {
  credits_remaining: number;
  credits_total: number;
  /** ISO-8601 instant a Premium allotment renews; `null` for the one-time free grant. */
  resets_at: string | null;
}

export interface BillingUsageEntry {
  id: string;
  at: string;
  label: string;
  model: string | null;
  credits: number;
  kind: string;
}

/** The caller's wallet as the backend reports it (snake_case mirrors the API). */
export interface BillingWalletResponse {
  paid_balance_credits: number;
  free_grant: BillingFreeGrant;
  premium_active: boolean;
  subscription_status: string | null;
  subscription_current_period_end: string | null;
  usage: BillingUsageEntry[];
}

/** Fetch the caller's credit wallet + subscription state. Reads work even without Stripe. */
export function getWallet() {
  return request<BillingWalletResponse>("/billing/wallet");
}

/** One day's run spend, billed vs refunded (the usage dashboard's time series). */
export interface BillingUsageDay {
  date: string;
  billed_credits: number;
  refunded_credits: number;
}

/** One model's share of run spend over the window. */
export interface BillingUsageModel {
  model: string | null;
  credits: number;
  runs: number;
}

/** A date-ranged usage rollup for the Usage dashboard (snake_case mirrors the API). */
export interface BillingUsageResponse {
  start: string;
  end: string;
  billed_credits: number;
  refunded_credits: number;
  runs: number;
  by_day: BillingUsageDay[];
  by_model: BillingUsageModel[];
  entries: BillingUsageEntry[];
}

/**
 * Fetch a date-ranged usage rollup (totals + per-day + per-model + recent rows).
 * `start`/`end` are ISO-8601; omit both for the backend's default 30-day window.
 */
export function getUsage(start?: string, end?: string) {
  const params = new URLSearchParams();
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  const qs = params.toString();
  return request<BillingUsageResponse>(`/billing/usage${qs ? `?${qs}` : ""}`);
}

/** Start a Stripe Checkout session for a credit pack; redirect the browser to `.url`. */
export function createCheckoutSession(packId: string) {
  return request<{ url: string }>("/billing/checkout", {
    method: "POST",
    body: JSON.stringify({ pack_id: packId }),
  });
}

/** Start a Stripe Checkout session for the Premium subscription; redirect to `.url`. */
export function createSubscriptionCheckout() {
  return request<{ url: string }>("/billing/subscribe", { method: "POST" });
}

/** The Founder's Rate availability: the deadline gate and the 12-month price-lock window. */
export interface FoundersRateResponse {
  open: boolean;
  closes_at: string;
  price_locked_until: string;
}

/** Read the Founder's Rate availability (open/closed + lock window). Works without Stripe. */
export function getFoundersRate() {
  return request<FoundersRateResponse>("/billing/founders");
}

/** Start a Stripe Checkout session for the Founder's Rate subscription; redirect to `.url`. */
export function createFoundersCheckout() {
  return request<{ url: string }>("/billing/founders/subscribe", { method: "POST" });
}

/** Open the Stripe Billing Portal (manage card / invoices / cancel); redirect to `.url`. */
export function openBillingPortal() {
  return request<{ url: string }>("/billing/portal", { method: "POST" });
}

/** One stored BYOK provider connection as the backend reports it — masked, never the secret. */
export interface ProviderKeyResponse {
  id: string;
  provider: string;
  label?: string | null;
  last4: string;
  api_base?: string | null;
  status: "verified" | "unverified" | "invalid";
  added_at: string;
}

/** Optional connection metadata sent alongside a saved key. */
export interface SaveProviderKeyOptions {
  label?: string | null;
  apiBase?: string | null;
  params?: Record<string, unknown>;
}

/** The caller's stored BYOK provider keys, masked. */
export interface ProviderKeysResponse {
  keys: ProviderKeyResponse[];
}

/** List the caller's stored BYOK provider keys (masked). Reads work without the vault key. */
export function getProviderKeys() {
  return request<ProviderKeysResponse>("/billing/byok/keys");
}

/**
 * Save (or rotate) a BYOK provider connection. The secret is encrypted at rest
 * on the backend and verified on entry — against `apiBase` when given, so a
 * custom endpoint is checked too. The response carries only the masked tail and
 * the entry-time verify verdict; the plaintext is never echoed back.
 */
export function saveProviderKey(provider: string, secret: string, opts?: SaveProviderKeyOptions) {
  return request<ProviderKeyResponse>("/billing/byok/keys", {
    method: "PUT",
    body: JSON.stringify({
      provider,
      secret,
      label: opts?.label ?? null,
      api_base: opts?.apiBase ?? null,
      params: opts?.params ?? {},
    }),
  });
}

/** Re-run the verify probe against a stored BYOK key and return the fresh verdict. */
export function verifyProviderKey(provider: string) {
  return request<ProviderKeyResponse>(`/billing/byok/keys/${provider}/verify`, { method: "POST" });
}

/** Forget a stored BYOK provider key; returns the remaining masked keys. */
export function removeProviderKey(provider: string) {
  return request<ProviderKeysResponse>(`/billing/byok/keys/${provider}`, { method: "DELETE" });
}

export interface DirectoryUserMatch {
  username: string;
  display_name?: string | null;
  email?: string | null;
  source: "db" | "directory";
}

export interface DirectoryUserSearchResponse {
  matches: DirectoryUserMatch[];
}

export function searchAdminUsers(query: string, limit = 10) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return request<DirectoryUserSearchResponse>(`/admin/users/search?${params.toString()}`);
}

export interface DashboardAnalyticsJob {
  optimization_id: string;
  name?: string | null;
  optimizer_name?: string | null;
  model_name?: string | null;
  status: string;
  baseline_test_metric?: number | null;
  optimized_test_metric?: number | null;
  metric_improvement?: number | null;
  elapsed_seconds?: number | null;
  dataset_rows?: number | null;
  optimization_type?: string | null;
  best_pair_label?: string | null;
  created_at?: string | null;
}

export interface DashboardAnalytics {
  filtered_total: number;
  status_counts: Record<string, number>;
  optimizer_counts: Record<string, number>;
  job_type_counts: Record<string, number>;
  model_usage: Array<{ name: string; value: number }>;
  owner_usage: Array<{ name: string; value: number }>;
  access_usage: Array<{ name: string; value: number }>;
  success_count: number;
  failed_count: number;
  running_count: number;
  terminal_count: number;
  success_rate: number;
  avg_improvement: number | null;
  avg_runtime_seconds: number | null;
  total_dataset_rows: number;
  total_pairs_run: number;
  grid_search_count: number;
  single_run_count: number;
  best_improvement: number | null;
  improvement_by_optimizer: Array<{ name: string; average: number; count: number }>;
  runtime_minutes_by_optimizer: Array<{ name: string; average: number; count: number }>;
  top_improvement: DashboardAnalyticsJob[];
  runtime_distribution: DashboardAnalyticsJob[];
  dataset_vs_improvement: DashboardAnalyticsJob[];
  efficiency: DashboardAnalyticsJob[];
  top_jobs_by_improvement: DashboardAnalyticsJob[];
  timeline: Array<{ date: string; count: number }>;
  available_optimizers: string[];
  available_models: string[];
}

export function getDashboardAnalytics(params?: {
  username?: string;
  optimizer?: string;
  model?: string;
  status?: string;
  optimization_id?: string;
  date?: string;
  include_shared?: boolean;
  owner?: string;
  access?: string;
}) {
  const q = new URLSearchParams();
  if (params?.username) q.set("username", params.username);
  if (params?.optimizer) q.set("optimizer", params.optimizer);
  if (params?.model) q.set("model", params.model);
  if (params?.status) q.set("status", params.status);
  if (params?.optimization_id) q.set("optimization_id", params.optimization_id);
  if (params?.date) q.set("date", params.date);
  if (params?.include_shared) q.set("include_shared", "true");
  if (params?.owner) q.set("owner", params.owner);
  if (params?.access) q.set("access", params.access);
  const qs = q.toString();
  return cachedGet<DashboardAnalytics>(`/analytics/dashboard${qs ? `?${qs}` : ""}`);
}

export function getJob(
  optimizationId: string,
  cursor?: { sinceProgress: number; sinceLog: number },
) {
  // A delta fetch asks only for stream rows past what the caller already
  // holds. The response is a tail slice (stateful w.r.t. the caller's buffer),
  // so it bypasses the value cache — replaying a cached tail against a
  // different local buffer would corrupt the splice. The full (no-cursor) path
  // stays cached so the detail gate's probe warms the view's first load.
  if (cursor) {
    const q = new URLSearchParams({
      since_progress: String(cursor.sinceProgress),
      since_log: String(cursor.sinceLog),
    });
    return request<OptimizationStatusResponse>(`/optimizations/${optimizationId}?${q.toString()}`);
  }
  return cachedGet<OptimizationStatusResponse>(`/optimizations/${optimizationId}`, JOB_CACHE_MS);
}

export function getOptimizationPayload(optimizationId: string) {
  return request<OptimizationPayloadResponse>(`/optimizations/${optimizationId}/payload`);
}

export function getOptimizationDataset(optimizationId: string) {
  return request<OptimizationDatasetResponse>(`/optimizations/${optimizationId}/dataset`);
}

export function getTestResults(optimizationId: string) {
  return request<{
    baseline: EvalExampleResult[];
    optimized: EvalExampleResult[];
  }>(`/optimizations/${optimizationId}/test-results`);
}

/**
 * Download the self-contained, runnable DSPy program export as a zip and hand
 * it to the browser. Unlike the other `request`-based calls this returns binary
 * (a `StreamingResponse` attachment), so it goes through `fetchWithAuthRetry`
 * and reads the body as a Blob rather than JSON.
 */
export async function downloadProgramExport(optimizationId: string): Promise<void> {
  const res = await fetchWithAuthRetry(`${apiBase()}/optimizations/${optimizationId}/program-export`, {
    method: "GET",
  });
  if (!res.ok) {
    const parsed = parseError(await res.text().catch(() => ""));
    throw new ApiError(
      parsed.message ?? formatMsg("auto.shared.lib.api.template.1", { p1: res.status }),
      { status: res.status, code: parsed.code, params: parsed.params },
    );
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `dspy_program_${optimizationId.slice(0, 8)}.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Effective/resolved tier on a shared optimization. `owner` is the creator's
 * (and admins') resolved role; it is never a *grantable* member tier — see
 * {@link MemberRole}. Reassigned only via {@link transferOwnership}.
 */
export type ShareRole = "viewer" | "editor" | "owner";

/**
 * Grantable member tier. Single-owner model: ownership belongs to the creator
 * and moves only by transfer, so an invited member is always viewer or editor.
 */
export type MemberRole = Exclude<ShareRole, "owner">;

/** General-access policy on the active share link. */
export type GeneralAccess = "restricted" | "anyone";

/**
 * Tier an "anyone with the link" link grants a *signed-in* visitor. Tops out at
 * editor (ownership is never transferred by link, mirroring Google Drive).
 * Access is login-gated, so the link never grants a logged-out visitor.
 */
export type LinkRole = Exclude<ShareRole, "owner">;

/** One invited member of an optimization (username + grantable tier role). */
export interface SharingMember {
  username: string;
  role: MemberRole;
}

/** Owner/editor-facing sharing config for one optimization. */
export interface SharingState {
  general_access: GeneralAccess;
  general_role: LinkRole;
  token: string | null;
  share_path: string | null;
  owner: string | null;
  members: SharingMember[];
  /**
   * Explore-corpus visibility — orthogonal to `general_access`. `true` hides the
   * job from the public /explore search; `general_access` governs link access.
   */
  is_private: boolean;
}

/** Envelope for ``GET /users/search`` — matching distinct usernames. */
export interface UserSearchResponse {
  usernames: string[];
}

/**
 * Composite, access-gated read behind a ``/share/<token>`` page.
 *
 * `role` is the caller's effective access (`viewer`/`editor`/`owner`); the read
 * is login-gated, so the floor is `viewer` (real owner shown). Cloning is
 * viewer+; serving/chat is editor+ (it spends the owner's key). `serve_info`
 * (the field schema behind the inference panel) is only populated for editor+;
 * it is `null` for viewers.
 */
export interface SharedOptimizationData {
  optimization_id: string;
  role: ShareRole;
  owner: string | null;
  status: OptimizationStatusResponse;
  payload: Record<string, unknown>;
  dataset: OptimizationDatasetResponse | null;
  test_results: { baseline: EvalExampleResult[]; optimized: EvalExampleResult[] } | null;
  serve_info: ServeInfoResponse | null;
}

/** Fetch the current sharing config (general access + members) for an optimization. */
export function getSharing(optimizationId: string) {
  return request<SharingState>(`/optimizations/${optimizationId}/sharing`);
}

/**
 * Set the link policy: `general_access` (restricted vs anyone-with-link) and,
 * optionally, `general_role` — the tier an anyone-link grants signed-in
 * visitors (viewer/editor). Mints a link if needed.
 */
export function putSharing(
  optimizationId: string,
  body: { general_access: GeneralAccess; general_role?: LinkRole },
) {
  return request<SharingState>(`/optimizations/${optimizationId}/sharing`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

/**
 * Flip an optimization's public-Explore visibility (owner-only). `isPrivate:
 * true` hides it from the public corpus; `false` lists it. Independent of the
 * share link's general access. Returns the refreshed sharing state.
 */
export function setOptimizationVisibility(optimizationId: string, isPrivate: boolean) {
  return request<SharingState>(`/optimizations/${optimizationId}/visibility`, {
    method: "PUT",
    body: JSON.stringify({ is_private: isPrivate }),
  });
}

/** Invite a user (add or replace a member grant). */
export function addShareMember(
  optimizationId: string,
  body: { username: string; role: MemberRole },
) {
  return request<SharingState>(`/optimizations/${optimizationId}/sharing/members`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Change an existing member's tier role. */
export function updateShareMember(
  optimizationId: string,
  username: string,
  body: { role: MemberRole },
) {
  return request<SharingState>(
    `/optimizations/${optimizationId}/sharing/members/${encodeURIComponent(username)}`,
    { method: "PATCH", body: JSON.stringify(body) },
  );
}

/** Remove a member's grant from the optimization. */
export function removeShareMember(optimizationId: string, username: string) {
  return request<SharingState>(
    `/optimizations/${optimizationId}/sharing/members/${encodeURIComponent(username)}`,
    { method: "DELETE" },
  );
}

/**
 * Transfer ownership to an existing member (owner-only). Single-owner model:
 * the new owner must already be a member, the previous owner is demoted to an
 * editor, and the new owner's member grant is dropped. The serving key stays
 * with the optimization, so this moves control, not billing. Returns the
 * refreshed sharing state (`owner` is now the new owner).
 */
export function transferOwnership(optimizationId: string, username: string) {
  return request<SharingState>(`/optimizations/${optimizationId}/sharing/transfer`, {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

/** Autocomplete distinct known usernames by prefix (any authed caller). */
export function searchUsers(q: string) {
  return request<UserSearchResponse>(`/users/search?q=${encodeURIComponent(q)}`);
}

// ── Personal dataset library ────────────────────────────────────────────────

/** Per-column roles/kinds/order saved with a dataset so the wizard pre-fills. */
export interface DatasetColumnSchema {
  column_order?: string[];
  column_roles?: Record<string, "input" | "output" | "ignore">;
  column_kinds?: Record<string, "text" | "image">;
}

/** One library dataset's metadata. `role` distinguishes owned from shared-in. */
export interface DatasetSummary {
  id: string;
  name: string;
  source: string;
  row_count: number;
  column_count: number;
  byte_size: number;
  content_hash: string;
  owner_username: string;
  role: ShareRole;
  created_at: string;
  updated_at: string;
}

/** Aggregate library storage used by the caller against their quota. */
export interface DatasetUsageMeter {
  used_bytes: number;
  quota_bytes: number;
}

/** Envelope for ``GET /datasets/library`` — the caller's entries and usage. */
export interface DatasetListResponse {
  datasets: DatasetSummary[];
  usage: DatasetUsageMeter;
}

/** Envelope for ``GET /datasets/library/{id}/rows`` — columns, rows, saved schema. */
export interface DatasetRowsResponse {
  id: string;
  columns: string[];
  rows: Array<Record<string, unknown>>;
  row_count: number;
  column_schema: DatasetColumnSchema;
}

/** Envelope for a save/clone — the entry plus whether the bytes deduped. */
export interface SaveDatasetResponse {
  dataset: DatasetSummary;
  deduplicated: boolean;
}

/** One optimization that was submitted from a library dataset (reverse link). */
export interface DatasetOptimizationRef {
  optimization_id: string;
  name?: string | null;
  status?: string | null;
  optimization_type?: string | null;
  username?: string | null;
  created_at?: string | null;
}

/** Envelope for ``GET /datasets/library/{id}/optimizations`` — the reverse link. */
export interface DatasetOptimizationsResponse {
  optimizations: DatasetOptimizationRef[];
}

/** One invited member of a dataset (username + tier role). */
export interface DatasetSharingMember {
  username: string;
  role: MemberRole;
}

/**
 * Owner-facing sharing config for one library dataset. Mirrors {@link SharingState}
 * minus the optimization-only Explore visibility (`is_private`).
 */
export interface DatasetSharingState {
  general_access: GeneralAccess;
  general_role: LinkRole;
  token: string | null;
  share_path: string | null;
  owner: string | null;
  members: DatasetSharingMember[];
}

/** List the caller's saved datasets plus those shared with them, with usage. */
export function listDatasets() {
  return request<DatasetListResponse>("/datasets/library");
}

/** Fetch one saved dataset's rows and saved column schema (viewer+). */
export function getDatasetRows(datasetId: string) {
  return request<DatasetRowsResponse>(`/datasets/library/${datasetId}/rows`);
}

/** Save rows as a new library entry the caller owns. */
export function saveDataset(body: {
  name: string;
  source?: string;
  dataset: Array<Record<string, unknown>>;
  column_schema?: DatasetColumnSchema;
}) {
  return request<SaveDatasetResponse>("/datasets/library", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Clone a dataset shared with the caller into their own library (viewer+). */
export function cloneDataset(datasetId: string) {
  return request<SaveDatasetResponse>(`/datasets/library/${datasetId}/clone`, {
    method: "POST",
  });
}

/**
 * Sidebar/list projection of a saved text-labeling (tagger) session — the
 * lightweight row, without the heavy dataset/annotation payload.
 */
export interface TaggerSessionSummary {
  id: string;
  name: string;
  phase: string;
  row_count: number;
  tagged_count: number;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

/** Full tagger session — everything needed to rehydrate the annotator. */
export interface TaggerSessionDetail extends TaggerSessionSummary {
  config: Record<string, unknown>;
  columns: string[];
  data: Array<Record<string, unknown>>;
  annotations: Record<string, unknown>;
  current_index: number;
}

/** List the caller's saved tagger sessions (pinned first, then newest). */
export function listTaggerSessions(params?: { limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return cachedGet<{ items: TaggerSessionSummary[]; total: number }>(
    `/tagging-sessions${qs ? `?${qs}` : ""}`,
    SIDEBAR_CACHE_MS,
  );
}

/** Fetch one saved session's full state to resume annotating. */
export function getTaggerSession(sessionId: string) {
  return request<TaggerSessionDetail>(`/tagging-sessions/${sessionId}`);
}

// Same-tab handoff from the setup wizard to the /tagger/[id] gate. The wizard
// creates a session then navigates to its URL; stashing the freshest local
// state here lets the gate resume instantly without a refetch (and without
// racing the autosave). A genuine reload finds an empty map and falls back to
// getTaggerSession.
const taggerHandoff = new Map<string, TaggerSessionDetail>();

/** Stash a just-created session so the gate can resume it after navigation. */
export function stashTaggerSession(detail: TaggerSessionDetail) {
  taggerHandoff.set(detail.id, detail);
}

/** Consume a stashed session (one-shot); null when nothing was handed off. */
export function takeTaggerSession(sessionId: string): TaggerSessionDetail | null {
  const detail = taggerHandoff.get(sessionId) ?? null;
  taggerHandoff.delete(sessionId);
  return detail;
}

/** Persist a new session (uploads the dataset once); returns it with its new id. */
export async function createTaggerSession(body: {
  name: string;
  phase?: string;
  config: Record<string, unknown>;
  columns: string[];
  data: Array<Record<string, unknown>>;
  annotations?: Record<string, unknown>;
  current_index?: number;
}) {
  const res = await request<TaggerSessionDetail>("/tagging-sessions", {
    method: "POST",
    body: JSON.stringify(body),
  });
  invalidateCache("/tagging-sessions");
  return res;
}

/**
 * Autosave annotation progress (annotations + cursor + phase) without
 * re-shipping the dataset. The 3s sidebar cache is intentionally left intact —
 * the heavy list invalidation only fires for create / rename / pin / delete.
 */
export function updateTaggerSession(
  sessionId: string,
  body: { annotations: Record<string, unknown>; current_index: number; phase?: string },
) {
  return request<TaggerSessionSummary>(`/tagging-sessions/${sessionId}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

/** Rename a saved session. */
export async function renameTaggerSession(sessionId: string, name: string) {
  const res = await request<TaggerSessionSummary>(`/tagging-sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
  invalidateCache("/tagging-sessions");
  return res;
}

/** Pin or unpin a saved session (``pinned`` is the desired next state). */
export async function setTaggerSessionPinned(sessionId: string, pinned: boolean) {
  const res = await request<TaggerSessionSummary>(`/tagging-sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ pinned }),
  });
  invalidateCache("/tagging-sessions");
  return res;
}

/** Delete a saved session. */
export async function deleteTaggerSession(sessionId: string) {
  const res = await request<{ id: string; deleted: boolean }>(
    `/tagging-sessions/${sessionId}`,
    { method: "DELETE" },
  );
  invalidateCache("/tagging-sessions");
  return res;
}

/**
 * The caller's account-wide storage usage against their budget. ``breakdown``
 * maps each storage category to its byte contribution; ``used_bytes`` is their
 * sum and the same total the save/run gate enforces.
 */
export interface StorageUsageResponse {
  used_bytes: number;
  quota_bytes: number;
  breakdown: Record<string, number>;
}

/** Fetch the caller's unified storage usage backing the meter and quota modal. */
export function getStorageUsage() {
  return request<StorageUsageResponse>("/usage/storage");
}

/**
 * One owned object ranked by its individual storage footprint. ``bytes`` is the
 * same per-row figure the meter attributes to it; ``id`` is the object's key so
 * the cleanup list can deep-link to it (optimizations and datasets) and delete
 * it (all types).
 */
export interface StorageItem {
  id: string;
  type: "optimization" | "dataset" | "chat" | "staged_upload";
  name: string;
  bytes: number;
}

/** Envelope for the cleanup item lists (biggest items and per-category drawers). */
export interface StorageItemsResponse {
  items: StorageItem[];
}

/**
 * List every deletable item in one storage category for that category's cleanup
 * drawer. ``category`` must be a deletable category (optimizations, datasets,
 * agent_chats, staged_uploads); byproduct categories 404.
 */
export function getStorageCategoryItems(category: string) {
  return request<StorageItemsResponse>(`/usage/storage/categories/${encodeURIComponent(category)}`);
}

/** Delete one of the caller's pending (staged) uploads to free its bytes. */
export function deleteStagedUpload(stagedId: string) {
  return request<{ deleted: boolean }>(`/usage/storage/staged/${encodeURIComponent(stagedId)}`, {
    method: "DELETE",
  });
}

/** Rename a saved dataset (owner only). */
export function renameDataset(datasetId: string, name: string) {
  return request<DatasetSummary>(`/datasets/library/${datasetId}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

/** Delete a saved dataset and its bytes (owner only). */
export function deleteDataset(datasetId: string) {
  return request<{ deleted: boolean }>(`/datasets/library/${datasetId}`, {
    method: "DELETE",
  });
}

/** List the runs the caller can see that were submitted from a dataset. */
export function listDatasetOptimizations(datasetId: string) {
  return request<DatasetOptimizationsResponse>(`/datasets/library/${datasetId}/optimizations`);
}

/** Fetch the current sharing config (general access + members) for a dataset. */
export function getDatasetSharing(datasetId: string) {
  return request<DatasetSharingState>(`/datasets/library/${datasetId}/sharing`);
}

/** Set the dataset link policy (general access + optional anyone-link role). */
export function putDatasetSharing(
  datasetId: string,
  body: { general_access: GeneralAccess; general_role?: LinkRole },
) {
  return request<DatasetSharingState>(`/datasets/library/${datasetId}/sharing`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

/** Invite a user to a dataset (add or replace a member grant). */
export function addDatasetShareMember(
  datasetId: string,
  body: { username: string; role: MemberRole },
) {
  return request<DatasetSharingState>(`/datasets/library/${datasetId}/sharing/members`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Change an existing dataset member's tier role. */
export function updateDatasetShareMember(
  datasetId: string,
  username: string,
  body: { role: MemberRole },
) {
  return request<DatasetSharingState>(
    `/datasets/library/${datasetId}/sharing/members/${encodeURIComponent(username)}`,
    { method: "PATCH", body: JSON.stringify(body) },
  );
}

/** Remove a member's grant from a dataset. */
export function removeDatasetShareMember(datasetId: string, username: string) {
  return request<DatasetSharingState>(
    `/datasets/library/${datasetId}/sharing/members/${encodeURIComponent(username)}`,
    { method: "DELETE" },
  );
}

/** Transfer dataset ownership to an existing member (owner-only). */
export function transferDatasetOwnership(datasetId: string, username: string) {
  return request<DatasetSharingState>(`/datasets/library/${datasetId}/sharing/transfer`, {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

/** Result of redeeming a dataset share link — the target id and granted tier. */
export interface ClaimDatasetResult {
  dataset_id: string;
  role: ShareRole;
}

/** Redeem an ``anyone`` dataset link, durably granting its tier to the caller. */
export function claimSharedDataset(token: string) {
  return request<ClaimDatasetResult>(`/datasets/share/${encodeURIComponent(token)}/claim`, {
    method: "POST",
  });
}

/** Public read — no auth token required; the token in the path is the capability. */
export function getSharedOptimization(token: string) {
  return request<SharedOptimizationData>(`/share/${encodeURIComponent(token)}`);
}

/**
 * Scrubbed read-only composite for a PUBLIC (Explore-corpus) optimization, by id.
 * Returns the same shape as a shared read, so the detail view can render it
 * in read-only mode. 404s when the optimization is unknown or private — backs the
 * Explore "public" tab so a listed run is also openable.
 */
export function getPublicOptimization(optimizationId: string) {
  return request<SharedOptimizationData>(
    `/optimizations/${encodeURIComponent(optimizationId)}/public`,
  );
}

/** Resolved target of a redeemed share link: the optimization and the caller's role. */
export interface ClaimShareResult {
  optimization_id: string;
  role: ShareRole;
}

/**
 * Redeem a share link (Google-Drive semantics): an anyone-with-link URL durably
 * grants the signed-in caller the link's tier, so the run then lives in their
 * account and the normal `/optimizations/{id}` routes resolve them to that role.
 * Returns where to send them. Requires the bearer (the app is login-gated).
 */
export function claimSharedOptimization(token: string) {
  return request<ClaimShareResult>(`/share/${encodeURIComponent(token)}/claim`, { method: "POST" });
}

/**
 * Run one inference through the owner's stored model on a shared optimization.
 * Requires an effective role of editor or higher (it spends the owner's key);
 * viewers and the anonymous `view` role get 403.
 */
export function serveSharedOptimization(token: string, inputs: Record<string, string>) {
  return request<ServeResponse>(`/share/${encodeURIComponent(token)}/serve`, {
    method: "POST",
    body: JSON.stringify({ inputs }),
  });
}

export async function cancelJob(optimizationId: string) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/cancel`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}

// Pause suspends a running run at its checkpoint (status → "paused"), keeping it
// resumable. Like cancel it acts on the SAME run in place, so callers refresh the
// current view rather than navigating.
export async function pauseJob(optimizationId: string) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/pause`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}

// Re-runs a failed/cancelled optimization. The response's optimization_id is the
// NEW run, so callers navigate to it after success.
export async function retryJob(optimizationId: string) {
  const res = await request<OptimizationSubmissionResponse>(
    `/optimizations/${optimizationId}/retry`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}
// Restart re-runs the SAME run from scratch in place: status flips to pending,
// the prior attempt's logs/progress/results are cleared, and the id is unchanged.
// Like resumeJob (and unlike retryJob) callers refresh the current view rather
// than navigating to a new run.
export async function restartJob(optimizationId: string) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/restart`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}
// Resume continues the SAME run from its checkpoint (no new id), so callers
// refresh the current view rather than navigating, unlike retryJob.
export async function resumeJob(optimizationId: string) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/resume`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}

export async function deleteJob(optimizationId: string) {
  const res = await request<{ optimization_id: string; deleted: boolean }>(
    `/optimizations/${optimizationId}`,
    { method: "DELETE" },
  );
  invalidateCache("/optimizations");
  return res;
}

export async function deleteGridPair(optimizationId: string, pairIndex: number) {
  const res = await request<GridSearchResult>(
    `/optimizations/${optimizationId}/pair/${pairIndex}`,
    { method: "DELETE" },
  );
  invalidateCache("/optimizations");
  return res;
}
// Per-pair re-run: like a single run's Restart/Resume scoped to one grid pair —
// re-queues the grid in place to re-run only this pair, keeping the others.
export async function restartGridPair(optimizationId: string, pairIndex: number) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/pair/${pairIndex}/restart`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}
export async function resumeGridPair(optimizationId: string, pairIndex: number) {
  const res = await request<{ optimization_id: string; status: string }>(
    `/optimizations/${optimizationId}/pair/${pairIndex}/resume`,
    { method: "POST" },
  );
  invalidateCache("/optimizations");
  return res;
}

export async function bulkDeleteJobs(optimizationIds: string[]) {
  const res = await request<{
    deleted: string[];
    skipped: Array<{ optimization_id: string; reason: string }>;
  }>("/optimizations/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ optimization_ids: optimizationIds }),
  });
  invalidateCache("/optimizations");
  return res;
}

/** Outcome of an id-keyed bulk delete: the ids removed and the ones skipped (with a reason). */
export interface BulkDeleteResult {
  deleted: string[];
  skipped: Array<{ id: string; reason: string }>;
}

/** Bulk-delete saved library datasets (owner only). */
export async function bulkDeleteDatasets(ids: string[]): Promise<BulkDeleteResult> {
  const res = await request<BulkDeleteResult>("/datasets/library/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
  invalidateCache("/datasets/library", "/usage/storage");
  return res;
}

/** Bulk-delete the caller's pending (staged) uploads. */
export async function bulkDeleteStagedUploads(ids: string[]): Promise<BulkDeleteResult> {
  const res = await request<BulkDeleteResult>("/usage/storage/staged/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
  invalidateCache("/usage/storage");
  return res;
}

/** Bulk-delete the caller's saved agent conversations. */
export async function bulkDeleteConversations(ids: string[]): Promise<BulkDeleteResult> {
  const res = await request<BulkDeleteResult>("/agent/conversations/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
  invalidateCache("/agent/conversations", "/usage/storage");
  return res;
}

/**
 * Route a batch of storage items of one ``type`` to its bulk-delete endpoint,
 * normalizing the optimizations response onto the shared ``{id, reason}`` shape so
 * the cleanup drawer can treat every category uniformly.
 */
export async function bulkDeleteStorageItems(
  type: StorageItem["type"],
  ids: string[],
): Promise<BulkDeleteResult> {
  if (type === "optimization") {
    const res = await bulkDeleteJobs(ids);
    invalidateCache("/usage/storage");
    return {
      deleted: res.deleted,
      skipped: res.skipped.map((s) => ({ id: s.optimization_id, reason: s.reason })),
    };
  }
  if (type === "dataset") return bulkDeleteDatasets(ids);
  if (type === "staged_upload") return bulkDeleteStagedUploads(ids);
  return bulkDeleteConversations(ids);
}

export async function renameOptimization(optimizationId: string, name: string) {
  const res = await request<{ optimization_id: string; name: string }>(
    `/optimizations/${optimizationId}/name`,
    {
      method: "PATCH",
      body: JSON.stringify({ name }),
    },
  );
  invalidateCache("/optimizations");
  return res;
}

export async function togglePinOptimization(optimizationId: string) {
  const res = await request<{ optimization_id: string; pinned: boolean }>(
    `/optimizations/${optimizationId}/pin`,
    { method: "PATCH" },
  );
  invalidateCache("/optimizations");
  return res;
}

export function profileDataset(payload: ProfileDatasetRequest) {
  return request<ProfileDatasetResponse>("/datasets/profile", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stageDatasetForAgent(payload: {
  dataset: Array<Record<string, unknown>>;
  dataset_filename: string;
}) {
  return request<{ staged_dataset_id: string; row_count: number }>("/datasets/stage-for-agent", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface StagedDatasetResponse {
  staged_dataset_id: string;
  columns: string[];
  rows: Array<Record<string, unknown>>;
  row_count: number;
}

/**
 * Fetch the rows a chat-side upload staged, by id, so the /submit wizard can
 * mirror the exact dataset the agent is working with. The shared wizard state
 * only carries the opaque `staged_dataset_id`; this materialises its rows.
 */
export function getStagedDataset(stagedDatasetId: string) {
  return request<StagedDatasetResponse>(`/datasets/staged/${encodeURIComponent(stagedDatasetId)}`);
}

export function validateCode(payload: {
  signature_code?: string;
  metric_code?: string;
  column_mapping: ColumnMapping;
  sample_row: Record<string, unknown>;
  optimizer_name?: string;
  module_name?: string;
}) {
  return request<ValidateCodeResponse>("/validate-code", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function validateDataset(payload: ValidateDatasetRequest) {
  return request<ValidateDatasetResponse>("/datasets/validate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getQueueStatus() {
  return cachedGet<QueueStatusResponse>("/queue", QUEUE_CACHE_MS);
}

export interface SidebarJobItem {
  optimization_id: string;
  status: string;
  name?: string | null;
  module_name?: string | null;
  optimizer_name?: string | null;
  model_name?: string | null;
  username?: string | null;
  created_at?: string | null;
  pinned?: boolean;
  optimization_type?: string | null;
  total_pairs?: number | null;
  completed_pairs?: number | null;
  failed_pairs?: number | null;
  /** True when this run stopped mid-optimization and can be resumed in place; drives Resume vs Restart. */
  resumable?: boolean;
  /** Caller's share role on a "shared with me" item; absent on own optimizations. */
  role?: ShareRole | null;
}

export function listJobsSidebar(params?: { username?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.username) q.set("username", params.username);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return cachedGet<{ items: SidebarJobItem[]; total: number }>(
    `/optimizations/sidebar${qs ? `?${qs}` : ""}`,
    SIDEBAR_CACHE_MS,
  );
}

/** List optimizations another user shared with the caller (Drive-style). */
export function listJobsSharedWithMe(params?: { limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return cachedGet<{ items: SidebarJobItem[]; total: number }>(
    `/optimizations/shared-with-me${qs ? `?${qs}` : ""}`,
    SIDEBAR_CACHE_MS,
  );
}

export function getServeInfo(optimizationId: string) {
  return request<ServeInfoResponse>(`/serve/${optimizationId}/info`);
}

export function getPairServeInfo(optimizationId: string, pairIndex: number) {
  return request<ServeInfoResponse>(`/serve/${optimizationId}/pair/${pairIndex}/info`);
}

export function getPairTestResults(optimizationId: string, pairIndex: number) {
  return request<{
    baseline: EvalExampleResult[];
    optimized: EvalExampleResult[];
  }>(`/optimizations/${optimizationId}/pair/${pairIndex}/test-results`);
}

export function serveProgram(optimizationId: string, inputs: Record<string, string>) {
  return request<ServeResponse>(`/serve/${optimizationId}`, {
    method: "POST",
    body: JSON.stringify({ inputs }),
  });
}

export interface StreamServeHandlers {
  onToken: (field: string, chunk: string) => void;
  onFinal: (result: {
    outputs: Record<string, unknown>;
    model_used: string;
    input_fields: string[];
    output_fields: string[];
  }) => void;
  onError: (message: string) => void;
  signal?: AbortSignal;
}

/** Stream program inference via SSE. Calls handlers as tokens arrive. */
export async function serveProgramStream(
  optimizationId: string,
  inputs: Record<string, string>,
  handlers: StreamServeHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetchWithAuthRetry(`${apiBase()}/serve/${optimizationId}/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ inputs }),
      signal: handlers.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError(msg("auto.shared.lib.api.literal.4"));
    return;
  }
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    handlers.onError(
      parseErrorMessage(text) ?? formatMsg("auto.shared.lib.api.template.3", { p1: res.status }),
    );
    return;
  }
  const processEvent = ({ event, data }: ServerSentEvent) => {
    if (event === "token") {
      handlers.onToken(String(data.field ?? ""), String(data.chunk ?? ""));
    } else if (event === "final") {
      handlers.onFinal({
        outputs: (data.outputs as Record<string, unknown>) ?? {},
        model_used: String(data.model_used ?? ""),
        input_fields: (data.input_fields as string[]) ?? [],
        output_fields: (data.output_fields as string[]) ?? [],
      });
    } else if (event === "error") {
      handlers.onError(String(data.error ?? msg("auto.shared.lib.api.literal.5")));
    }
  };
  try {
    await readServerSentEvents(res.body, processEvent);
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      handlers.onError(err instanceof Error ? err.message : msg("auto.shared.lib.api.literal.6"));
    }
  }
}

/** Stream pair program inference via SSE. Same as serveProgramStream but for a specific grid pair. */
export async function servePairProgramStream(
  optimizationId: string,
  pairIndex: number,
  inputs: Record<string, string>,
  handlers: StreamServeHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetchWithAuthRetry(
      `${apiBase()}/serve/${optimizationId}/pair/${pairIndex}/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ inputs }),
        signal: handlers.signal,
      },
    );
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError(msg("auto.shared.lib.api.literal.7"));
    return;
  }
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    handlers.onError(
      parseErrorMessage(text) ?? formatMsg("auto.shared.lib.api.template.4", { p1: res.status }),
    );
    return;
  }
  const processEvent = ({ event, data }: ServerSentEvent) => {
    if (event === "token") {
      handlers.onToken(String(data.field ?? ""), String(data.chunk ?? ""));
    } else if (event === "final") {
      handlers.onFinal({
        outputs: (data.outputs as Record<string, unknown>) ?? {},
        model_used: String(data.model_used ?? ""),
        input_fields: (data.input_fields as string[]) ?? [],
        output_fields: (data.output_fields as string[]) ?? [],
      });
    } else if (event === "error") {
      handlers.onError(String(data.error ?? msg("auto.shared.lib.api.literal.8")));
    }
  };
  try {
    await readServerSentEvents(res.body, processEvent);
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      handlers.onError(err instanceof Error ? err.message : msg("auto.shared.lib.api.literal.9"));
    }
  }
}

export interface CodeAgentChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface CodeAgentRequest {
  dataset_columns: string[];
  // Plain string roles (input/output/ignore); the code agent only consumes
  // the input/output roles.
  column_roles: Record<string, string>;
  column_kinds?: Record<string, "text" | "image">;
  sample_rows: Array<Record<string, unknown>>;
  user_message?: string;
  chat_history?: CodeAgentChatTurn[];
  prior_signature?: string;
  prior_metric?: string;
  prior_signature_validation?: string;
  prior_metric_validation?: string;
  initial_signature?: string;
  initial_metric?: string;
}

export type CodeAgentToolName = "edit_signature" | "edit_metric";

export interface CodeAgentToolStart {
  id: string;
  tool: CodeAgentToolName;
  reason: string;
}

export interface CodeAgentToolEnd {
  id: string;
  tool: CodeAgentToolName;
  status: string;
}

export interface CodeAgentHandlers {
  onSignaturePatch: (chunk: string) => void;
  onMetricPatch: (chunk: string) => void;
  onReasoningPatch?: (chunk: string) => void;
  onMessagePatch?: (chunk: string) => void;
  onSignatureReplace?: (code: string) => void;
  onMetricReplace?: (code: string) => void;
  onToolStart?: (ev: CodeAgentToolStart) => void;
  onToolEnd?: (ev: CodeAgentToolEnd) => void;
  onDone: (result: {
    signature_code: string;
    metric_code: string;
    assistant_message: string;
    model: string | null;
    /**
     * Seed-path validation outcome. The seed runner validates (and repairs)
     * the generated code; these flags are absent on the chat path (where
     * ``edit_signature``/``edit_metric`` already validate every edit), so a
     * missing flag is treated as valid.
     */
    signatureValid?: boolean;
    metricValid?: boolean;
    validationError?: string | null;
  }) => void;
  onError: (message: string) => void;
  signal?: AbortSignal;
}

/** Stream AI-generated signature + metric code via SSE. */
export async function streamCodeAgent(
  req: CodeAgentRequest,
  handlers: CodeAgentHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetchWithAuthRetry(`${apiBase()}/optimizations/ai-generate-code`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(req),
      signal: handlers.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError(msg("auto.shared.lib.api.literal.11"));
    return;
  }
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    handlers.onError(
      parseErrorMessage(text) ?? formatMsg("auto.shared.lib.api.template.5", { p1: res.status }),
    );
    return;
  }
  const processEvent = ({ event, data }: ServerSentEvent) => {
    if (event === "signature_patch") {
      handlers.onSignaturePatch(String(data.chunk ?? ""));
    } else if (event === "metric_patch") {
      handlers.onMetricPatch(String(data.chunk ?? ""));
    } else if (event === "reasoning_patch") {
      handlers.onReasoningPatch?.(String(data.chunk ?? ""));
    } else if (event === "message_patch") {
      handlers.onMessagePatch?.(String(data.chunk ?? ""));
    } else if (event === "signature_replace") {
      handlers.onSignatureReplace?.(String(data.code ?? ""));
    } else if (event === "metric_replace") {
      handlers.onMetricReplace?.(String(data.code ?? ""));
    } else if (event === "tool_start") {
      const tool = String(data.tool ?? "");
      if (tool === "edit_signature" || tool === "edit_metric") {
        handlers.onToolStart?.({
          id: String(data.id ?? ""),
          tool,
          reason: String(data.reason ?? ""),
        });
      }
    } else if (event === "tool_end") {
      const tool = String(data.tool ?? "");
      if (tool === "edit_signature" || tool === "edit_metric") {
        handlers.onToolEnd?.({
          id: String(data.id ?? ""),
          tool,
          status: String(data.status ?? "ok"),
        });
      }
    } else if (event === "done") {
      const rawModel = data.model;
      handlers.onDone({
        signature_code: String(data.signature_code ?? ""),
        metric_code: String(data.metric_code ?? ""),
        assistant_message: String(data.assistant_message ?? ""),
        model: typeof rawModel === "string" && rawModel.length > 0 ? rawModel : null,
        // Absent on the chat path → treat as valid; the seed path sends
        // explicit booleans after its validate-and-repair pass.
        signatureValid: data.signature_valid !== false,
        metricValid: data.metric_valid !== false,
        validationError: typeof data.validation_error === "string" ? data.validation_error : null,
      });
    } else if (event === "error") {
      handlers.onError(String(data.error ?? msg("auto.shared.lib.api.literal.12")));
    }
  };
  try {
    await readServerSentEvents(res.body, processEvent);
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      handlers.onError(err instanceof Error ? err.message : msg("auto.shared.lib.api.literal.10"));
    }
  }
}

export interface PublicDashboardPoint {
  optimization_id: string;
  optimization_type: string | null;
  winning_model: string | null;
  baseline_metric: number | null;
  optimized_metric: number | null;
  summary_text: string | null;
  task_name: string | null;
  module_name: string | null;
  optimizer_name: string | null;
  created_at: string | null;
  siblings: string[];
  task_fingerprint: string | null;
  compare_fingerprint: string | null;
}

export interface PublicDashboardResponse {
  points: PublicDashboardPoint[];
}

export function getPublicDashboard(): Promise<PublicDashboardResponse> {
  return cachedGet("/dashboard/public", 15000);
}

export interface CorpusFacets {
  models: string[];
  optimizers: string[];
  modules: string[];
}

/**
 * Distinct filter options (models / optimizers / modules) present in one
 * corpus scope, so each /explore tab offers exactly the chips it can filter
 * to. Pass no scope for the public archive; pass `owner_username` for the
 * caller's own runs or `shared_with_username` for runs shared with them (the
 * backend requires the bearer token to match the requested username).
 */
export function getCorpusFacets(
  scope: { owner_username?: string; shared_with_username?: string } = {},
): Promise<CorpusFacets> {
  const params = new URLSearchParams();
  if (scope.owner_username) params.set("owner_username", scope.owner_username);
  else if (scope.shared_with_username)
    params.set("shared_with_username", scope.shared_with_username);
  const qs = params.toString();
  return cachedGet(`/dashboard/facets${qs ? `?${qs}` : ""}`, 15000);
}

export type SearchSort = "relevance" | "recent" | "gain";

export interface SearchFilters {
  query?: string;
  models?: string[];
  optimizers?: string[];
  optimization_types?: string[];
  tasks?: string[];
  modules?: string[];
  date_from?: string; // ISO date (YYYY-MM-DD)
  date_to?: string; // ISO date (YYYY-MM-DD)
  sort?: SearchSort;
  page?: number;
  size?: number;
  /**
   * Scope the search to the caller's own jobs (including private rows). The
   * backend requires the bearer token to match this username, so only the
   * logged-in user can set this to their own name.
   */
  owner_username?: string;
  /**
   * Scope the search to runs shared with the caller via a member grant
   * (mutually exclusive with `owner_username`). Same session-match
   * requirement: only the logged-in user can set this to their own name.
   */
  shared_with_username?: string;
}

export interface SearchResult {
  optimization_id: string;
  optimization_type: string | null;
  winning_model: string | null;
  baseline_metric: number | null;
  optimized_metric: number | null;
  summary_text: string | null;
  task_name: string | null;
  module_name: string | null;
  optimizer_name: string | null;
  created_at: string | null;
  relevance: number | null;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  matched_ids: string[];
  /** Which backend branch served the response — drives per-row source badges. */
  search_type?: "semantic" | "lexical";
}

export function searchPublicDashboard(
  filters: SearchFilters,
  init?: { signal?: AbortSignal },
): Promise<SearchResponse> {
  return request<SearchResponse>("/dashboard/search", {
    method: "POST",
    body: JSON.stringify(filters),
    signal: init?.signal,
  });
}

export interface PopularQuery {
  query: string;
  count: number;
}

export interface PopularQueriesResponse {
  queries: PopularQuery[];
}

/** Trending public-corpus search queries, ranked by frequency over a recent window. */
export function getPopularQueries(): Promise<PopularQueriesResponse> {
  return cachedGet("/dashboard/search/popular", 60000);
}

/**
 * Record an explicitly-committed public search query for trending. Fire-and-
 * forget: only call on an explicit commit (Enter / opening a result), never on
 * debounced typing, and never for the "mine" corpus.
 *
 * Uses ``keepalive`` so the request still completes when the result-open click
 * navigates the page away mid-flight. The 204 response is ignored and all
 * errors are swallowed — a failed log never affects the user.
 */
export function logSearchQuery(query: string): void {
  try {
    void fetch(`${apiBase()}/dashboard/search/log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      keepalive: true,
    }).catch(() => {
      // Best-effort telemetry; ignore network failures.
    });
  } catch {
    // Ignore synchronous fetch construction errors.
  }
}
