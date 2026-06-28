/**
 * First-party telemetry client — a small, lossy, best-effort event pipeline.
 *
 * `track()` enqueues an event; the queue flushes when it reaches `BATCH_SIZE`,
 * on a `FLUSH_INTERVAL_MS` timer, or when the tab is hidden/closed (so a closing
 * tab still delivers, via `keepalive`). Sending goes through `postTelemetry` in
 * the API layer, which attaches the bearer token when present — so attribution
 * to a signed-in user is handled there, and this module never needs to know who
 * the user is. Identity in events is therefore *only* the opaque anonymous id.
 *
 * Privacy: respects Do-Not-Track and a local opt-out, both of which make
 * `track()` a no-op. Everything is guarded for SSR — on the server it is inert.
 */

import { postTelemetry } from "@/shared/lib/api";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { getAnonymousId, getSessionId } from "./ids";

const BATCH_SIZE = 20;
const FLUSH_INTERVAL_MS = 10_000;
// Hard cap so a tab that goes offline can't grow the queue without bound; the
// oldest events are dropped first (telemetry is lossy by design).
const MAX_QUEUE = 200;
const OPT_OUT_KEY = "skynet.telemetry.optout";

interface QueuedEvent {
  name: string;
  ts: number;
  path?: string;
  locale?: string;
  app_version?: string;
  properties?: Record<string, unknown>;
  context?: Record<string, unknown>;
}

// Do-Not-Track can't change within a page lifetime, so resolve it once. The
// header spelling differs across browsers; "1" is the universal "on" value.
const _dnt =
  (typeof navigator !== "undefined" && navigator.doNotTrack === "1") ||
  (typeof window !== "undefined" &&
    (window as unknown as { doNotTrack?: string }).doNotTrack === "1");

let _queue: QueuedEvent[] = [];
let _flushTimer: ReturnType<typeof setTimeout> | null = null;
let _listenersBound = false;
let _optedOut = readStoredOptOut();

/** Read the persisted opt-out flag, defaulting to opted-in when unreadable. */
function readStoredOptOut(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return localStorage.getItem(OPT_OUT_KEY) === "1";
  } catch {
    return false;
  }
}

/** Whether telemetry may run right now (client-side, not DNT, not opted out). */
function enabled(): boolean {
  return typeof window !== "undefined" && !_dnt && !_optedOut;
}

/** Current UI locale, or undefined if the runtime locale isn't resolvable yet. */
function safeLocale(): string | undefined {
  try {
    return getActiveLocale();
  } catch {
    return undefined;
  }
}

/**
 * Record one interaction event.
 *
 * No-op when telemetry is disabled (SSR, Do-Not-Track, or opt-out). `properties`
 * and `context` must stay PII-free — they are stored verbatim. The event is
 * stamped with the current path, locale, app version, and client time here.
 */
export function track(
  name: string,
  properties?: Record<string, unknown>,
  context?: Record<string, unknown>,
): void {
  if (!enabled()) return;
  const event: QueuedEvent = {
    name,
    ts: Date.now(),
    path: window.location.pathname,
    locale: safeLocale(),
    app_version: getRuntimeEnv().appVersion,
    properties,
    context,
  };
  _queue.push(event);
  if (_queue.length > MAX_QUEUE) _queue.splice(0, _queue.length - MAX_QUEUE);
  bindLifecycleListeners();
  if (_queue.length >= BATCH_SIZE) {
    flush();
  } else {
    scheduleFlush();
  }
}

/** Send everything queued now, clearing the pending flush timer. */
export function flush(): void {
  if (_flushTimer !== null) {
    clearTimeout(_flushTimer);
    _flushTimer = null;
  }
  if (_queue.length === 0) return;
  const events = _queue;
  _queue = [];
  postTelemetry({
    anonymous_id: getAnonymousId() || undefined,
    session_id: getSessionId() || undefined,
    events,
  });
}

/** Arm the debounced interval flush if one isn't already pending. */
function scheduleFlush(): void {
  if (_flushTimer !== null) return;
  _flushTimer = setTimeout(() => {
    _flushTimer = null;
    flush();
  }, FLUSH_INTERVAL_MS);
}

/** Bind the tab-leaving flush signals exactly once. */
function bindLifecycleListeners(): void {
  if (_listenersBound || typeof document === "undefined") return;
  _listenersBound = true;
  // visibilitychange→hidden fires reliably when a tab is backgrounded or
  // closed on mobile (where pagehide/unload often don't); pagehide covers the
  // desktop navigate-away. Both flush with keepalive so the queue still lands.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flush();
  });
  window.addEventListener("pagehide", () => flush());
}

/**
 * Turn telemetry off (or back on) and persist the choice. Turning it off also
 * drops anything queued so a just-opted-out user sends nothing further. Exposed
 * for a future privacy control in settings.
 */
export function setTelemetryOptOut(optedOut: boolean): void {
  _optedOut = optedOut;
  if (optedOut) _queue = [];
  try {
    if (optedOut) localStorage.setItem(OPT_OUT_KEY, "1");
    else localStorage.removeItem(OPT_OUT_KEY);
  } catch {
    /* storage blocked — the in-memory flag still applies for this session */
  }
}

/** Whether telemetry is currently suppressed (Do-Not-Track or explicit opt-out). */
export function isTelemetryOptedOut(): boolean {
  return _dnt || _optedOut;
}
