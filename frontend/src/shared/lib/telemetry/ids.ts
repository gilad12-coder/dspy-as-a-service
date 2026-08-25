/**
 * Opaque, PII-free identifiers for telemetry.
 *
 * `anonymous_id` is stable per browser (localStorage) so a person's activity can
 * be funnel-counted across visits — and so pre-login events line up with the
 * same browser's post-login events — without ever storing who they are.
 * `session_id` is per-tab (sessionStorage), scoping a single visit. Both are
 * random UUIDs with no derivation from anything personal. Storage access is
 * wrapped because private-mode browsers throw on `localStorage`; the fallback is
 * an empty id (the event is still sent, just unlinked).
 */

const ANON_KEY = "skynet.telemetry.anon";
const SESSION_KEY = "skynet.telemetry.session";

/** Return a random UUID, falling back to a manual v4 where `crypto` is absent. */
function uuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Read-or-create a persistent value in a Web Storage area, "" if blocked. */
function readOrCreate(storage: () => Storage, key: string): string {
  if (typeof window === "undefined") return "";
  try {
    const store = storage();
    let id = store.getItem(key);
    if (!id) {
      id = uuid();
      store.setItem(key, id);
    }
    return id;
  } catch {
    return "";
  }
}

/** The stable per-browser anonymous id (created on first read). */
export function getAnonymousId(): string {
  return readOrCreate(() => localStorage, ANON_KEY);
}

/** The per-tab session id (created on first read). */
export function getSessionId(): string {
  return readOrCreate(() => sessionStorage, SESSION_KEY);
}
