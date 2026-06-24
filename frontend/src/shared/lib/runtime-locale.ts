/**
 * Sync active-locale resolution for the argument-free `msg()` / `tI18n()`
 * helpers, across the RSC server/client boundary.
 *
 * `msg("some.key")` is called from ~2,250 sites with no locale argument, so the
 * active locale has to be ambient rather than threaded. The two runtimes need
 * different mechanisms:
 *
 * - **Client**: a module-level variable seeded from the `window.__SKYNET_LOCALE__`
 *   shim the layout injects before hydration. The browser is single-threaded,
 *   so a mutable module global is race-free; the switcher updates it in place.
 * - **Server**: a per-request slot held by React's `cache()`, which returns a
 *   fresh object per request render. The `force-dynamic` root layout sets it
 *   once at the top of render (before any child resolves a message), so even
 *   concurrent requests for different locales never read each other's value.
 *
 * Mirrors the `runtime-env.ts` injection shim so the two `<script>` tags share
 * one shape.
 */

import { cache } from "react";
import { DEFAULT_LOCALE, dirForLocale, isLocale, type Locale } from "@/shared/lib/locale";

declare global {
  interface Window {
    __SKYNET_LOCALE__?: Locale;
  }
}

let clientLocale: Locale | null = null;

// React `cache()` gives one object per server request render; mutating
// `.current` scopes the locale to that request without cross-request bleed. The
// wrapper is created lazily so cache() is only ever invoked during a server
// render — never when this module loads in the client bundle.
let serverLocaleSlot: (() => { current: Locale }) | null = null;

function getServerLocaleSlot(): { current: Locale } {
  serverLocaleSlot ??= cache((): { current: Locale } => ({ current: DEFAULT_LOCALE }));
  return serverLocaleSlot();
}

/**
 * Pin the active locale for the current server request. Call once at the top of
 * the root layout (and `generateMetadata`) before any descendant resolves a
 * message. No-op semantics on the client, which seeds from the injected shim.
 */
export function setServerLocale(locale: Locale): void {
  getServerLocaleSlot().current = locale;
}

/**
 * Update the client's active locale in place. The switcher calls this right
 * before it remounts the tree so the re-render resolves messages in the new
 * locale.
 */
export function setClientLocale(locale: Locale): void {
  clientLocale = locale;
}

/**
 * Resolve the locale every `msg()` / `tI18n()` call should render in.
 *
 * Server and client components are separate module bundles that do NOT share
 * state, even during SSR. So resolution branches by which bundle is asking:
 *
 * - **Client bundle** (browser AND client-component SSR): `clientLocale`, which
 *   LocaleProvider seeds before any descendant renders. Preferred whenever set,
 *   because during SSR `window` is undefined yet we still must use the request
 *   locale — the server slot lives in the *other* bundle and would read default.
 * - **Server bundle** (Server Components, `window` undefined, `clientLocale`
 *   never set because the provider only runs client-side): the per-request
 *   `cache()` slot the root layout pins.
 * - **Client first paint** before the provider mounts: seed from the injected
 *   `window.__SKYNET_LOCALE__` shim.
 *
 * Caveat: `clientLocale` is a module global, so under concurrent SSR of mixed
 * locales a client-component render could briefly observe another request's
 * value; the browser then corrects it on hydration from the per-request shim.
 * Acceptable for a Hebrew-default app where English is opt-in.
 */
export function getActiveLocale(): Locale {
  if (clientLocale !== null) return clientLocale;
  if (typeof window === "undefined") {
    return getServerLocaleSlot().current;
  }
  const injected = window.__SKYNET_LOCALE__;
  clientLocale = isLocale(injected) ? injected : DEFAULT_LOCALE;
  return clientLocale;
}

/**
 * Writing direction for the active locale — a one-call convenience for the many
 * `dir={...}` attributes that must follow the locale (RTL for Hebrew, LTR for
 * English) rather than being pinned to one direction. Safe in both bundles since
 * it delegates to `getActiveLocale()`.
 */
export function getActiveDir(): "rtl" | "ltr" {
  return dirForLocale(getActiveLocale());
}

/**
 * Inline `<script>` body that sets `window.__SKYNET_LOCALE__` before hydration,
 * so the first client render resolves messages in the request's locale. The
 * value is a single quoted token, so the XSS surface of `runtime-env.ts` (where
 * values can be arbitrary URLs) does not apply — `JSON.stringify` of a known
 * locale literal is sufficient.
 */
export function serializeLocale(locale: Locale): string {
  return `window.__SKYNET_LOCALE__=${JSON.stringify(locale)};`;
}
